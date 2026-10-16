"""End-to-end experiment: data -> CV model selection -> test evaluation -> cold start -> reports."""
from __future__ import annotations

import json
import logging
from itertools import product

import joblib
import numpy as np

from . import plots
from .config import Config
from .data import Split, generate, make_split
from .evaluate import ranking_metrics, rmse_mae
from .features import item_features
from .models import ALSRecommender, BiasModel, ContentColdStart, ItemKNN, PopularityRecommender, SVDRecommender

log = logging.getLogger(__name__)


def _cv_rmse(make, R: np.ndarray, M: np.ndarray, folds: int, seed: int) -> float:
    """K-fold CV over observed ratings; returns mean validation RMSE."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, folds, size=M.shape)
    out = []
    for f in range(folds):
        val = M & (fold == f)
        model = make().fit(R, M & ~val)
        out.append(rmse_mae(model.predict_matrix(), R, val)["rmse"])
    return float(np.mean(out))


def select_hyperparameters(sp: Split, cfg: Config) -> dict:
    """Grid-search each tunable model by CV RMSE on the training ratings."""
    R, M = sp.R_train, sp.M_train
    als = np.array([[_cv_rmse(lambda f=f, r=r: ALSRecommender(f, r, iters=10, seed=cfg.seed), R, M, cfg.cv_folds, cfg.seed)
                     for r in cfg.als_reg_grid] for f in cfg.als_factors_grid])
    svd = {k: _cv_rmse(lambda k=k: SVDRecommender(k, cfg.seed), R, M, cfg.cv_folds, cfg.seed) for k in cfg.svd_k_grid}
    knn = {k: _cv_rmse(lambda k=k: ItemKNN(k), R, M, cfg.cv_folds, cfg.seed) for k in cfg.knn_k_grid}
    fi, ri = np.unravel_index(als.argmin(), als.shape)
    return {
        "als_grid": als, "svd": svd, "knn": knn,
        "als_best": {"factors": cfg.als_factors_grid[fi], "reg": cfg.als_reg_grid[ri], "cv_rmse": float(als.min())},
        "svd_best": {"k": min(svd, key=svd.get), "cv_rmse": float(min(svd.values()))},
        "knn_best": {"k": min(knn, key=knn.get), "cv_rmse": float(min(knn.values()))},
    }


def blend_scores(pred: np.ndarray, count: np.ndarray, w: float) -> np.ndarray:
    """Hybrid ranking score: z-scored predicted rating + w * z-scored log item popularity."""
    z = (pred - pred.mean()) / pred.std()
    lp = np.log1p(count)
    return z + w * ((lp - lp.mean()) / lp.std())[None, :]


def select_blend_weight(sp: Split, cfg: Config, als_params: dict, warm_items: np.ndarray,
                        grid: tuple = (0.0, 0.25, 0.5, 1.0, 2.0)) -> tuple[float, dict]:
    """Choose the popularity weight on a validation slice carved out of the training ratings."""
    rng = np.random.default_rng(cfg.seed + 3)
    val = sp.M_train & (rng.random(sp.M_train.shape) < 0.15)
    tr = sp.M_train & ~val
    m = ALSRecommender(als_params["factors"], als_params["reg"], iters=12, seed=cfg.seed).fit(sp.R_train, tr)
    pred, cnt = m.predict_matrix(), tr.sum(0)
    scores = {}
    for w in grid:
        scores[w] = ranking_metrics(blend_scores(pred, cnt, w), tr, sp.R_train, val, sp.warm_users,
                                    candidates=warm_items, k=cfg.top_k, thresh=cfg.relevance_threshold)[f"ndcg@{cfg.top_k}"]
    return max(scores, key=scores.get), {str(k): v for k, v in scores.items()}


def evaluate_cold_users(als: ALSRecommender, pop: PopularityRecommender, sp: Split, cfg: Config) -> dict:
    """Reveal k ratings of each cold user, fold in, rank warm items by predicted rating vs popularity."""
    rng = np.random.default_rng(cfg.seed + 5)
    warm_items = np.ones(sp.n_items, dtype=bool)
    warm_items[sp.cold_items] = False
    res: dict[str, list[float]] = {"als_fold_in": [], "popularity": []}
    for k in cfg.cold_user_reveals:
        S_als = np.zeros((sp.n_users, sp.n_items))
        revealed = np.zeros_like(sp.M_cold_user)
        for u in sp.cold_users:
            items = np.nonzero(sp.M_cold_user[u])[0]
            pick = rng.choice(items, size=min(k, len(items)), replace=False)
            revealed[u, pick] = True
            b, p = als.fold_in(pick, sp.R_cold_user[u, pick])
            S_als[u] = als.mu + b + als.bi + als.Q @ p
        test_M = sp.M_cold_user & ~revealed
        kw = dict(exclude=revealed, R_test=sp.R_cold_user, M_test=test_M, users=sp.cold_users,
                  candidates=warm_items, k=cfg.top_k, thresh=cfg.relevance_threshold)
        res["als_fold_in"].append(ranking_metrics(S_als, **kw)[f"ndcg@{cfg.top_k}"])
        res["popularity"].append(ranking_metrics(pop.rank_scores(), **kw)[f"ndcg@{cfg.top_k}"])
    return res


def evaluate_cold_items(als: ALSRecommender, sp: Split, X: np.ndarray, cfg: Config) -> dict:
    """New items (no ratings): content-ridge latent prediction vs genre-mean vs random."""
    ci = sp.cold_items
    cs = ContentColdStart().fit(als, X, sp.M_train.sum(0))
    content = cs.score_new_items(X[ci])
    # genre-mean baseline: average of warm item mean ratings sharing a genre
    G = X[:, :-1]
    warm = sp.M_train.sum(0) >= 5
    item_mean = als.mu + als.bi
    genre_mean = (G[warm].T @ item_mean[warm]) / np.maximum(G[warm].sum(0), 1)
    gm = (G[ci] @ genre_mean) / np.maximum(G[ci].sum(1), 1)
    rng = np.random.default_rng(cfg.seed + 9)
    scores = {"random": rng.random((sp.n_users, len(ci))), "genre_mean": np.tile(gm, (sp.n_users, 1)), "content_ridge_mf": content}
    out = {}
    Rc, Mc = sp.R_cold_item[:, ci], sp.M_cold_item[:, ci]
    for name, s in scores.items():
        m = ranking_metrics(s, np.zeros_like(Mc), Rc, Mc, sp.warm_users, k=cfg.top_k, thresh=cfg.relevance_threshold)
        out[name] = {f"ndcg@{cfg.top_k}": m[f"ndcg@{cfg.top_k}"], f"recall@{cfg.top_k}": m[f"recall@{cfg.top_k}"]}
    out["content_ridge_mf"].update(rmse_mae(content, Rc, Mc))
    out["genre_mean"].update(rmse_mae(np.tile(gm, (sp.n_users, 1)), Rc, Mc))
    return out


def run(cfg: Config | None = None, data_only: bool = False) -> dict:
    """Run the whole pipeline and write data, figures, model and metrics.json."""
    cfg = cfg or Config()
    for d in (cfg.data_dir, cfg.reports_dir / "figures", cfg.models_dir):
        d.mkdir(parents=True, exist_ok=True)
    users, items, ratings = generate(cfg)
    ratings.to_csv(cfg.data_dir / "ratings.csv", index=False)
    items.to_csv(cfg.data_dir / "items.csv", index=False)
    if data_only:
        return {}
    sp = make_split(ratings, cfg)
    X = item_features(items)
    log.info("train=%d test=%d", sp.M_train.sum(), sp.M_test.sum())

    hp = select_hyperparameters(sp, cfg)
    log.info("selected: %s", {k: v for k, v in hp.items() if k.endswith("best")})

    models = {
        "popularity": PopularityRecommender(),
        "bias_baseline": BiasModel(),
        "item_knn": ItemKNN(hp["knn_best"]["k"]),
        "svd": SVDRecommender(hp["svd_best"]["k"], cfg.seed),
        "als": ALSRecommender(hp["als_best"]["factors"], hp["als_best"]["reg"], iters=15, seed=cfg.seed),
    }
    warm_items = np.ones(cfg.n_items, dtype=bool)
    warm_items[sp.cold_items] = False
    results: dict[str, dict] = {}
    for name, m in models.items():
        m.fit(sp.R_train, sp.M_train)
        r = rmse_mae(m.predict_matrix(), sp.R_test, sp.M_test)
        r.update(ranking_metrics(m.rank_scores(), sp.M_train, sp.R_test, sp.M_test, sp.warm_users,
                                 candidates=warm_items, k=cfg.top_k, thresh=cfg.relevance_threshold))
        results[name] = r
        log.info("%s %s", name, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)})
    w, w_scores = select_blend_weight(sp, cfg, hp["als_best"], warm_items)
    blend = ranking_metrics(blend_scores(models["als"].predict_matrix(), sp.M_train.sum(0), w), sp.M_train, sp.R_test,
                            sp.M_test, sp.warm_users, candidates=warm_items, k=cfg.top_k, thresh=cfg.relevance_threshold)
    blend.update(rmse_mae(models["als"].predict_matrix(), sp.R_test, sp.M_test))
    blend["pop_weight"] = w
    results["als_pop_blend"] = blend
    for r in results.values():
        r["rmse_pct_vs_popularity"] = float(100 * (results["popularity"]["rmse"] - r["rmse"]) / results["popularity"]["rmse"])
    results["global_mean"] = rmse_mae(np.full(sp.R_test.shape, sp.R_train[sp.M_train].mean()), sp.R_test, sp.M_test)

    als = models["als"]
    cold_users = evaluate_cold_users(als, models["popularity"], sp, cfg)
    cold_items = evaluate_cold_items(als, sp, X, cfg)

    fig_dir = cfg.reports_dir / "figures"
    plots.plot_data_overview(sp.R_train, sp.M_train, fig_dir / "data_overview.png")
    plots.plot_model_comparison({k: v for k, v in results.items() if k != "global_mean"}, fig_dir / "model_comparison.png", cfg.top_k)
    plots.plot_cv_grid(hp["als_grid"], cfg.als_factors_grid, cfg.als_reg_grid, als.history, fig_dir / "als_cv_and_convergence.png")
    plots.plot_cold_start(cold_users, list(cfg.cold_user_reveals),
                          {k: v[f"ndcg@{cfg.top_k}"] for k, v in cold_items.items()}, fig_dir / "cold_start.png")

    joblib.dump({"als": als, "popularity": models["popularity"], "train_mask": sp.M_train,
                 "cold_start": ContentColdStart().fit(als, X, sp.M_train.sum(0)), "item_features": X},
                cfg.models_dir / "als_model.joblib")

    metrics = {
        "dataset": {"type": "synthetic", "n_users": cfg.n_users, "n_items": cfg.n_items, "n_ratings": int(len(ratings)),
                    "density": float(len(ratings) / (cfg.n_users * cfg.n_items)), "n_train": int(sp.M_train.sum()),
                    "n_test": int(sp.M_test.sum()), "n_cold_items": cfg.n_cold_items, "n_cold_users": cfg.n_cold_users},
        "selected_hyperparameters": {k: v for k, v in hp.items() if k.endswith("best")},
        "cv": {"svd_rmse_by_k": {str(k): v for k, v in hp["svd"].items()},
               "knn_rmse_by_k": {str(k): v for k, v in hp["knn"].items()},
               "als_rmse_grid": hp["als_grid"].tolist()},
        "blend_weight_validation_ndcg": w_scores,
        "test": results,
        "cold_users_ndcg@10": {"reveals": list(cfg.cold_user_reveals), **cold_users},
        "cold_items": cold_items,
    }
    (cfg.reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics
