"""Rating (RMSE/MAE) and top-N ranking (NDCG@k, recall@k, coverage) metrics."""
from __future__ import annotations

import numpy as np


def rmse_mae(pred: np.ndarray, R: np.ndarray, M: np.ndarray, lo: float = 1.0, hi: float = 5.0) -> dict[str, float]:
    """RMSE / MAE on observed entries of M after clipping predictions to the rating range."""
    err = np.clip(pred, lo, hi)[M] - R[M]
    return {"rmse": float(np.sqrt((err ** 2).mean())), "mae": float(np.abs(err).mean())}


def ranking_metrics(scores: np.ndarray, exclude: np.ndarray, R_test: np.ndarray, M_test: np.ndarray,
                    users: np.ndarray, candidates: np.ndarray | None = None, k: int = 10,
                    thresh: float = 4.0) -> dict[str, float]:
    """NDCG@k (binary gain), recall@k and catalogue coverage over `users`.

    `exclude` (bool, users x items) removes already-seen items; `candidates` (bool, items)
    restricts the ranked pool. Users without a relevant held-out item are skipped.
    """
    S = scores[users].astype(float).copy()
    S[exclude[users]] = -np.inf
    if candidates is not None:
        S[:, ~candidates] = -np.inf
    rel = (M_test[users] & (R_test[users] >= thresh))
    kk = min(k, S.shape[1])
    top = np.argpartition(-S, kk - 1, axis=1)[:, :kk]
    order = np.argsort(-np.take_along_axis(S, top, 1), axis=1)
    top = np.take_along_axis(top, order, 1)
    discounts = 1.0 / np.log2(np.arange(kk) + 2)
    gains = np.take_along_axis(rel, top, 1).astype(float)
    dcg = (gains * discounts).sum(1)
    n_rel = rel.sum(1)
    idcg = np.array([discounts[: min(n, kk)].sum() for n in n_rel])
    valid = n_rel > 0
    return {
        f"ndcg@{k}": float((dcg[valid] / idcg[valid]).mean()),
        f"recall@{k}": float((gains.sum(1)[valid] / n_rel[valid]).mean()),
        "coverage": float(len(np.unique(top)) / (candidates.sum() if candidates is not None else S.shape[1])),
        "n_users": int(valid.sum()),
    }
