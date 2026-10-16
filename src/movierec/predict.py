"""Load the saved ALS model and produce top-N recommendations for a user, or for a new user from ratings."""
from __future__ import annotations

import argparse
import json

import joblib
import numpy as np

from .config import Config


def recommend(bundle: dict, user_id: int, n: int = 10) -> list[tuple[int, float]]:
    """Top-n unseen item ids with predicted ratings for an existing user."""
    als, seen = bundle["als"], bundle["train_mask"][user_id]
    scores = als.predict_matrix()[user_id].copy()
    scores[seen] = -np.inf
    scores[bundle["cold_start"].als.Q.sum(1) == 0] = -np.inf  # items never rated in training
    top = np.argsort(-scores)[:n]
    return [(int(i), float(np.clip(scores[i], 1, 5))) for i in top]


def recommend_new_user(bundle: dict, rated: dict[int, float], n: int = 10) -> list[tuple[int, float]]:
    """Cold-start user: fold in a handful of (item_id -> rating) pairs; empty dict falls back to popularity."""
    als, pop = bundle["als"], bundle["popularity"]
    if not rated:
        top = np.argsort(-pop.count)[:n]
        return [(int(i), float(pop.item_mean[i])) for i in top]
    idx = np.array(list(rated))
    b, p = als.fold_in(idx, np.array(list(rated.values()), dtype=float))
    s = als.mu + b + als.bi + als.Q @ p
    s[idx] = -np.inf
    top = np.argsort(-s)[:n]
    return [(int(i), float(np.clip(s[i], 1, 5))) for i in top]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", type=int, help="existing user id")
    ap.add_argument("--rated", type=str, help='new user ratings as JSON, e.g. \'{"12": 5, "40": 4}\'')
    ap.add_argument("-n", type=int, default=10)
    a = ap.parse_args()
    bundle = joblib.load(Config().models_dir / "als_model.joblib")
    if a.rated is not None:
        recs = recommend_new_user(bundle, {int(k): float(v) for k, v in json.loads(a.rated).items()}, a.n)
    else:
        recs = recommend(bundle, a.user or 0, a.n)
    for i, s in recs:
        print(f"item {i:4d}  predicted {s:.2f}")


if __name__ == "__main__":
    main()
