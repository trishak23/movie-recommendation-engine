"""Seeded synthetic rating generator with latent factors and train/test/cold-start splits."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

GENRES = ["action", "comedy", "drama", "thriller", "romance", "scifi",
          "horror", "animation", "documentary", "fantasy", "crime", "family"]


def generate(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generate (users, items, ratings) from a latent-factor model.

    rating = clip(round(mu + b_u + b_i + scale * <p_u, q_i> + noise), 1, 5)
    Item factors q_i are driven by the item's genres (so content is informative for
    cold start). Which ratings are observed depends on item popularity, user
    activity and (mildly) on affinity, i.e. exposure is not missing at random.
    """
    rng = np.random.default_rng(cfg.seed)
    nu, ni, d = cfg.n_users, cfg.n_items, cfg.latent_dim

    # items: 1-3 genres each, release year
    genre_prior = rng.dirichlet(np.ones(len(GENRES)) * 3)
    G = np.zeros((ni, len(GENRES)))
    for i in range(ni):
        n_g = rng.choice([1, 2, 3], p=[0.35, 0.45, 0.2])
        G[i, rng.choice(len(GENRES), size=n_g, replace=False, p=genre_prior)] = 1
    year = rng.integers(1980, 2025, size=ni)
    W = rng.normal(0, 1, size=(len(GENRES), d))
    Q = (G @ W) / np.sqrt(G.sum(1, keepdims=True)) * 0.8 + rng.normal(0, 0.4, size=(ni, d))
    P = rng.normal(0, 1, size=(nu, d))
    aff = P @ Q.T
    aff = (aff - aff.mean()) / aff.std()

    bu = rng.normal(0, 0.35, size=nu)
    bi = rng.normal(0, 0.4, size=ni)
    raw = 3.3 + bu[:, None] + bi[None, :] + 0.85 * aff + rng.normal(0, cfg.rating_noise, size=(nu, ni))
    rating = np.clip(np.rint(raw), 1, 5)

    pop = rng.normal(0, 1, size=ni)
    act = rng.normal(0, 1, size=nu)
    a = 0.9 * pop[None, :] + 0.9 * act[:, None] + 0.4 * aff
    p = np.exp(a)
    p = np.clip(p * cfg.density / p.mean(), 0, 0.6)
    obs = rng.random((nu, ni)) < p
    uu, ii = np.nonzero(obs)
    ratings = pd.DataFrame({"user_id": uu, "item_id": ii, "rating": rating[uu, ii].astype(int)})
    items = pd.DataFrame(G.astype(int), columns=[f"g_{g}" for g in GENRES])
    items.insert(0, "item_id", np.arange(ni))
    items["year"] = year
    users = pd.DataFrame({"user_id": np.arange(nu)})
    log.info("generated %d ratings (density %.3f)", len(ratings), len(ratings) / (nu * ni))
    return users, items, ratings


@dataclass
class Split:
    """Matrices (dense, full index space) and masks for every evaluation scenario."""

    n_users: int
    n_items: int
    R_train: np.ndarray
    M_train: np.ndarray
    R_test: np.ndarray
    M_test: np.ndarray
    cold_items: np.ndarray
    cold_users: np.ndarray
    warm_users: np.ndarray
    R_cold_item: np.ndarray      # warm users x cold items ratings (full shape, masked)
    M_cold_item: np.ndarray
    R_cold_user: np.ndarray      # cold users x warm items ratings
    M_cold_user: np.ndarray


def _dense(df: pd.DataFrame, nu: int, ni: int) -> tuple[np.ndarray, np.ndarray]:
    R = np.zeros((nu, ni))
    M = np.zeros((nu, ni), dtype=bool)
    R[df.user_id.values, df.item_id.values] = df.rating.values
    M[df.user_id.values, df.item_id.values] = True
    return R, M


def make_split(ratings: pd.DataFrame, cfg: Config) -> Split:
    """Warm random split (80/20), plus held-out cold items and cold users."""
    rng = np.random.default_rng(cfg.seed + 1)
    nu, ni = cfg.n_users, cfg.n_items
    cold_items = np.sort(rng.choice(ni, cfg.n_cold_items, replace=False))
    cold_users = np.sort(rng.choice(nu, cfg.n_cold_users, replace=False))
    ci, cu = np.isin(ratings.item_id, cold_items), np.isin(ratings.user_id, cold_users)
    warm = ratings[~ci & ~cu]
    is_test = rng.random(len(warm)) < cfg.test_frac
    R_tr, M_tr = _dense(warm[~is_test], nu, ni)
    R_te, M_te = _dense(warm[is_test], nu, ni)
    R_ci, M_ci = _dense(ratings[ci & ~cu], nu, ni)
    R_cu, M_cu = _dense(ratings[cu & ~ci], nu, ni)
    warm_users = np.setdiff1d(np.arange(nu), cold_users)
    return Split(nu, ni, R_tr, M_tr, R_te, M_te, cold_items, cold_users, warm_users,
                 R_ci, M_ci, R_cu, M_cu)
