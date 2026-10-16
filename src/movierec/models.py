"""Recommenders implemented on dense (users x items) matrices with observation masks."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import RidgeCV


def fit_baseline(R: np.ndarray, M: np.ndarray, lam_i: float = 10.0, lam_u: float = 15.0,
                 iters: int = 6) -> tuple[float, np.ndarray, np.ndarray]:
    """Regularised global mean + user bias + item bias via alternating shrunk means."""
    mu = float(R[M].mean())
    bu = np.zeros(R.shape[0])
    bi = np.zeros(R.shape[1])
    Mf = M.astype(float)
    for _ in range(iters):
        bi = ((R - mu - bu[:, None]) * Mf).sum(0) / (lam_i + Mf.sum(0))
        bu = ((R - mu - bi[None, :]) * Mf).sum(1) / (lam_u + Mf.sum(1))
    return mu, bu, bi


class PopularityRecommender:
    """Ranks by interaction count; rates with a damped item mean (the non-personalised baseline)."""

    name = "popularity"

    def fit(self, R: np.ndarray, M: np.ndarray) -> "PopularityRecommender":
        self.mu, _, _ = fit_baseline(R, M)
        n = M.sum(0)
        self.count = n.astype(float)
        self.item_mean = (self.mu * 5 + (R * M).sum(0)) / (5 + n)
        self.shape = R.shape
        return self

    def predict_matrix(self) -> np.ndarray:
        return np.tile(self.item_mean, (self.shape[0], 1))

    def rank_scores(self) -> np.ndarray:
        return np.tile(self.count, (self.shape[0], 1))


class BiasModel:
    """mu + b_u + b_i."""

    name = "bias_baseline"

    def fit(self, R: np.ndarray, M: np.ndarray) -> "BiasModel":
        self.mu, self.bu, self.bi = fit_baseline(R, M)
        return self

    def predict_matrix(self) -> np.ndarray:
        return self.mu + self.bu[:, None] + self.bi[None, :]

    def rank_scores(self) -> np.ndarray:
        return self.predict_matrix()


class ItemKNN:
    """Item-item CF: cosine over mean-centred residuals with shrinkage and top-k neighbours."""

    name = "item_knn"

    def __init__(self, k: int = 40, shrink: float = 10.0) -> None:
        self.k, self.shrink = k, shrink

    def fit(self, R: np.ndarray, M: np.ndarray) -> "ItemKNN":
        self.mu, self.bu, self.bi = fit_baseline(R, M)
        base = self.mu + self.bu[:, None] + self.bi[None, :]
        Mf = M.astype(float)
        self.resid = (R - base) * Mf
        self.M = Mf
        self.base = base
        num = self.resid.T @ self.resid
        norms = np.sqrt((self.resid ** 2).sum(0)) + 1e-9
        common = Mf.T @ Mf
        S = num / np.outer(norms, norms) * (common / (common + self.shrink))
        np.fill_diagonal(S, 0.0)
        S[S < 0] = 0.0
        # keep top-k neighbours per target item (row = target)
        k = min(self.k, S.shape[0] - 1)
        thresh = -np.partition(-S, k - 1, axis=1)[:, k - 1:k]
        self.S = np.where(S >= thresh, S, 0.0)
        return self

    def predict_matrix(self) -> np.ndarray:
        num = self.resid @ self.S.T
        den = self.M @ np.abs(self.S).T + 1e-6
        return self.base + num / den

    def rank_scores(self) -> np.ndarray:
        return self.predict_matrix()


class SVDRecommender:
    """Truncated SVD of the zero-filled residual matrix (sklearn); baseline + low-rank part."""

    name = "svd"

    def __init__(self, k: int = 10, seed: int = 0) -> None:
        self.k, self.seed = k, seed

    def fit(self, R: np.ndarray, M: np.ndarray) -> "SVDRecommender":
        self.mu, self.bu, self.bi = fit_baseline(R, M)
        base = self.mu + self.bu[:, None] + self.bi[None, :]
        self.base = base
        resid = (R - base) * M
        svd = TruncatedSVD(n_components=self.k, random_state=self.seed)
        self.U = svd.fit_transform(resid)  # = U S
        self.Vt = svd.components_
        return self

    def predict_matrix(self) -> np.ndarray:
        return self.base + self.U @ self.Vt

    def rank_scores(self) -> np.ndarray:
        return self.predict_matrix()


class ALSRecommender:
    """Biased matrix factorisation fitted by alternating ridge solves on observed entries only.

    r_ui ~ mu + b_u + b_i + p_u . q_i ; each user/item update is a closed-form
    weighted-lambda ridge regression (ALS-WR style: lambda scaled by the number of ratings).
    """

    name = "als"

    def __init__(self, factors: int = 8, reg: float = 0.1, iters: int = 15, seed: int = 0) -> None:
        self.f, self.reg, self.iters, self.seed = factors, reg, iters, seed
        self.history: list[float] = []

    @staticmethod
    def _solve(F: np.ndarray, T: np.ndarray, Mm: np.ndarray, reg: float) -> np.ndarray:
        """Solve for every row's factors given fixed other-side factors F (m x f)."""
        out = np.zeros((T.shape[0], F.shape[1]))
        eye = np.eye(F.shape[1])
        for u in range(T.shape[0]):
            idx = np.nonzero(Mm[u])[0]
            if idx.size == 0:
                continue
            Fo = F[idx]
            out[u] = np.linalg.solve(Fo.T @ Fo + reg * idx.size * eye, Fo.T @ T[u, idx])
        return out

    def fit(self, R: np.ndarray, M: np.ndarray) -> "ALSRecommender":
        rng = np.random.default_rng(self.seed)
        self.mu, self.bu, self.bi = fit_baseline(R, M)
        resid = (R - self.mu - self.bu[:, None] - self.bi[None, :]) * M
        self.P = rng.normal(0, 0.1, (R.shape[0], self.f))
        self.Q = rng.normal(0, 0.1, (R.shape[1], self.f))
        self.history = []
        for _ in range(self.iters):
            self.P = self._solve(self.Q, resid, M, self.reg)
            self.Q = self._solve(self.P, resid.T, M.T, self.reg)
            err = (resid - self.P @ self.Q.T)[M]
            self.history.append(float(np.sqrt((err ** 2).mean())))
        return self

    def predict_matrix(self) -> np.ndarray:
        return self.mu + self.bu[:, None] + self.bi[None, :] + self.P @ self.Q.T

    def rank_scores(self) -> np.ndarray:
        return self.predict_matrix()

    def fold_in(self, item_idx: np.ndarray, ratings: np.ndarray, lam_u: float = 15.0) -> tuple[float, np.ndarray]:
        """Estimate (bias, factors) for an unseen user from a few ratings; zeros when none given."""
        if len(item_idx) == 0:
            return 0.0, np.zeros(self.f)
        r = ratings - self.mu - self.bi[item_idx]
        b = float(r.sum() / (lam_u + len(r)))
        Fo = self.Q[item_idx]
        p = np.linalg.solve(Fo.T @ Fo + self.reg * len(r) * np.eye(self.f), Fo.T @ (r - b))
        return b, p


class ContentColdStart:
    """Predict bias and latent factors of brand-new items from content features (ridge)."""

    def __init__(self) -> None:
        self.ridge = RidgeCV(alphas=np.logspace(-2, 2, 9))

    def fit(self, als: ALSRecommender, X: np.ndarray, counts: np.ndarray, min_count: int = 5) -> "ContentColdStart":
        warm = counts >= min_count
        Y = np.column_stack([als.bi, als.Q])
        self.ridge.fit(X[warm], Y[warm])
        self.als = als
        return self

    def score_new_items(self, X_new: np.ndarray) -> np.ndarray:
        """Predicted ratings (all users x new items)."""
        Y = self.ridge.predict(X_new)
        a = self.als
        return a.mu + a.bu[:, None] + Y[:, 0][None, :] + a.P @ Y[:, 1:].T
