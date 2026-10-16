"""Figure generation (Agg backend)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_data_overview(R: np.ndarray, M: np.ndarray, path: Path) -> None:
    """Rating histogram and item-popularity long tail."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    vals, cnt = np.unique(R[M], return_counts=True)
    ax[0].bar(vals, cnt, color="#3b6ea8")
    ax[0].set(title="Rating distribution (train)", xlabel="rating", ylabel="count")
    ax[1].plot(np.sort(M.sum(0))[::-1], color="#c0504d")
    ax[1].set(title="Item popularity (long tail)", xlabel="item rank", ylabel="# train ratings")
    _save(fig, path)


def plot_model_comparison(results: dict[str, dict], path: Path, k: int = 10) -> None:
    """RMSE and NDCG@k / recall@k per model."""
    names = list(results)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    for a, key, title in zip(ax, ["rmse", f"ndcg@{k}", f"recall@{k}"], ["RMSE (lower better)", f"NDCG@{k}", f"Recall@{k}"]):
        v = [results[n].get(key, np.nan) for n in names]
        a.bar(range(len(names)), v, color="#3b6ea8")
        a.set_xticks(range(len(names)))
        a.set_xticklabels(names, rotation=30, ha="right")
        a.set_title(title)
        for i, x in enumerate(v):
            if not np.isnan(x):
                a.text(i, x, f"{x:.3f}", ha="center", va="bottom", fontsize=8)
    _save(fig, path)


def plot_cv_grid(grid: np.ndarray, factors: tuple, regs: tuple, history: list[float], path: Path) -> None:
    """ALS CV RMSE heatmap and training-error convergence of the chosen model."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    im = ax[0].imshow(grid, cmap="viridis_r", aspect="auto")
    ax[0].set_xticks(range(len(regs)))
    ax[0].set_xticklabels(regs)
    ax[0].set_yticks(range(len(factors)))
    ax[0].set_yticklabels(factors)
    ax[0].set(title="ALS CV RMSE", xlabel="regularisation", ylabel="factors")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax[0].text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="w", fontsize=8)
    fig.colorbar(im, ax=ax[0])
    ax[1].plot(range(1, len(history) + 1), history, marker="o")
    ax[1].set(title="ALS train RMSE (residual) per iteration", xlabel="iteration", ylabel="RMSE")
    _save(fig, path)


def plot_cold_start(user_curve: dict[str, list[float]], reveals: list[int], item_res: dict[str, float], path: Path) -> None:
    """Cold-user NDCG vs number of revealed ratings, and cold-item NDCG by method."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, ys in user_curve.items():
        ax[0].plot(reveals, ys, marker="o", label=name)
    ax[0].set(title="Cold users: NDCG@10 vs revealed ratings", xlabel="ratings revealed", ylabel="NDCG@10")
    ax[0].legend()
    ax[1].bar(list(item_res), list(item_res.values()), color=["#999", "#c0504d", "#3b6ea8"][: len(item_res)])
    ax[1].set(title="Cold items: NDCG@10 over new-item pool", ylabel="NDCG@10")
    ax[1].tick_params(axis="x", rotation=15)
    _save(fig, path)
