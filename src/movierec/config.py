"""Central configuration (no hardcoded absolute paths)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class Config:
    """Experiment configuration."""

    seed: int = 7
    n_users: int = 800
    n_items: int = 500
    latent_dim: int = 6
    density: float = 0.06
    rating_noise: float = 0.45
    n_cold_items: int = 50
    n_cold_users: int = 60
    test_frac: float = 0.2
    relevance_threshold: float = 4.0
    top_k: int = 10
    cv_folds: int = 3
    als_factors_grid: tuple = (8, 16, 32)
    als_reg_grid: tuple = (0.05, 0.1, 0.3)
    svd_k_grid: tuple = (5, 10, 20)
    knn_k_grid: tuple = (20, 40)
    cold_user_reveals: tuple = (0, 1, 3, 5, 10)
    data_dir: Path = field(default_factory=lambda: ROOT / "data")
    reports_dir: Path = field(default_factory=lambda: ROOT / "reports")
    models_dir: Path = field(default_factory=lambda: ROOT / "models")
