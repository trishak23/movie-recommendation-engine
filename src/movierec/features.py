"""Item content features used for cold-start item modelling."""
from __future__ import annotations

import numpy as np
import pandas as pd


def item_features(items: pd.DataFrame) -> np.ndarray:
    """Genre multi-hot plus standardised release year, ordered by item_id."""
    items = items.sort_values("item_id")
    g = items[[c for c in items.columns if c.startswith("g_")]].to_numpy(float)
    y = items["year"].to_numpy(float)
    y = (y - y.mean()) / y.std()
    return np.column_stack([g, y])
