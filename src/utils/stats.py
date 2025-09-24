from __future__ import annotations

from typing import Tuple

import numpy as np


def bootstrap_mean_ci(
    values: np.ndarray, num_bootstrap: int = 1000, ci: float = 0.95, seed: int = 0
) -> Tuple[float, float, float]:
    if values.ndim != 1:
        values = values.reshape(-1)
    n = values.shape[0]
    rng = np.random.default_rng(seed)
    means = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        idx = rng.integers(0, n, size=n)
        means[i] = float(values[idx].mean())
    lower_q = (1.0 - ci) / 2.0
    upper_q = 1.0 - lower_q
    mean = float(values.mean())
    low = float(np.quantile(means, lower_q))
    high = float(np.quantile(means, upper_q))
    return mean, low, high
