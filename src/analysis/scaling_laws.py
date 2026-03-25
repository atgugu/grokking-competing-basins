"""Linear regression for scaling law analysis."""

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class LinearFit:
    slope: float
    intercept: float
    r_squared: float
    x: np.ndarray
    y: np.ndarray

    def predict(self, x: np.ndarray) -> np.ndarray:
        return self.slope * np.asarray(x) + self.intercept

    def __str__(self):
        return f"y = {self.slope:.2f}x + {self.intercept:.2f}, R² = {self.r_squared:.3f}"


def fit_linear(x: list | np.ndarray, y: list | np.ndarray) -> LinearFit:
    """Fit a linear regression and return results."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    result = stats.linregress(x, y)
    return LinearFit(
        slope=result.slope,
        intercept=result.intercept,
        r_squared=result.rvalue ** 2,
        x=x,
        y=y,
    )
