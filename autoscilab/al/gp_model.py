"""
GP surrogate model wrapping scikit-learn GaussianProcessRegressor.

Works in log-space for the y values (measurements span many orders of magnitude
across NewtonBench domains) and normalizes X to [0,1] per parameter.
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel


class GPSurrogate:
    def __init__(self, use_log_y: bool = True):
        """
        Args:
            use_log_y: If True, fit GP on log(y) — recommended for physical measurements
                       that span orders of magnitude (forces, energies, etc.)
        """
        self.use_log_y = use_log_y
        self._bounds: dict[str, tuple[float, float]] | None = None
        self._gp: GaussianProcessRegressor | None = None
        self._is_fitted = False

    def _make_gp(self, n_dims: int = 1) -> GaussianProcessRegressor:
        """
        Build a GP.  If n_dims > 1 the kernel uses per-dimension (ARD)
        length-scales so that irrelevant variables get large length-scales.
        """
        ls_init = np.ones(n_dims) if n_dims > 1 else 1.0
        ls_bounds = (1e-2, 1e2)
        kernel = (
            ConstantKernel(1.0, constant_value_bounds=(1e-3, 1e3))
            * Matern(length_scale=ls_init, length_scale_bounds=ls_bounds, nu=2.5)
            + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e0))
        )
        return GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=5,
            normalize_y=True,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        bounds: dict[str, tuple[float, float]],
    ) -> None:
        """
        Fit GP on all accumulated data.

        Args:
            X: (N, D) parameter array in natural units
            y: (N,) measurement array
            bounds: {param_name: (min, max)} — used for normalization
        """
        if len(X) == 0:
            self._is_fitted = False
            return

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        finite_mask = np.isfinite(y)
        if X.ndim == 2:
            finite_mask &= np.all(np.isfinite(X), axis=1)

        if not np.all(finite_mask):
            X = X[finite_mask]
            y = y[finite_mask]

        if len(X) == 0:
            self._is_fitted = False
            return

        self._bounds = bounds
        X_norm = self._normalize(X, bounds)

        if self.use_log_y:
            y_fit = np.log(np.abs(y) + 1e-300)
        else:
            y_fit = y.copy()

        y_fit = np.asarray(y_fit, dtype=float)
        finite_y_fit = np.isfinite(y_fit)
        if not np.all(finite_y_fit):
            X_norm = X_norm[finite_y_fit]
            y_fit = y_fit[finite_y_fit]

        if len(X_norm) == 0:
            self._is_fitted = False
            return

        n_dims = X_norm.shape[1]
        self._gp = self._make_gp(n_dims=n_dims)
        self._gp.fit(X_norm, y_fit)
        self._is_fitted = True

    def predict(
        self,
        X: np.ndarray,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and std at X (in natural units).

        Returns:
            mean: (N,) — in natural units (exponentiated if use_log_y)
            std:  (N,) — in log-space std (useful for acquisition functions)
        """
        if not self._is_fitted:
            raise RuntimeError("GP not fitted yet.")
        b = bounds or self._bounds
        X_norm = self._normalize(X, b)
        mean_log, std_log = self._gp.predict(X_norm, return_std=True)

        if self.use_log_y:
            mean = np.exp(mean_log)
        else:
            mean = mean_log

        return mean, std_log  # return log-space std for EI/UCB

    def predict_log(
        self,
        X: np.ndarray,
        bounds: dict[str, tuple[float, float]] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return raw (log-space) mean and std — used by acquisition functions."""
        if not self._is_fitted:
            raise RuntimeError("GP not fitted yet.")
        b = bounds or self._bounds
        X_norm = self._normalize(X, b)
        return self._gp.predict(X_norm, return_std=True)

    @staticmethod
    def _normalize(X: np.ndarray, bounds: dict[str, tuple[float, float]]) -> np.ndarray:
        """
        Normalize X to [0,1] with mixed linear/log scaling.

        Log scaling is ideal for wide positive ranges (e.g. 1e-6..1e2), but it
        collapses low-dynamic ranges (e.g. 0..1 action controls) near zero.
        Use linear normalization when range is narrow; otherwise use log.
        """
        keys = list(bounds.keys())
        mins = np.array([bounds[k][0] for k in keys], dtype=float)
        maxs = np.array([bounds[k][1] for k in keys], dtype=float)

        X_norm = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            lo = mins[j]
            hi = maxs[j]
            xj = X[:, j]
            ratio = hi / max(lo, 1e-12)
            use_linear = lo <= 0 or ratio < 20.0
            if use_linear:
                X_norm[:, j] = (xj - lo) / (hi - lo + 1e-12)
            else:
                lx = np.log(np.clip(xj, 1e-300, None))
                llo = np.log(max(lo, 1e-300))
                lhi = np.log(max(hi, lo * 1.01))
                X_norm[:, j] = (lx - llo) / (lhi - llo + 1e-12)
        return np.clip(X_norm, 0.0, 1.0)

    def ard_lengthscales(self) -> np.ndarray | None:
        """
        Return per-dimension ARD length-scales from the fitted kernel.

        The GP is fitted on X normalized to [0,1], so all length-scales live in
        the same space.  A large length-scale (>> 1) means the kernel is nearly
        flat in that direction → the variable explains little variance → likely
        irrelevant.  Returns None if the GP is not fitted.
        """
        if not self._is_fitted or self._gp is None:
            return None
        # The fitted kernel is  ConstantKernel * Matern + WhiteKernel
        # Walk the kernel tree to find the Matern component.
        k = self._gp.kernel_
        for part in [k, getattr(k, "k1", None), getattr(k, "k2", None)]:
            if part is None:
                continue
            for sub in [part, getattr(part, "k1", None), getattr(part, "k2", None)]:
                if isinstance(sub, Matern):
                    ls = np.atleast_1d(sub.length_scale)
                    return ls
        return None

    def relevant_variables(
        self,
        param_names: list[str],
        threshold: float = 2.0,
    ) -> list[str]:
        """
        Return variable names whose ARD length-scale is below *threshold*.

        A length-scale above *threshold* in [0,1]-normalised space means the
        kernel changes negligibly over the full variable range — the variable
        is empirically irrelevant.  A threshold of 2.0 is conservative
        (variables that account for any meaningful variance are kept).

        Falls back to all variables if ARD is unavailable or the GP is
        scalar-length-scale (length_scale is a 1-element array but param_names
        has more than one element).
        """
        ls = self.ard_lengthscales()
        if ls is None or len(ls) != len(param_names):
            return list(param_names)
        relevant = [p for p, l in zip(param_names, ls) if l < threshold]
        # Always keep at least 2 variables
        if len(relevant) < 2:
            sorted_idx = np.argsort(ls)
            relevant = [param_names[i] for i in sorted_idx[:2]]
        return relevant

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
