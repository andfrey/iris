"""
FUCCI label generator: maps 2D intensity points (label_488, label_561)
to a smooth reference cell-cycle curve and returns the phase on [0, 2π).

Implements:
- Compute centroid-centered angles for points
- Bin by angle, average within bins to form a closed curve
- Smooth curve with periodic Gaussian filter
- Interpolate curve (x(t), y(t)) over t ∈ [0, 2π)
- Project points to nearest curve location to get phases

Extras provided here:
- fit_from_dataset(dataset, ...) to build curve directly from a Dataset producing labels
- Caching: avoid recomputing the phase curve using a deterministic cache key
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict
from pathlib import Path
import json
import hashlib

from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d

try:
    # Prefer project utilities if available for stable hashing
    from .utils import make_hash_from_dict  # type: ignore
except Exception:  # pragma: no cover
    make_hash_from_dict = None  # Fallback to hashlib if not available


@dataclass
class FucciLabelGeneratorConfig:
    M: int = 100  # number of phase bins for the reference curve
    sigma: float = 2.0  # smoothing sigma for gaussian_filter1d
    n_samples: int = 1000  # samples along curve for nearest-search
    feature_cols: Tuple[str, str] = ("label_488", "label_561")


class FucciCurveProjector:
    """Project FUCCI intensities to phases by projecting 2D intensity points onto a smooth reference curve.

    API:
        fit(points): build the reference curve and interpolants
        transform(points): -> (phases (N,), projected_points (N,2))
        fit_transform(points): convenience fit + transform
        plot_reference_curve / plot_mapping: optional visualization
    """

    def __init__(
        self,
        dataset: Any,
        M: int = 100,
        sigma: float = 2.0,
        n_samples: int = 1000,
        feature_cols: Tuple[str, str] = ("label_488", "label_561"),
        default_cache_dir: Optional[str] = None,
    ):
        self.config = FucciLabelGeneratorConfig(
            M=M, sigma=sigma, n_samples=n_samples, feature_cols=feature_cols
        )
        self.dataset = dataset
        # Learned artifacts after fit
        self.centroid: Optional[np.ndarray] = None
        self.t_uniform: Optional[np.ndarray] = None
        self.curve_points: Optional[np.ndarray] = None  # (M,2)
        self._interp_x = None
        self._interp_y = None
        self.default_cache_dir = Path(default_cache_dir) if default_cache_dir else None

    # -----------------------------
    # Public API
    # -----------------------------
    def fit(self, points) -> "FucciCurveProjector":
        P = _as_points(points, self.config.feature_cols)
        (
            self.t_uniform,
            self.curve_points,
            self._interp_x,
            self._interp_y,
            self.centroid,
        ) = self._compute_phase_curve(P, M=self.config.M, sigma=self.config.sigma)
        return self

    def project(self, intensities) -> Tuple[float, np.ndarray]:
        """Project a single 2D intensity point onto the curve and return (phase, projected_point)."""
        self._ensure_fitted()
        projected, phase = self._map_point_to_curve(
            intensities, self._interp_x, self._interp_y, n_samples=self.config.n_samples
        )
        return projected, phase

    def point_at_phase(self, phase) -> np.ndarray:
        """Map a phase value (or array of phases) in radians back to 2D coordinates on the curve.

        Args:
            phase: float or array-like of phases in radians. Values can be outside [0, 2π);
                   they'll be wrapped into that range.

        Returns:
            If input is scalar: np.ndarray with shape (2,) for the (x, y) point.
            If input is array-like of shape (N,): np.ndarray with shape (N, 2).
        """
        self._ensure_fitted()

        p = np.asarray(phase, dtype=float)
        p = (p + 2 * np.pi) % (2 * np.pi)

        if p.ndim == 0:
            x = float(self._interp_x(p))
            y = float(self._interp_y(p))
            return np.array([x, y], dtype=float)
        else:
            x = self._interp_x(p)
            y = self._interp_y(p)
            return np.stack([x, y], axis=-1).astype(float)

    def fit_from_dataset(
        self,
        use_cache: bool = True,
        cache_key_extra: Optional[Dict[str, Any]] = None,
    ) -> "FucciCurveProjector":
        """Fit the reference curve directly from a dataset.

        The dataset can be one of:
          - An object with get_dataset_df() returning a DataFrame containing feature_cols
          - An iterable indexable dataset returning (..., labels) where labels shape is (2,) for (488, 561)

        Args:
            max_samples: Optional subsample size for speed on large datasets
            use_cache: If True, will attempt to load/save cached curve
            cache_key_extra: Extra fields to include in the cache key for disambiguation
        """
        # Resolve cache path and key
        cdir = self.default_cache_dir or Path(".cache")
        cdir.mkdir(parents=True, exist_ok=True)

        ds_len = len(self.dataset)
        cache_params: Dict[str, Any] = {
            "M": self.config.M,
            "sigma": self.config.sigma,
            "feature_cols": list(self.config.feature_cols),
            "dataset_len": ds_len,
            "datasource": self.dataset.data_source.__class__.__name__,
        }
        if cache_key_extra:
            cache_params.update(cache_key_extra)
        cache_key = _hash_dict(cache_params)
        cache_path = cdir / f"fucci_curve_{cache_key}.npz"

        if use_cache and cache_path.exists():
            data = np.load(cache_path, allow_pickle=False)
            self.t_uniform = data["t_uniform"]
            self.curve_points = data["curve_points"]
            self.centroid = data["centroid"]
            # rebuild interpolants
            self._interp_x = interp1d(
                self.t_uniform,
                self.curve_points[:, 0],
                kind="cubic",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            self._interp_y = interp1d(
                self.t_uniform,
                self.curve_points[:, 1],
                kind="cubic",
                fill_value="extrapolate",
                assume_sorted=True,
            )
            print(f"✓ Loaded FUCCI curve from cache: {cache_path}")
            return self

        # Extract points from dataset
        print("⟳ Computing FUCCI reference curve from dataset...")
        P = self._extract_points_from_dataset(self.dataset)
        self.fit(P)

        if use_cache:
            np.savez_compressed(
                cache_path,
                t_uniform=self.t_uniform,
                curve_points=self.curve_points,
                centroid=self.centroid,
            )
            print(f"✓ Saved FUCCI curve cache: {cache_path}")
        return self

    # -----------------------------
    # Core implementation
    # -----------------------------
    @staticmethod
    def _compute_phase_curve(points: np.ndarray, M: int = 200, sigma: float = 2.0):
        # Remove points with any negative values
        points = points[(points[:, 0] >= 0) & (points[:, 1] >= 0)]
        # Centroid and polar angles for binning
        centroid = points.mean(axis=0)
        angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])
        angles = (angles + 2 * np.pi) % (2 * np.pi)

        # Phase bins and mean per bin
        phase_bins = np.linspace(0.0, 2 * np.pi, M + 1)
        curve_points = np.zeros((M, 2), dtype=np.float64)
        for i in range(M):
            mask = (angles >= phase_bins[i]) & (angles < phase_bins[i + 1])
            if mask.sum() > 0:
                curve_points[i] = points[mask].mean(axis=0)
            else:
                # If no points in bin, repeat previous point (or mean if first)
                curve_points[i] = curve_points[i - 1] if i > 0 else points.mean(axis=0)

        # Smooth with periodic gaussian filter (wrap mode)
        curve_points[:, 0] = gaussian_filter1d(curve_points[:, 0], sigma=sigma, mode="wrap")
        curve_points[:, 1] = gaussian_filter1d(curve_points[:, 1], sigma=sigma, mode="wrap")

        # Build cubic interpolants over uniform t in [0, 2π)
        t_uniform = np.linspace(0.0, 2 * np.pi, M, endpoint=False)
        interp_x = interp1d(
            t_uniform,
            curve_points[:, 0],
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        interp_y = interp1d(
            t_uniform,
            curve_points[:, 1],
            kind="cubic",
            fill_value="extrapolate",
            assume_sorted=True,
        )
        return t_uniform, curve_points, interp_x, interp_y, centroid

    @staticmethod
    def _map_point_to_curve(
        point: np.ndarray, interp_x, interp_y, n_samples: int = 2000
    ) -> Tuple[np.ndarray, float]:
        """Map a single 2D point to the closest point on the interpolated curve.

        Args:
            point: shape (2,)
        Returns:
            projected_point: shape (2,)
            phase: float in [0, 2π)
        """
        pt = np.asarray(point, dtype=float).reshape(-1)
        if pt.shape[0] != 2:
            raise ValueError(f"Expected a single 2D point, got shape {pt.shape}")

        ts = np.linspace(0.0, 2 * np.pi, n_samples, endpoint=True)
        curve_eval = np.stack([interp_x(ts), interp_y(ts)], axis=1)  # (n_samples, 2)

        # Compute squared distances to each sampled curve point
        diffs = curve_eval - pt[None, :]  # (n_samples, 2)
        dists2 = np.sum(diffs * diffs, axis=1)  # (n_samples,)
        idx = int(np.argmin(dists2))  # scalar index

        projected = curve_eval[idx]
        phase = ts[idx]
        phase = float((phase + 2 * np.pi) % (2 * np.pi))
        return projected, phase

    # -----------------------------
    # Plotting helpers
    # -----------------------------
    def plot_curve(
        self,
        ax=None,
        show: bool = True,
        color: str = "crimson",
        lw: float = 2.0,
        plot_centroid: bool = False,
        markers: bool = False,
        n_markers: int = 0,
        intensity_points: Optional[np.ndarray] = None,
        plot_random_projections: bool = False,
    ):
        """Plot the fitted reference curve (and optionally the centroid and markers).

        Args:
            ax: Matplotlib Axes; if None, uses current axes
            show: Call plt.show() when done
            color: Line color for the curve
            lw: Line width for the curve
            plot_centroid: Whether to plot the centroid used for angle computation
            markers: Whether to draw small markers along the curve
            n_markers: If >0, place this many evenly spaced markers along the curve

        Returns:
            The Matplotlib Axes with the plot
        """
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        ax = ax or plt.gca()
        # Add random points and plot their projections to the curve
        rng = np.random.default_rng(42)
        n_rand = 10
        rand_pts = rng.uniform(0, 10, size=(n_rand, 2))
        # Plot the reference curve
        ax.plot(
            self.curve_points[:, 0],
            self.curve_points[:, 1],
            "-",
            color=color,
            lw=lw,
            label="Cell cycle curve",
        )

        # Optional centroid
        if plot_centroid and self.centroid is not None:
            ax.scatter(
                [self.centroid[0]],
                [self.centroid[1]],
                marker="x",
                s=60,
                c="black",
                label="Centroid",
            )
            ax.set_xlabel("FUCCI 488nm intensity")
            ax.set_ylabel("FUCCI 561nm intensity")

        # Optional markers
        if markers or n_markers > 0:
            if n_markers <= 0:
                n_markers = 8
            idxs = np.linspace(0, len(self.curve_points) - 1, n_markers, dtype=int)
            ax.scatter(
                self.curve_points[idxs, 0],
                self.curve_points[idxs, 1],
                s=20,
                c=color,
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
                label=None,
            )
        if intensity_points is not None:
            pts = _as_points(intensity_points, self.config.feature_cols)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                c="gray",
                alpha=0.5,
                label="Intensity points",
            )
        ax.set_aspect("equal", adjustable="box")
        ax.legend(loc="best")
        if plot_random_projections:
            for pt in rand_pts:
                projected, phase = self._map_point_to_curve(
                    pt, self._interp_x, self._interp_y, n_samples=self.config.n_samples
                )
                ax.plot([pt[0], projected[0]], [pt[1], projected[1]], "b--", lw=1, alpha=0.7)
                ax.scatter([pt[0]], [pt[1]], c="blue", s=30, marker="o", label=None)
                ax.scatter([projected[0]], [projected[1]], c="red", s=30, marker="x", label=None)
        if show:
            plt.show()
        return ax

    # -----------------------------
    # Internals
    # -----------------------------
    def _ensure_fitted(self):
        if self.curve_points is None or self._interp_x is None or self._interp_y is None:
            self.fit_from_dataset()

    def _extract_points_from_dataset(self, dataset: Any) -> np.ndarray:
        """Try several strategies to obtain (N,2) points from a dataset.

        Iterate and collect labels assuming each sample yields (..., labels)
        """

        # Iterable dataset returning labels at index 1 or last position
        pts_list = []
        for sample in tqdm(dataset, desc="Extracting FUCCI intensities from dataset"):
            # allow ((x, feat), y) or (x, y)
            if isinstance(sample, tuple):
                labels = sample[-1]
            else:
                raise ValueError(
                    "Each dataset item must be a tuple ending with labels of shape (2,)"
                )
            arr = np.asarray(labels, dtype=float).reshape(-1)
            if arr.shape[0] != 2:
                raise ValueError(f"Expected label shape (2,) for (488,561), got {arr.shape}")
            pts_list.append(arr)
        return np.stack(pts_list, axis=0)


def _as_points(points_like, feature_cols: Tuple[str, str]) -> np.ndarray:
    """Convert DataFrame or array-like to numpy array of shape (N,2)."""
    if hasattr(points_like, "to_numpy") and hasattr(points_like, "__getitem__"):
        # Likely a pandas DataFrame
        return points_like[list(feature_cols)].to_numpy(dtype=float)
    arr = np.asarray(points_like, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"Expected shape (N,2) points, got {arr.shape}")
    return arr


def _hash_dict(d: Dict[str, Any]) -> str:
    """Stable short hash for cache keys, using project helper if available."""
    if make_hash_from_dict is not None:
        return make_hash_from_dict(d, length=12)
    # fallback
    dumped = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(dumped).hexdigest()[:12]
