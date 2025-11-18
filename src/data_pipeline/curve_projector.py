"""
FUCCI label projector: projects 2D intensity points (label_488, label_561)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, Optional, Any, Dict
from pathlib import Path
import json
import hashlib

from networkx import radius
from tqdm import tqdm
import numpy as np

try:
    # Prefer project utilities if available for stable hashing
    from .utils import make_hash_from_dict  # type: ignore
except Exception:  # pragma: no cover
    make_hash_from_dict = None  # Fallback to hashlib if not available


class FucciCurveProjector:
    """Project FUCCI intensities to phases by projecting 2D intensity points onto a smooth reference curve."""

    def __init__(
        self,
        dataset: Any,
        feature_cols: Tuple[str, str] = ("label_488", "label_561"),
        default_cache_dir: Optional[str] = None,
    ):
        self.feature_cols = feature_cols
        self.dataset = dataset
        # Learned artifacts after fit
        self.centroid: Optional[np.ndarray] = None
        self.default_cache_dir = Path(default_cache_dir) if default_cache_dir else None

    def fit(self, points) -> "FucciCurveProjector":
        print("⟳ Fitting FUCCI curve projector…")
        P = _as_points(points, self.feature_cols)
        self.centroid, self.radius = self._fit_cycle_center(P)
        # self.theta_1, self.theta_2, self.theta_3, self.c1, self.c2, self.c3 = self._fit_triangle(P)
        return self

    def project(self, intensities) -> Tuple[float, np.ndarray]:
        """Project a single 2D intensity point onto the curve and return (phase, projected_point)."""
        self._ensure_fitted()
        # projected, phase = self._map_point_to_curve(
        #     intensities, self._interp_x, self._interp_y, n_samples=self.config.n_samples
        # )
        phase = self.compute_phase(intensities)
        return phase, None

    def compute_phase(self, point) -> float:
        """Compute phases and projected points for an array of 2D intensity points."""

        return np.arctan2(point[1] - self.centroid[1], point[0] - self.centroid[0])

    def _phase_to_circle(self, phase, radius=1) -> np.ndarray:
        """Convert phase (angle) to circle coordinates."""
        return [
            self.centroid[0] + radius * np.cos(phase),
            self.centroid[1] + radius * np.sin(phase),
        ]

    def fit_from_dataset(
        self,
        use_cache: bool = True,
        cache_key_extra: Optional[Dict[str, Any]] = None,
    ) -> "FucciCurveProjector":
        """Fit the reference curve directly from a dataset with two-level caching.

        1) Cache extracted (N,2) intensity points from the dataset
        2) Cache derived curve artifacts: t_uniform, curve_points, centroid

        This avoids re-iterating the dataset if only curve parameters change, and
        also skips recompute if the same points and parameters were already processed.
        """
        # Resolve cache directory
        cdir = self.default_cache_dir or Path(".cache")
        cdir.mkdir(parents=True, exist_ok=True)

        # Build a cache key for the raw extracted points
        ds_len = len(self.dataset)
        pts_key_params: Dict[str, Any] = {
            "feature_cols": list(self.feature_cols),
            "dataset_len": ds_len,
            "datasource": getattr(self.dataset, "data_source", self.dataset).__class__.__name__,
        }
        if cache_key_extra:
            pts_key_params.update({f"extra_{k}": v for k, v in cache_key_extra.items()})
        pts_key = _hash_dict(pts_key_params)
        pts_cache_path = cdir / f"fucci_points_{pts_key}.npz"

        # Load or compute points
        if use_cache and pts_cache_path.exists():
            data = np.load(pts_cache_path, allow_pickle=False)
            P = data["points"]
        else:
            print("⟳ Extracting FUCCI intensity points from dataset…")
            P = self._extract_points_from_dataset(self.dataset)
            if use_cache:
                np.savez_compressed(pts_cache_path, points=P)

        # Compute curve from points
        print("⟳ Computing FUCCI reference curve from points…")
        self.fit(P)

        return self

    # -----------------------------
    # Core implementation
    # -----------------------------
    @staticmethod
    def _fit_triangle(points: np.ndarray) -> Tuple[float, float, float, float, float, float]:
        import numpy as np
        from scipy import optimize

        # Function to calculate distance from each point to the three lines
        def calc_distances(theta_1, theta_2, theta_3, c1, c2, c3):
            # Perpendicular vectors (normals to the lines)
            n1 = np.array([-np.sin(theta_1), np.cos(theta_1)])
            n2 = np.array([-np.sin(theta_2), np.cos(theta_2)])
            n3 = np.array([-np.sin(theta_3), np.cos(theta_3)])

            # Distance from each point to each line: |n · p + c|
            d1 = np.abs(points @ n1 + c1)
            d2 = np.abs(points @ n2 + c2)
            d3 = np.abs(points @ n3 + c3)

            d = np.stack([d1, d2, d3], axis=1)  # (N, 3)
            # For each point, take the minimum distance to any of the three lines
            min_distances = np.min(d, axis=1)
            return min_distances

        # Residual function: array of distances for leastsq
        def residuals(params):
            return calc_distances(*params)

        # Initial guess for three vectors forming a triangular shape
        theta_1_init = 1 * np.pi / 2  # towards low 488, high 561 (G1 region)
        theta_2_init = 3 * np.pi / 10  # 144° - towards low 488, high 561 (G1 region)
        theta_3_init = 1 * np.pi / 6  # 30° - bottom region

        # Initial constants (offset from origin)
        c1_init = 7.3
        c2_init = 1.0
        c3_init = -1.4

        initial_guess = np.array(
            [theta_1_init, theta_2_init, theta_3_init, c1_init, c2_init, c3_init]
        )

        # Least squares optimization to find optimal parameters
        result, ier = optimize.leastsq(residuals, initial_guess, maxfev=20000)

        theta_1, theta_2, theta_3, c1, c2, c3 = result

        return theta_1, theta_2, theta_3, c1, c2, c3

    @staticmethod
    def _fit_cycle_center(points: np.ndarray) -> np.ndarray:
        import numpy as np
        from scipy import optimize

        # Function to calculate distance from center (xc, yc) to each point
        def calc_R(xc, yc, r):
            return np.sqrt((points[:, 0] - xc) ** 2 + (points[:, 1] - yc) ** 2) - r

        # Residual function: difference between each point's distance and mean distance (radius)
        def residuals(c):
            Ri = calc_R(*c)
            return Ri

        # Initial guess: mean of points as center
        center_estimate = np.mean(points, axis=0)
        radius_estimate = np.mean(
            np.sqrt(
                (points[:, 0] - center_estimate[0]) ** 2 + (points[:, 1] - center_estimate[1]) ** 2
            )
        )
        initial_guess = (center_estimate[0], center_estimate[1], radius_estimate)

        # Least squares optimization to find center
        (center_x, center_y, radius), ier = optimize.leastsq(residuals, initial_guess)
        centroid = np.array([center_x, center_y])
        radius = radius
        return centroid, radius

    # -----------------------------
    # Plotting helpers
    # -----------------------------
    def plot_triangle(
        self,
        ax=None,
        show: bool = True,
        color: str = "green",
        lw: float = 2.0,
        intensity_points: Optional[np.ndarray] = None,
        plot_random_projections: bool = False,
    ):
        """Plot the fitted reference triangle.

        Args:
            ax: Matplotlib Axes; if None, uses current axes
            show: Call plt.show() when done
            color: Line color for the triangle edges
            lw: Line width for the triangle
            intensity_points: Optional array-like of shape (N,2) to plot as gray points
            plot_random_projections: Whether to add random points and plot their projections to the triangle

        Returns:
            The Matplotlib Axes with the plot
        """
        import matplotlib.pyplot as plt

        self._ensure_fitted()

        # Extract triangle parameters from attributes
        theta_1 = self.theta_1
        theta_2 = self.theta_2
        theta_3 = self.theta_3

        c1 = self.c1
        c2 = self.c2
        c3 = self.c3
        print(f"Triangle parameters:")
        print(f"  θ1: {np.degrees(theta_1):.1f}°")
        print(f"  θ2: {np.degrees(theta_2):.1f}°")
        print(f"  θ3: {np.degrees(theta_3):.1f}°")
        print(f"  c1: {c1:.1f}")
        print(f"  c2: {c2:.1f}")
        print(f"  c3: {c3:.1f}")
        # Create normal vectors (perpendicular to each edge)
        n1 = np.array([-np.sin(theta_1), np.cos(theta_1)])
        n2 = np.array([-np.sin(theta_2), np.cos(theta_2)])
        n3 = np.array([-np.sin(theta_3), np.cos(theta_3)])

        ax = ax or plt.gca()

        # Estimate a suitable scale based on the data range
        if intensity_points is not None:
            pts = _as_points(intensity_points, self.feature_cols)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
            center = pts.mean(axis=0)
            scale = max(pts.max(axis=0) - pts.min(axis=0)) * 1.0
        else:
            center = np.array([0, 0])
            scale = 1.0

        # Draw the three lines (edges of the triangle)
        # Each line is defined by: n · p + c = 0
        # We can rewrite as: n1*x + n2*y + c = 0  =>  y = -(n1*x + c)/n2
        for i, (n, c_val, theta, color) in enumerate(
            [
                (n1, c1, theta_1, "yellow"),
                (n2, c2, theta_2, "darkred"),
                (n3, c3, theta_3, "darkgreen"),
            ]
        ):
            angle_deg = np.degrees(theta)
            # Create a line perpendicular to normal n
            # Direction along the line (perpendicular to normal)
            line_dir = np.array([n[1], -n[0]])  # Rotate normal by 90°

            # Find a point on the line: n · p + c = 0
            # Choose p such that it's near the center
            if abs(n[1]) > abs(n[0]):
                # Solve for y: n[1]*y = -c - n[0]*center[0]
                p_on_line = np.array([center[0], -(c_val + n[0] * center[0]) / n[1]])
            else:
                # Solve for x: n[0]*x = -c - n[1]*center[1]
                p_on_line = np.array([-(c_val + n[1] * center[1]) / n[0], center[1]])

            # Draw line through p_on_line in direction line_dir
            t = np.linspace(-scale, scale, 100)
            line_x = p_on_line[0] + t * line_dir[0]
            line_y = p_on_line[1] + t * line_dir[1]

            ax.plot(
                line_x,
                line_y,
                "-",
                color=color,
                lw=lw,
                # label=(
                #     f"Edge {i+1} (θ={angle_deg:.1f}°)"
                #     if i == 0
                #     else f"Edge {i+1} (θ={angle_deg:.1f}°)"
                # ),
            )

        # Plot intensity points if provided
        if intensity_points is not None:
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                c="gray",
                alpha=0.5,
                # label="Intensity points",
            )

            if plot_random_projections:
                # For triangle projection, we'd need to implement point-to-triangle projection
                # This is more complex than circle projection
                print("Warning: plot_random_projections not yet implemented for triangle")

        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("FUCCI 488nm intensity")
        ax.set_ylabel("FUCCI 561nm intensity")
        ax.legend(loc="best")

        if show:
            plt.show()

        return ax

    def plot_circle(
        self,
        ax=None,
        show: bool = True,
        color: str = "crimson",
        lw: float = 2.0,
        intensity_points: Optional[np.ndarray] = None,
        plot_random_projections: bool = False,
    ):
        """Plot the fitted reference circle.

        Args:
            ax: Matplotlib Axes; if None, uses current axes
            show: Call plt.show() when done
            color: Line color for the circle
            lw: Line width for the circle
            intensity_points: Optional array-like of shape (N,2) to plot as gray points
            plot_random_projections: Whether to add random points and plot their projections to the circle

        Returns:
            The Matplotlib Axes with the plot
        """
        import matplotlib.pyplot as plt

        self._ensure_fitted()
        radius = self.radius
        ax = ax or plt.gca()
        theta = np.linspace(0, 2 * np.pi, 100)
        x = self.centroid[0] + radius * np.cos(theta)
        y = self.centroid[1] + radius * np.sin(theta)

        ax.plot(
            x,
            y,
            "-",
            color=color,
            lw=lw,
            label="Fitted Circle",
        )
        ax.scatter(
            self.centroid[0], self.centroid[1], c="black", marker="x", s=60, label="Centroid"
        )

        # Add phase annotations around the circle
        phase_labels = [
            (0, "0"),
            (np.pi / 4, "π/4"),
            (np.pi / 2, "π/2"),
            (3 * np.pi / 4, "3π/4"),
            (np.pi, "π"),
            (-np.pi / 2, "-π/2"),
            (-np.pi / 4, "-π/4"),
            (-3 * np.pi / 4, "-3π/4"),
        ]

        for phase, label in phase_labels:
            # Position on the circle
            px = self.centroid[0] + radius * np.cos(phase)
            py = self.centroid[1] + radius * np.sin(phase)

            # Offset for text (slightly outside the circle)
            text_offset = 1.2
            tx = self.centroid[0] + radius * text_offset * np.cos(phase)
            ty = self.centroid[1] + radius * text_offset * np.sin(phase)

            # Plot marker on circle
            ax.scatter(px, py, c="blue", marker="o", s=40, zorder=5)

            # Add text label
            ax.text(
                tx,
                ty,
                label,
                fontsize=10,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black", alpha=1),
            )

        if intensity_points is not None:
            pts = _as_points(intensity_points, self.feature_cols)
            pts = pts[(pts[:, 0] >= 0) & (pts[:, 1] >= 0)]
            ax.scatter(
                pts[:, 0],
                pts[:, 1],
                s=10,
                c="gray",
                alpha=0.5,
                label="Intensity points",
            )
            if plot_random_projections:
                rng = np.random.default_rng(42)
                n_rand = min(len(intensity_points), 50)
                rand_pts = rng.choice(range(0, len(intensity_points)), size=n_rand, replace=False)
                for pt in rand_pts:
                    phase, _ = self.project(intensity_points[pt])
                    ax.plot(
                        [intensity_points[pt][0], self.centroid[0] + radius * np.cos(phase)],
                        [intensity_points[pt][1], self.centroid[1] + radius * np.sin(phase)],
                        "b--",
                        lw=1,
                        alpha=0.7,
                    )
                    ax.scatter(
                        [intensity_points[pt][0]],
                        [intensity_points[pt][1]],
                        c="blue",
                        s=30,
                        marker="o",
                        label=None,
                    )
        #             ax.scatter([projected[0]], [projected[1]], c="red", s=30, marker="x", label=None)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("FUCCI 488nm intensity")
        ax.set_ylabel("FUCCI 561nm intensity")
        ax.legend(loc="best")
        if show:
            plt.show()
        return ax

    def _ensure_fitted(self):
        # if self.curve_points is None or self._interp_x is None or self._interp_y is None:
        if self.centroid is None:
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
