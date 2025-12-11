#!/usr/bin/env python3
"""
Post-experiment WPPM visualization.

Creates the 4-panel covariance field visualization from Hong et al. (2025):
- σ²_dim1: Variance in dimension 1
- σ_(dim1,dim2): Covariance (off-diagonal)
- σ²_dim2: Variance in dimension 2
- Discrimination ellipses at grid of reference locations

Also supports visualization from AEPsych GP (non-WPPM) after experiment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import jax.numpy as jnp
import jax.random as jr
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
from adaptive_engine.wppm_fitter import load_wppm, compute_covariance, fit_wppm, WPPMConfig, TrialData, save_wppm, WPPMParams
import numpy as np
from typing import Optional, Tuple, Dict, Any
import argparse
from pathlib import Path


from scipy.linalg import sqrtm


def bures_wasserstein(Σ1: np.ndarray, Σ2: np.ndarray) -> float:
    """
    Compute Bures-Wasserstein distance between two covariance matrices.
    
    From Hong et al. (2025) Eq. S13:
    d(Σ1, Σ2) = [tr(Σ1) + tr(Σ2) - 2·tr((Σ1^{1/2} · Σ2 · Σ1^{1/2})^{1/2})]^{1/2}
    
    Measures the "effort" required to morph one ellipse into another.
    """
    sqrt_Σ1 = sqrtm(Σ1)
    inner = sqrtm(sqrt_Σ1 @ Σ2 @ sqrt_Σ1)
    # Ensure real (numerical issues can introduce small imaginary parts)
    inner = np.real(inner)
    d_squared = np.trace(Σ1) + np.trace(Σ2) - 2 * np.trace(inner)
    return np.sqrt(max(d_squared, 0))


def compare_covariance_fields(
    Σ_field1: Dict[str, np.ndarray],
    Σ_field2: Dict[str, np.ndarray],
    normalize: bool = True
) -> Dict[str, Any]:
    """
    Compare two covariance fields using Bures-Wasserstein distance.
    
    Args:
        Σ_field1: First field (e.g., monitor viewing) from compute_covariance_field()
        Σ_field2: Second field (e.g., headset viewing) from compute_covariance_field()
        normalize: If True, compute relative distance normalized by ellipse size
        
    Returns:
        Dictionary with pointwise distances, statistics, and significance assessment
    """
    assert Σ_field1['sigma_11'].shape == Σ_field2['sigma_11'].shape, \
        "Fields must have same resolution"
    
    n = Σ_field1['sigma_11'].shape[0]
    distances = np.zeros((n, n))
    relative_distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            Σ1 = np.array([
                [Σ_field1['sigma_11'][i, j], Σ_field1['sigma_12'][i, j]],
                [Σ_field1['sigma_12'][i, j], Σ_field1['sigma_22'][i, j]]
            ])
            Σ2 = np.array([
                [Σ_field2['sigma_11'][i, j], Σ_field2['sigma_12'][i, j]],
                [Σ_field2['sigma_12'][i, j], Σ_field2['sigma_22'][i, j]]
            ])
            
            d = bures_wasserstein(Σ1, Σ2)
            distances[i, j] = d
            
            if normalize:
                # Normalize by geometric mean of ellipse sizes
                size1 = np.sqrt(np.trace(Σ1))
                size2 = np.sqrt(np.trace(Σ2))
                relative_distances[i, j] = d / np.sqrt(size1 * size2) if size1 * size2 > 0 else 0
    
    # Compute benchmark: max distance any ellipse could have to a "maximally different" circle
    max_major_axis = 0
    for i in range(n):
        for j in range(n):
            Σ1 = np.array([
                [Σ_field1['sigma_11'][i, j], Σ_field1['sigma_12'][i, j]],
                [Σ_field1['sigma_12'][i, j], Σ_field1['sigma_22'][i, j]]
            ])
            eigenvalues = np.linalg.eigvalsh(Σ1)
            max_major_axis = max(max_major_axis, np.sqrt(max(eigenvalues)))
    
    Σ_benchmark = (max_major_axis ** 2) * np.eye(2)
    benchmark_distances = []
    for i in range(n):
        for j in range(n):
            Σ1 = np.array([
                [Σ_field1['sigma_11'][i, j], Σ_field1['sigma_12'][i, j]],
                [Σ_field1['sigma_12'][i, j], Σ_field1['sigma_22'][i, j]]
            ])
            benchmark_distances.append(bures_wasserstein(Σ1, Σ_benchmark))
    significance_threshold = max(benchmark_distances)
    
    return {
        'x': Σ_field1['x'],
        'y': Σ_field1['y'],
        'distances': distances,
        'relative_distances': relative_distances,
        'mean_distance': np.mean(distances),
        'max_distance': np.max(distances),
        'mean_relative': np.mean(relative_distances),
        'max_relative': np.max(relative_distances),
        'significance_threshold': significance_threshold,
        'significant_fraction': np.mean(distances > 0.1 * significance_threshold),
    }


def find_optimal_correction_matrix(
    Σ_monitor_fn,
    Σ_headset_fn,
    grid_points: np.ndarray,
    diagonal_only: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Find the linear correction matrix M that minimizes total BW distance.
    
    M transforms headset covariance: Σ_corrected = M @ Σ_headset @ M.T
    
    Args:
        Σ_monitor_fn: Function(coords) -> 2x2 covariance matrix for monitor
        Σ_headset_fn: Function(coords) -> 2x2 covariance matrix for headset
        grid_points: Array of reference locations [N, 2]
        diagonal_only: If True, optimize only diagonal elements of M
        
    Returns:
        Tuple of (optimal M, residual total BW distance)
    """
    from scipy.optimize import minimize
    
    def total_residual(m_params):
        if diagonal_only:
            M = np.diag(m_params)
        else:
            M = m_params.reshape(3, 3)
        
        total = 0
        for x in grid_points:
            Σ_m = Σ_monitor_fn(x)
            Σ_h = Σ_headset_fn(x)
            
            # Apply correction: M @ Σ_h @ M.T
            # For 2D case, use 2x2 submatrix
            M_2d = M[:2, :2] if M.shape[0] > 2 else M
            Σ_corrected = M_2d @ Σ_h @ M_2d.T
            
            total += bures_wasserstein(Σ_m, Σ_corrected) ** 2
        return total
    
    if diagonal_only:
        x0 = np.array([1., 1., 1.])
        bounds = [(0.3, 3.0)] * 3
    else:
        x0 = np.eye(3).flatten()
        bounds = [(-2., 2.)] * 9
    
    result = minimize(total_residual, x0=x0, bounds=bounds, method='L-BFGS-B')
    
    if diagonal_only:
        M_opt = np.diag(result.x)
    else:
        M_opt = result.x.reshape(3, 3)
    
    return M_opt, np.sqrt(result.fun / len(grid_points))


class FieldComparisonVisualizer:
    """
    Visualize comparison between two covariance fields (e.g., monitor vs headset).
    """
    
    def __init__(
        self,
        model_bounds: Tuple[float, float] = (-0.7, 0.7),
        figsize: Tuple[int, int] = (16, 5)
    ):
        self.model_bounds = model_bounds
        self.figsize = figsize
    
    def plot_field_comparison(
        self,
        comparison: Dict[str, Any],
        title: str = "Covariance Field Comparison (Bures-Wasserstein)",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize BW distance between two covariance fields.
        
        Args:
            comparison: Output from compare_covariance_fields()
            title: Figure title
            save_path: Optional save path
            show: Whether to display
        """
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        extent = [
            self.model_bounds[0], self.model_bounds[1],
            self.model_bounds[0], self.model_bounds[1]
        ]
        
        # Panel 1: Absolute BW distance
        ax1 = axes[0]
        im1 = ax1.imshow(
            comparison['distances'],
            extent=extent, origin='lower',
            cmap='Reds', vmin=0,
            aspect='equal'
        )
        ax1.set_title(f"BW Distance\n(mean={comparison['mean_distance']:.4f})")
        ax1.set_xlabel('Model space dimension 1')
        ax1.set_ylabel('Model space dimension 2')
        plt.colorbar(im1, ax=ax1, label='Distance')
        
        # Panel 2: Relative BW distance (normalized)
        ax2 = axes[1]
        im2 = ax2.imshow(
            comparison['relative_distances'],
            extent=extent, origin='lower',
            cmap='Reds', vmin=0, vmax=0.5,
            aspect='equal'
        )
        ax2.set_title(f"Relative BW Distance\n(mean={comparison['mean_relative']:.3f})")
        ax2.set_xlabel('Model space dimension 1')
        ax2.set_ylabel('Model space dimension 2')
        plt.colorbar(im2, ax=ax2, label='Relative distance')
        
        # Panel 3: Significance map
        ax3 = axes[2]
        threshold = 0.1 * comparison['significance_threshold']
        significant = comparison['distances'] > threshold
        im3 = ax3.imshow(
            significant.astype(float),
            extent=extent, origin='lower',
            cmap='RdYlGn_r', vmin=0, vmax=1,
            aspect='equal'
        )
        pct_sig = 100 * comparison['significant_fraction']
        ax3.set_title(f"Significant Difference\n({pct_sig:.1f}% of locations)")
        ax3.set_xlabel('Model space dimension 1')
        ax3.set_ylabel('Model space dimension 2')
        
        # Add interpretation
        mean_rel = comparison['mean_relative']
        if mean_rel < 0.1:
            interpretation = "Fields nearly identical"
        elif mean_rel < 0.2:
            interpretation = "Small difference (likely correctable)"
        elif mean_rel < 0.3:
            interpretation = "Moderate difference"
        else:
            interpretation = "Substantial geometric distortion"
        
        fig.suptitle(f"{title}\nInterpretation: {interpretation}", fontsize=12)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved figure to {save_path}")
        
        if show:
            plt.show()
        
        return fig


class WPPMVisualizer:
    """
    Visualize WPPM covariance fields and discrimination ellipses.
    
    Creates Figure 3-style plots from Hong et al. (2025).
    """
    
    def __init__(
        self,
        model_bounds: Tuple[float, float] = (-0.7, 0.7),
        grid_resolution: int = 50,
        ellipse_grid_size: int = 7,
        figsize: Tuple[int, int] = (14, 6)
    ):
        """
        Initialize visualizer.
        
        Args:
            model_bounds: Bounds of model space
            grid_resolution: Resolution for covariance field heatmaps
            ellipse_grid_size: Number of ellipses per dimension
            figsize: Figure size
        """
        self.model_bounds = model_bounds
        self.grid_resolution = grid_resolution
        self.ellipse_grid_size = ellipse_grid_size
        self.figsize = figsize
        
    def compute_covariance_field(
        self,
        get_covariance_fn,
        config: Any = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute covariance matrices over a grid of locations.
        
        Args:
            get_covariance_fn: Function(coords) -> 2x2 covariance matrix
            config: Optional config to pass to function
            
        Returns:
            Dictionary with σ²_11, σ²_22, σ_12 fields and coordinates
        """
        lb, ub = self.model_bounds
        
        # Create grid
        x = np.linspace(lb, ub, self.grid_resolution)
        y = np.linspace(lb, ub, self.grid_resolution)
        X, Y = np.meshgrid(x, y)
        
        # Compute covariance at each point
        sigma_11 = np.zeros_like(X)
        sigma_22 = np.zeros_like(X)
        sigma_12 = np.zeros_like(X)
        
        for i in range(self.grid_resolution):
            for j in range(self.grid_resolution):
                coords = np.array([X[i, j], Y[i, j]])
                
                if config is not None:
                    cov = get_covariance_fn(coords, config)
                else:
                    cov = get_covariance_fn(coords)
                
                # Convert JAX array to numpy if needed
                if hasattr(cov, 'numpy'):
                    cov = np.array(cov)
                elif hasattr(cov, '__array__'):
                    cov = np.asarray(cov)
                
                sigma_11[i, j] = cov[0, 0]
                sigma_22[i, j] = cov[1, 1]
                sigma_12[i, j] = cov[0, 1]
        
        return {
            'x': x,
            'y': y,
            'X': X,
            'Y': Y,
            'sigma_11': sigma_11,
            'sigma_22': sigma_22,
            'sigma_12': sigma_12
        }
    
    def plot_covariance_field(
        self,
        cov_field: Dict[str, np.ndarray],
        title: str = "WPPM Covariance Field",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create the 4-panel covariance visualization.
        
        Args:
            cov_field: Output from compute_covariance_field()
            title: Figure title
            save_path: Optional path to save figure
            show: Whether to display figure
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=self.figsize)
        
        # Create grid: 2x2 heatmaps on left, ellipses on right
        gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.5], wspace=0.3, hspace=0.3)
        
        # Determine symmetric colormap bounds
        max_var = max(
            np.abs(cov_field['sigma_11']).max(),
            np.abs(cov_field['sigma_22']).max()
        )
        max_cov = np.abs(cov_field['sigma_12']).max()
        
        extent = [
            self.model_bounds[0], self.model_bounds[1],
            self.model_bounds[0], self.model_bounds[1]
        ]
        
        # Panel 1: σ²_dim1 (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        im1 = ax1.imshow(
            cov_field['sigma_11'],
            extent=extent, origin='lower',
            cmap='PiYG', vmin=-max_cov, vmax=max_cov,
            aspect='equal'
        )
        ax1.set_title(r'$\sigma^2_{\mathrm{dim1}}$', fontsize=12)
        ax1.set_xlabel('Model space dimension 1')
        ax1.set_ylabel('Model space dimension 2')
        
        # Panel 2: σ_(dim1,dim2) (top-right of left section)
        ax2 = fig.add_subplot(gs[0, 1])
        im2 = ax2.imshow(
            cov_field['sigma_12'],
            extent=extent, origin='lower',
            cmap='PiYG', vmin=-max_cov, vmax=max_cov,
            aspect='equal'
        )
        ax2.set_title(r'$\sigma_{(\mathrm{dim1,dim2})}$', fontsize=12)
        ax2.set_xlabel('Model space dimension 1')
        ax2.set_ylabel('Model space dimension 2')
        
        # Panel 3: σ_(dim1,dim2) again (bottom-left) - same as panel 2 for symmetry
        ax3 = fig.add_subplot(gs[1, 0])
        im3 = ax3.imshow(
            cov_field['sigma_12'],
            extent=extent, origin='lower',
            cmap='PiYG', vmin=-max_cov, vmax=max_cov,
            aspect='equal'
        )
        ax3.set_title(r'$\sigma_{(\mathrm{dim1,dim2})}$', fontsize=12)
        ax3.set_xlabel('Model space dimension 1')
        ax3.set_ylabel('Model space dimension 2')
        
        # Panel 4: σ²_dim2 (bottom-right of left section)
        ax4 = fig.add_subplot(gs[1, 1])
        im4 = ax4.imshow(
            cov_field['sigma_22'],
            extent=extent, origin='lower',
            cmap='PiYG', vmin=-max_cov, vmax=max_cov,
            aspect='equal'
        )
        ax4.set_title(r'$\sigma^2_{\mathrm{dim2}}$', fontsize=12)
        ax4.set_xlabel('Model space dimension 1')
        ax4.set_ylabel('Model space dimension 2')
        
        # Colorbar for covariance panels
        cbar_ax = fig.add_axes([0.48, 0.15, 0.015, 0.7])
        cbar = fig.colorbar(im1, cax=cbar_ax)
        cbar.set_label('Covariance')
        
        # Panel 5: Discrimination ellipses (right side)
        ax5 = fig.add_subplot(gs[:, 2])
        self._plot_ellipses(ax5, cov_field)
        ax5.set_title('Discrimination Ellipses', fontsize=12)
        ax5.set_xlabel('Model space dimension 1')
        ax5.set_ylabel('Model space dimension 2')
        
        fig.suptitle(title, fontsize=14, y=1.02)
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved figure to {save_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def _plot_ellipses(
        self,
        ax: plt.Axes,
        cov_field: Dict[str, np.ndarray],
        scale: float = 10.0
    ):
        """
        Plot discrimination ellipses at a grid of locations.
        
        Ellipse axes are determined by eigendecomposition of covariance.
        """
        lb, ub = self.model_bounds
        
        # Grid for ellipse centers
        centers = np.linspace(lb + 0.1, ub - 0.1, self.ellipse_grid_size)
        
        ax.set_xlim(lb - 0.1, ub + 0.1)
        ax.set_ylim(lb - 0.1, ub + 0.1)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        for cx in centers:
            for cy in centers:
                # Find nearest grid point
                ix = np.argmin(np.abs(cov_field['x'] - cx))
                iy = np.argmin(np.abs(cov_field['y'] - cy))
                
                # Get covariance matrix
                cov = np.array([
                    [cov_field['sigma_11'][iy, ix], cov_field['sigma_12'][iy, ix]],
                    [cov_field['sigma_12'][iy, ix], cov_field['sigma_22'][iy, ix]]
                ])
                
                # Eigendecomposition for ellipse parameters
                try:
                    eigenvalues, eigenvectors = np.linalg.eigh(cov)
                    eigenvalues = np.maximum(eigenvalues, 1e-8)  # Ensure positive
                    
                    # Ellipse dimensions (sqrt of eigenvalues = std dev)
                    width = 2 * scale * np.sqrt(eigenvalues[0])
                    height = 2 * scale * np.sqrt(eigenvalues[1])
                    
                    # Rotation angle
                    angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
                    
                    # Draw ellipse
                    ellipse = Ellipse(
                        (cx, cy), width, height, angle=angle,
                        fill=False, edgecolor='black', linewidth=1.5
                    )
                    ax.add_patch(ellipse)
                    
                except np.linalg.LinAlgError:
                    # Skip if eigendecomposition fails
                    pass
    
    def visualize_from_wppm(
        self,
        wppm_params: Any,
        wppm_config: Any,
        title: str = "WPPM Fitted Covariance Field",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize covariance field from fitted WPPM model.
        
        Args:
            wppm_params: WPPMParams from wppm_fitter
            wppm_config: WPPMConfig from wppm_fitter
            title: Figure title
            save_path: Optional save path
            show: Whether to display
            
        Returns:
            Matplotlib figure
        """
        
        def get_cov(coords, config=None):
            coords_jax = jnp.array(coords)
            return compute_covariance(coords_jax, wppm_params.weights, wppm_config.n_basis)
        
        cov_field = self.compute_covariance_field(get_cov)
        return self.plot_covariance_field(cov_field, title, save_path, show)
    
    def visualize_synthetic(
        self,
        title: str = "Synthetic Covariance Field (Demo)",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Create visualization with synthetic covariance field for testing.
        
        Uses a spatially-varying covariance model for demonstration.
        """
        def synthetic_covariance(coords):
            x, y = coords
            
            # Spatially varying variances
            sigma_11 = 0.002 * (1 + 0.5 * np.sin(2 * np.pi * x))
            sigma_22 = 0.003 * (1 + 0.5 * np.cos(2 * np.pi * y))
            
            # Spatially varying covariance
            sigma_12 = 0.001 * np.sin(np.pi * x) * np.cos(np.pi * y)
            
            return np.array([[sigma_11, sigma_12], [sigma_12, sigma_22]])
        
        cov_field = self.compute_covariance_field(synthetic_covariance)
        return self.plot_covariance_field(cov_field, title, save_path, show)


class AEPsychPostVisualizer:
    """
    Visualize collected trial data after AEPsych experiment completes.
    
    Shows:
    - Distribution of tested locations
    - Response accuracy by region
    - Optional: GP posterior if available
    """
    
    def __init__(
        self,
        model_bounds: Tuple[float, float] = (-0.7, 0.7),
        figsize: Tuple[int, int] = (14, 5)
    ):
        self.model_bounds = model_bounds
        self.figsize = figsize
    
    def visualize_trial_distribution(
        self,
        trial_data: Dict[str, np.ndarray],
        title: str = "Trial Distribution Analysis",
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize the distribution of collected trials.
        
        Args:
            trial_data: Dictionary with 'reference_coords', 'comparison_coords', 'responses'
            title: Figure title
            save_path: Optional save path
            show: Whether to display
        """
        ref_coords = trial_data['reference_coords']
        comp_coords = trial_data['comparison_coords']
        responses = trial_data['responses']
        
        fig, axes = plt.subplots(1, 3, figsize=self.figsize)
        
        # Panel 1: Reference locations colored by response
        ax1 = axes[0]
        correct = responses > 0.5
        ax1.scatter(
            ref_coords[correct, 0], ref_coords[correct, 1],
            c='blue', alpha=0.5, s=20, label='Correct'
        )
        ax1.scatter(
            ref_coords[~correct, 0], ref_coords[~correct, 1],
            c='red', alpha=0.5, s=20, label='Incorrect'
        )
        ax1.set_xlim(self.model_bounds[0] - 0.1, self.model_bounds[1] + 0.1)
        ax1.set_ylim(self.model_bounds[0] - 0.1, self.model_bounds[1] + 0.1)
        ax1.set_aspect('equal')
        ax1.set_title('Reference Locations')
        ax1.set_xlabel('Dimension 1')
        ax1.set_ylabel('Dimension 2')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Delta magnitudes histogram
        ax2 = axes[1]
        deltas = comp_coords - ref_coords
        delta_mags = np.linalg.norm(deltas, axis=1)
        
        ax2.hist(delta_mags[correct], bins=30, alpha=0.6, color='blue', label='Correct')
        ax2.hist(delta_mags[~correct], bins=30, alpha=0.6, color='red', label='Incorrect')
        ax2.set_xlabel('Delta Magnitude')
        ax2.set_ylabel('Count')
        ax2.set_title('Response by Delta Magnitude')
        ax2.legend()
        
        # Panel 3: Accuracy heatmap
        ax3 = axes[2]
        
        # Bin reference locations and compute accuracy per bin
        n_bins = 10
        x_edges = np.linspace(self.model_bounds[0], self.model_bounds[1], n_bins + 1)
        y_edges = np.linspace(self.model_bounds[0], self.model_bounds[1], n_bins + 1)
        
        accuracy_grid = np.zeros((n_bins, n_bins))
        count_grid = np.zeros((n_bins, n_bins))
        
        for i in range(len(ref_coords)):
            x_bin = np.clip(np.digitize(ref_coords[i, 0], x_edges) - 1, 0, n_bins - 1)
            y_bin = np.clip(np.digitize(ref_coords[i, 1], y_edges) - 1, 0, n_bins - 1)
            accuracy_grid[y_bin, x_bin] += responses[i]
            count_grid[y_bin, x_bin] += 1
        
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            accuracy_grid = np.where(count_grid > 0, accuracy_grid / count_grid, np.nan)
        
        im = ax3.imshow(
            accuracy_grid, extent=[self.model_bounds[0], self.model_bounds[1],
                                   self.model_bounds[0], self.model_bounds[1]],
            origin='lower', cmap='RdYlGn', vmin=0.33, vmax=1.0,
            aspect='equal'
        )
        ax3.set_title('Accuracy by Region')
        ax3.set_xlabel('Dimension 1')
        ax3.set_ylabel('Dimension 2')
        plt.colorbar(im, ax=ax3, label='Accuracy')
        
        fig.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"Saved figure to {save_path}")
        
        if show:
            plt.show()
        
        return fig


def main():
    parser = argparse.ArgumentParser(description="WPPM Post-experiment Visualizer")
    parser.add_argument(
        '--mode', choices=['demo', 'wppm', 'trials'], default='demo',
        help='Visualization mode: demo (synthetic), wppm (fitted model), trials (trial data)'
    )
    parser.add_argument(
        '--data', type=str, default=None,
        help='Path to trial data CSV or WPPM params file'
    )
    parser.add_argument(
        '--resolution', type=int, default=50,
        help='Grid resolution for covariance field'
    )
    parser.add_argument(
        '--ellipses', type=int, default=7,
        help='Number of ellipses per dimension'
    )
    parser.add_argument(
        '--save', type=str, default=None,
        help='Path to save figure'
    )
    parser.add_argument(
        '--no-show', action='store_true',
        help='Do not display figure'
    )
    args = parser.parse_args()
    
    print("=" * 60)
    print("WPPM Post-experiment Visualizer")
    print("=" * 60)
    
    if args.mode == 'demo':
        print("Creating synthetic covariance field demonstration...")
        viz = WPPMVisualizer(
            grid_resolution=args.resolution,
            ellipse_grid_size=args.ellipses
        )
        viz.visualize_synthetic(
            save_path=args.save,
            show=not args.no_show
        )
        
    elif args.mode == 'wppm':
        if args.data is None:
            print("Error: --data required for WPPM visualization")
            return
        
        print(f"Loading WPPM data from {args.data}...")

        # Check file extension to determine format
        if args.data.endswith('.json'):
            # Load pre-fitted WPPM parameters from JSON
            import json
            with open(args.data, "r") as f:
                wppm_data = json.load(f)

            weights = np.array(wppm_data['params']['weights'])
            cfg = wppm_data['config']
            
            params = WPPMParams(weights=jnp.array(weights))
            config = WPPMConfig(
                n_basis=cfg['n_basis'],
                gamma=cfg['gamma'],
                epsilon=cfg['epsilon'],
                n_mc_samples=cfg['n_mc_samples'],
                coord_range=tuple(cfg['coord_range']),
                bandwidth=cfg['bandwidth'],
            )
            
        elif args.data.endswith('.npz'):
            params, config, losses, metadata = load_wppm(args.data)
            print(f"Loaded {params.weights.size} parameters")
                
        elif args.data.endswith('.csv'):
            # Load trial data and fit WPPM parameters
            print("CSV file detected - fitting WPPM parameters from trial data...")
            # Load CSV trial data
            data = np.loadtxt(args.data, delimiter=',', skiprows=1)
            if data.shape[1] >= 5:
                # Format: ref_x,ref_y,comp_x,comp_y,response
                ref_coords = data[:, :2]
                comp_coords = data[:, 2:4]
                responses = data[:, 4]
            else:
                print("Error: CSV must have at least 5 columns: ref_x, ref_y, comp_x, comp_y, response")
                return

            # Convert to WPPM format
            ref_coords_jax = jnp.array(ref_coords)
            comp_coords_jax = jnp.array(comp_coords)
            responses_jax = jnp.array(responses)

            trial_data = TrialData(
                reference_coords=ref_coords_jax,
                comparison_coords=comp_coords_jax,
                responses=responses_jax
            )

            print(f"Fitting WPPM to {len(responses)} trials...")

            # Configure WPPM
            config = WPPMConfig(
                n_basis=5,
                n_mc_samples=500,
                bandwidth=1.0,
                gamma=3e-4,
                epsilon=0.5
            )

            # Fit model
            params, losses = fit_wppm(
                trial_data,
                config=config,
                n_iterations=200,
                learning_rate=1e-2,
                key=jr.PRNGKey(42),
                verbose=True
            )

            print("WPPM fitting completed.")
            
            # Save fitted model
            save_path_base = args.data.replace('.csv', '')
            save_path_npz = f"{save_path_base}_wppm.npz"
            save_wppm(
                save_path_npz,
                params,
                config,
                losses=losses,
                metadata={'n_trials': len(responses), 'source': args.data}
            )
            print(f"Saved fitted model to {save_path_npz}")
            
        else:
            print(f"Error: Unsupported file format '{args.data}'. Use .json, .npz, or .csv")
            return

        # Visualize the fitted model
        viz = WPPMVisualizer(
            grid_resolution=args.resolution,
            ellipse_grid_size=args.ellipses
        )
        viz.visualize_from_wppm(
            params,
            config,
            save_path=args.save,
            show=not args.no_show
        )
        
    elif args.mode == 'trials':
        if args.data is None:
            print("Error: --data required for trial visualization")
            return
        
        print(f"Loading trial data from {args.data}...")
        
        # Load CSV data
        data = np.loadtxt(args.data, delimiter=',', skiprows=1)
        trial_data = {
            'reference_coords': data[:, :2],
            'comparison_coords': data[:, 2:4],
            'responses': data[:, 4]
        }
        
        viz = AEPsychPostVisualizer()
        viz.visualize_trial_distribution(
            trial_data,
            save_path=args.save,
            show=not args.no_show
        )
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()