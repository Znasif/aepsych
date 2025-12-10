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
import numpy as np
from typing import Optional, Tuple, Dict, Any
import argparse
from pathlib import Path


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
        # Import WPPM functions
        try:
            from wppm_fitter import compute_covariance
            import jax.numpy as jnp
        except ImportError:
            raise ImportError("wppm_fitter module required for WPPM visualization")
        
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
        
        print(f"Loading WPPM params from {args.data}...")
        # Load WPPM params and visualize
        # (Implementation depends on how params are saved)
        raise NotImplementedError("WPPM params loading not yet implemented")
        
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