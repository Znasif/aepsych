#!/usr/bin/env python3
"""
Simulation for inferring hidden color transformation matrix M.

Setup:
- T (Target): Inaccessible region showing M_true × XYZ (observer sees, we don't know M_true)
- P (Probe): Accessible region showing baseline × XYZ (starts as I, advances when C wins)
- C (Comparison): Accessible region showing M_star × XYZ (AEPsych proposal)

Per-reference flow:
1. Fix reference XYZ
2. AEPsych proposes M_star = diag(m00, m11, m22)
3. Observer 2AFC: "Is C or P closer to T?"
4. If C picked → baseline = M_star, get new proposal
5. Repeat until convergence (~50% accuracy)
6. Record (XYZ, baseline_final × XYZ) as matched pair

Final regression: M_est = argmin Σᵢ ‖M × XYZᵢ - Yᵢ‖²
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import pygame
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adaptive_engine.color import ColorTransformations
from adaptive_engine.simple_stimulus import StimulusPresentation
# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    # True hidden matrix (diagonal to start)
    m_true_diagonal: Tuple[float, float, float] = (1.15, 0.92, 1.08)
    
    # Observer noise (Gaussian std in each channel)
    observer_noise_std: float = 0.02
    
    # AEPsych GP bounds for diagonal elements
    m_bounds: Tuple[float, float] = (0.7, 1.4)
    
    # Convergence: window of trials to assess accuracy
    convergence_window: int = 8
    convergence_threshold: float = 0.10  # Below this = converged (near 50%)
    
    # Maximum trials per reference before forced termination
    max_trials_per_reference: int = 50
    
    # Minimum trials before checking convergence
    min_trials_per_reference: int = 15
    
    # Whether to reset GP when moving to new reference
    reset_gp_on_new_reference: bool = True
    
    # Whether to reset baseline when moving to new reference  
    reset_baseline_on_new_reference: bool = True
    
    # Full 3x3 matrix support
    use_full_matrix: bool = False


# =============================================================================
# Munsell Color Chips (32 reference colors in XYZ-like space)
# =============================================================================

def get_munsell_references() -> np.ndarray:
    """
    Generate 32 Munsell-like reference colors.
    
    Returns array of shape [32, 3] in normalized color space.
    """
    # Approximate Munsell chips spanning hue, value, chroma
    # Using a simplified representation in a [-1, 1]^3 space
    references = []
    
    # Sample hues around the color circle at different values/chromas
    n_hues = 8
    n_values = 2
    n_chromas = 2
    
    for v_idx in range(n_values):
        value = 0.3 + 0.4 * v_idx  # 0.3, 0.7
        for c_idx in range(n_chromas):
            chroma = 0.3 + 0.3 * c_idx  # 0.3, 0.6
            for h_idx in range(n_hues):
                hue = 2 * np.pi * h_idx / n_hues
                
                # Convert to approximate XYZ-like coordinates
                x = chroma * np.cos(hue)
                y = value - 0.5  # Center around 0
                z = chroma * np.sin(hue)
                
                references.append([x, y, z])
    
    return np.array(references)


# =============================================================================
# Observer Model
# =============================================================================

class Observer:
    """
    Simulated observer with Gaussian perceptual noise.
    """
    
    def __init__(self, m_true: np.ndarray, noise_std: float = 0.02):
        """
        Args:
            m_true: True hidden transformation matrix [3, 3]
            noise_std: Standard deviation of Gaussian perceptual noise
        """
        self.m_true = m_true
        self.noise_std = noise_std
        pygame.init()
        self.screen_width = 1920
        self.screen_height = 1080
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        pygame.display.set_caption("Color Discrimination - 2AFC")
        self.clock = pygame.time.Clock()
        ppd = 50  # pixels per degree
        self.stimulus_size_px = int(2.0 * ppd)
        self.stimulus_spacing_px = int(1.0 * ppd)
        self._calculate_positions()

    def _calculate_positions(self):
        """Calculate stimulus positions in triangular arrangement."""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        radius = self.stimulus_spacing_px

        # Equilateral triangle arrangement
        # Top vertex at 12 o'clock, other two at 4 and 8 o'clock
        self.stimulus_positions = [
            (int(center_x), int(center_y - radius)),  # Top (position 1)
            (int(center_x - radius * 0.866), int(center_y + radius * 0.5)),  # Bottom-left (position 2)
            (int(center_x + radius * 0.866), int(center_y + radius * 0.5)),  # Bottom-right (position 3)
        ]
    
    def perceive(self, color: np.ndarray) -> np.ndarray:
        """
        Perceive a color with added noise.
        
        Args:
            color: Input color [3]
            
        Returns:
            Perceived color with noise [3]
        """
        noise = np.random.randn(3) * self.noise_std
        return color + noise
    
    def get_target_appearance(self, xyz: np.ndarray) -> np.ndarray:
        """
        Get the appearance of target T (through hidden matrix).
        
        Args:
            xyz: Reference color [3]
            
        Returns:
            T = M_true @ xyz (perceived with noise)
        """
        t_true = self.m_true @ xyz
        return t_true #self.perceive(t_true)
    
    def compare_2afc(
        self,
        xyz: np.ndarray,
        baseline: np.ndarray,
        m_star: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Perform 2AFC: Is C (m_star @ xyz) closer to T than P (baseline @ xyz)?
        
        Args:
            xyz: Reference color [3]
            baseline: Current baseline matrix [3, 3]
            m_star: Proposed comparison matrix [3, 3]
            
        Returns:
            Tuple of (c_chosen, dist_t_c, dist_t_p)
        """
        # Compute appearances (with independent noise samples)
        t_perceived = self.get_target_appearance(xyz)
        p_color = baseline @ xyz
        c_color = m_star @ xyz
        
        p_perceived = self.perceive(p_color)
        c_perceived = self.perceive(c_color)
        
        # Perceptual distances
        dist_t_p = np.linalg.norm(t_perceived - p_perceived)
        dist_t_c = np.linalg.norm(t_perceived - c_perceived)

        self.screen.fill((255, 255, 255))

        for pos, rgb in zip(self.stimulus_positions, np.array([t_perceived, p_perceived, c_perceived])):
            color = self._rgb_float_to_int(rgb)
            pygame.draw.circle(self.screen, color, pos, self.stimulus_size_px)

        pygame.display.flip()
        
        # Observer chooses the closer one (with noise already included)
        c_chosen = dist_t_c < dist_t_p
        pygame.time.wait(10)
        return c_chosen, dist_t_c, dist_t_p
    
    def _rgb_float_to_int(self, rgb: np.ndarray) -> Tuple[int, int, int]:
        """Convert RGB from [0,1] float to [0,255] int."""
        rgb = np.asarray(rgb).flatten()
        rgb = np.clip(rgb, 0.0, 1.0)
        return tuple(int(c * 255) for c in rgb)


# =============================================================================
# Simple GP-like Acquisition (AEPsych-style)
# =============================================================================

class SimpleGP:
    """
    Simple GP surrogate for M_star proposals.
    
    Uses UCB acquisition for exploration-exploitation.
    """
    
    def __init__(
        self,
        bounds: Tuple[float, float],
        n_dims: int = 3,
        length_scale: float = 0.15,
        noise_var: float = 0.2,
        n_initial_random: int = 5
    ):
        self.bounds = bounds
        self.n_dims = n_dims
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.n_initial_random = n_initial_random
        
        self.X: List[np.ndarray] = []  # Observations: M_star params
        self.y: List[float] = []  # Outcomes: 1 if C chosen, 0 otherwise
        
    def reset(self):
        """Reset observations."""
        self.X = []
        self.y = []
    
    def add_observation(self, m_params: np.ndarray, outcome: float):
        """Add an observation."""
        self.X.append(m_params.copy())
        self.y.append(outcome)
    
    def _kernel(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """RBF kernel."""
        diff = x1 - x2
        return np.exp(-0.5 * np.sum(diff**2) / self.length_scale**2)
    
    def _compute_posterior(self, x: np.ndarray) -> Tuple[float, float]:
        """Compute posterior mean and std at x."""
        if len(self.X) == 0:
            return 0.5, 1.0  # Prior
        
        X = np.array(self.X)
        y = np.array(self.y)
        n = len(y)
        
        # Kernel matrix
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i, j] = self._kernel(X[i], X[j])
        K += self.noise_var * np.eye(n)
        
        # Kernel vector
        k = np.array([self._kernel(x, X[i]) for i in range(n)])
        
        # Posterior
        try:
            K_inv = np.linalg.inv(K)
            mean = k @ K_inv @ y
            var = 1.0 - k @ K_inv @ k
            var = max(var, 1e-6)
        except np.linalg.LinAlgError:
            mean = 0.5
            var = 1.0
        
        return mean, np.sqrt(var)
    
    def propose(self, n_candidates: int = 200, ucb_beta: float = 1.5) -> np.ndarray:
        """
        Propose next M_star parameters.
        
        Uses random sampling initially, then UCB.
        
        Returns diagonal elements [m00, m11, m22].
        """
        # Initial random exploration
        if len(self.X) < self.n_initial_random:
            return np.random.uniform(
                self.bounds[0], self.bounds[1],
                size=self.n_dims
            )
        
        # Random candidates
        candidates = np.random.uniform(
            self.bounds[0], self.bounds[1],
            size=(n_candidates, self.n_dims)
        )
        
        # Also include perturbations of best observed
        if len(self.X) > 0:
            best_idx = np.argmax(self.y)
            best_x = self.X[best_idx]
            for _ in range(20):
                perturbed = best_x + np.random.randn(self.n_dims) * 0.05
                perturbed = np.clip(perturbed, self.bounds[0], self.bounds[1])
                candidates = np.vstack([candidates, perturbed])
        
        # Evaluate UCB
        best_ucb = -np.inf
        best_x = candidates[0]
        
        for x in candidates:
            mean, std = self._compute_posterior(x)
            ucb = mean + ucb_beta * std
            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x
        
        return best_x


# =============================================================================
# Matching Session (per reference color)
# =============================================================================

@dataclass
class MatchingResult:
    """Result from matching a single reference color."""
    reference_xyz: np.ndarray
    matched_output: np.ndarray  # baseline_final @ reference_xyz
    baseline_final: np.ndarray  # Final baseline matrix
    n_trials: int
    converged: bool
    accuracy_history: List[float] = field(default_factory=list)


def run_matching_session(
    observer: Observer,
    gp: SimpleGP,
    reference_xyz: np.ndarray,
    config: SimulationConfig,
    initial_baseline: Optional[np.ndarray] = None
) -> MatchingResult:
    """
    Run matching session for a single reference color.
    
    Args:
        observer: Observer model
        gp: GP for proposals
        reference_xyz: Reference color [3]
        config: Simulation configuration
        initial_baseline: Starting baseline (default: identity)
        
    Returns:
        MatchingResult with matched output
    """
    # Initialize baseline
    if initial_baseline is None:
        baseline = np.eye(3)
    else:
        baseline = initial_baseline.copy()
    
    # Track outcomes for convergence
    outcomes: List[bool] = []
    accuracy_history: List[float] = []
    baseline_advances: int = 0  # Count how many times C beat P
    
    n_trials = 0
    converged = False
    
    while n_trials < config.max_trials_per_reference:
        # Get proposal from GP
        m_params = gp.propose()
        
        # Build M_star matrix
        if config.use_full_matrix:
            m_star = np.diag(m_params)
        else:
            m_star = np.diag(m_params)
        
        # Observer comparison
        c_chosen, dist_t_c, dist_t_p = observer.compare_2afc(
            reference_xyz, baseline, m_star
        )
        
        # Record outcome
        outcomes.append(c_chosen)
        gp.add_observation(m_params, float(c_chosen))
        n_trials += 1
        
        # Update baseline if C was chosen
        if c_chosen:
            baseline = m_star.copy()
            baseline_advances += 1
        
        # Check convergence only after baseline has advanced at least once
        # AND we've done minimum trials
        # AND recent accuracy is near chance (can't improve further)
        if n_trials >= config.min_trials_per_reference and baseline_advances >= 1:
            recent = outcomes[-config.convergence_window:]
            accuracy = np.mean(recent)
            accuracy_history.append(accuracy)
            
            # Converged when: we've improved baseline but now can't do better
            # (accuracy near 50% means current baseline ≈ best possible)
            if accuracy < config.convergence_threshold and len(outcomes) >= config.convergence_window:
                # Additional check: ensure we're not just starting out
                # Require at least 2 advances or low accuracy for extended window
                if baseline_advances >= 2 or n_trials >= config.min_trials_per_reference + 5:
                    converged = True
                    break
    
    # Compute final matched output
    matched_output = baseline @ reference_xyz
    
    return MatchingResult(
        reference_xyz=reference_xyz,
        matched_output=matched_output,
        baseline_final=baseline,
        n_trials=n_trials,
        converged=converged,
        accuracy_history=accuracy_history
    )


# =============================================================================
# Full Simulation
# =============================================================================

@dataclass
class SimulationResults:
    """Results from full simulation."""
    m_true: np.ndarray
    m_estimated: np.ndarray
    matching_results: List[MatchingResult]
    regression_error: float
    
    def print_summary(self):
        """Print summary of results."""
        print("\n" + "=" * 60)
        print("SIMULATION RESULTS")
        print("=" * 60)
        
        print(f"\nTrue M: {self.m_true}")
        print(f"Estimated M:\n{self.m_estimated}")
        print(f"Error: {np.linalg.norm(self.m_true - self.m_estimated):.4f}")
        
        print(f"\nFull matrix error (Frobenius): {self.regression_error:.4f}")
        
        n_converged = sum(1 for r in self.matching_results if r.converged)
        total_trials = sum(r.n_trials for r in self.matching_results)
        print(f"\nReferences converged: {n_converged}/{len(self.matching_results)}")
        print(f"Total trials: {total_trials}")
        print(f"Avg trials per reference: {total_trials / len(self.matching_results):.1f}")


def run_simulation(config: SimulationConfig) -> SimulationResults:
    """
    Run full simulation.
    
    Args:
        config: Simulation configuration
        
    Returns:
        SimulationResults
    """
    print("=" * 60)
    print("HIDDEN MATRIX INFERENCE SIMULATION")
    print("=" * 60)
    
    # Build true hidden matrix
    if config.use_full_matrix:
        # For now, still use diagonal even if full matrix enabled
        m_true = np.array([[1.44, -0.21, 0.01], [-0.38, 2.15, -0.67], [-0.61, -0.73, 2.9]])
    else:
        m_true = np.diag(config.m_true_diagonal)
    
    print(f"\nTrue M: {m_true}")
    print(f"Observer noise std: {config.observer_noise_std}")
    
    # Create observer
    observer = Observer(m_true, noise_std=config.observer_noise_std)
    
    # Create GP
    gp = SimpleGP(bounds=config.m_bounds, n_dims=3)
    
    # Get reference colors
    references = get_munsell_references()
    print(f"Number of reference colors: {len(references)}")
    
    # Run matching for each reference
    matching_results: List[MatchingResult] = []
    
    for i, ref_xyz in enumerate(references):
        print(f"\nReference {i+1}/{len(references)}: {ref_xyz}")
        
        # Reset GP for new reference (if configured)
        if config.reset_gp_on_new_reference:
            gp.reset()
        
        # Initial baseline
        if config.reset_baseline_on_new_reference or i == 0:
            initial_baseline = None
        else:
            # Use previous final baseline as starting point
            initial_baseline = matching_results[-1].baseline_final
        
        # Run matching
        result = run_matching_session(
            observer, gp, ref_xyz, config, initial_baseline
        )
        matching_results.append(result)
        pygame.time.wait(1500)
        status = "✓" if result.converged else "✗"
        print(f"  {status} Trials: {result.n_trials}, Converged: {result.converged},Final Baseline:\n{result.baseline_final}, \nMatched Output: {result.matched_output}")
    
    # Collect matched pairs for regression
    X = np.array([r.reference_xyz for r in matching_results])  # [N, 3]
    Y = np.array([r.matched_output for r in matching_results])  # [N, 3]
    
    # Least squares regression: Y ≈ X @ M.T  →  M ≈ (X.T @ X)^{-1} @ X.T @ Y
    # Or equivalently: Y.T ≈ M @ X.T
    m_estimated, residuals, rank, s = np.linalg.lstsq(X, Y, rcond=None)
    m_estimated = m_estimated.T  # [3, 3]
    
    # Compute regression error
    Y_pred = X @ m_estimated.T
    regression_error = np.linalg.norm(Y - Y_pred, 'fro') / len(Y)
    
    return SimulationResults(
        m_true=m_true,
        m_estimated=m_estimated,
        matching_results=matching_results,
        regression_error=regression_error
    )


# =============================================================================
# Visualization
# =============================================================================

def plot_results(results: SimulationResults, save_path: Optional[str] = None):
    """
    Plot simulation results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Panel 1: True vs Estimated diagonal elements
    ax1 = axes[0, 0]
    diag_true = np.diag(results.m_true)
    diag_est = np.diag(results.m_estimated)
    
    x_pos = np.arange(3)
    width = 0.35
    ax1.bar(x_pos - width/2, diag_true, width, label='True M', color='blue', alpha=0.7)
    ax1.bar(x_pos + width/2, diag_est, width, label='Estimated M', color='orange', alpha=0.7)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(['m00', 'm11', 'm22'])
    ax1.set_ylabel('Value')
    ax1.set_title('Diagonal Elements: True vs Estimated')
    ax1.legend()
    ax1.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    # Panel 2: Matched vs True outputs
    ax2 = axes[0, 1]
    
    X = np.array([r.reference_xyz for r in results.matching_results])
    Y_matched = np.array([r.matched_output for r in results.matching_results])
    Y_true = X @ results.m_true.T
    
    # Plot each channel
    for ch, (color, label) in enumerate(zip(['red', 'green', 'blue'], ['R', 'G', 'B'])):
        ax2.scatter(Y_true[:, ch], Y_matched[:, ch], c=color, alpha=0.6, label=label)
    
    # Perfect match line
    lims = [min(Y_true.min(), Y_matched.min()), max(Y_true.max(), Y_matched.max())]
    ax2.plot(lims, lims, 'k--', alpha=0.5)
    ax2.set_xlabel('True output (M_true @ XYZ)')
    ax2.set_ylabel('Matched output (M_star_final @ XYZ)')
    ax2.set_title('Matched vs True Outputs')
    ax2.legend()
    ax2.set_aspect('equal')
    
    # Panel 3: Trials per reference
    ax3 = axes[1, 0]
    trials = [r.n_trials for r in results.matching_results]
    converged = [r.converged for r in results.matching_results]
    colors = ['green' if c else 'red' for c in converged]
    ax3.bar(range(len(trials)), trials, color=colors, alpha=0.7)
    ax3.set_xlabel('Reference index')
    ax3.set_ylabel('Number of trials')
    ax3.set_title('Trials per Reference (green=converged, red=timeout)')
    ax3.axhline(y=np.mean(trials), color='blue', linestyle='--', label=f'Mean: {np.mean(trials):.1f}')
    ax3.legend()
    
    # Panel 4: Accuracy histories
    ax4 = axes[1, 1]
    for i, r in enumerate(results.matching_results):
        if r.accuracy_history:
            ax4.plot(r.accuracy_history, alpha=0.3, color='blue')
    ax4.axhline(y=0.5, color='red', linestyle='--', label='Chance (50%)')
    ax4.axhline(y=0.55, color='orange', linestyle='--', label='Convergence threshold')
    ax4.set_xlabel('Trial (after min trials)')
    ax4.set_ylabel('Recent accuracy')
    ax4.set_title('Accuracy History (all references)')
    ax4.legend()
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nSaved figure to {save_path}")
    
    plt.show()


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Hidden Matrix Inference Simulation")
    parser.add_argument('--noise', type=float, default=0.02, help='Observer noise std')
    parser.add_argument('--m00', type=float, default=1.15, help='True M diagonal element 0')
    parser.add_argument('--m11', type=float, default=0.92, help='True M diagonal element 1')
    parser.add_argument('--m22', type=float, default=1.08, help='True M diagonal element 2')
    parser.add_argument('--max-trials', type=int, default=50, help='Max trials per reference')
    parser.add_argument('--save', type=str, default=None, help='Path to save figure')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    np.random.seed(args.seed)
    
    # Configure
    config = SimulationConfig(
        m_true_diagonal=(args.m00, args.m11, args.m22),
        observer_noise_std=args.noise,
        max_trials_per_reference=args.max_trials,
        use_full_matrix=False
    )
    
    # Run simulationq
    results = run_simulation(config)
    
    # Print summary
    results.print_summary()
    
    # Plot results
    plot_results(results, save_path=args.save)


if __name__ == "__main__":
    main()