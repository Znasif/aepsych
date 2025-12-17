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

# Import AEPsych components for parameter optimization
from aepsych.server import AEPsychServer
from aepsych.config import Config
from aepsych.generators import OptimizeAcqfGenerator
from aepsych.acquisition import EAVC
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
    
    # AEPsych parameter bounds for diagonal elements m00, m11 (m22 fixed at 1.0)
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
    Generate 32 Munsell-like reference colors in 2D model space.

    Returns array of shape [32, 2] in model space [-1, 1]².
    """
    # Approximate Munsell chips spanning hue and chroma
    # Using representation in model space (2D isoluminant plane)
    references = []

    # Sample hues around the color circle at different chromas
    n_hues = 8
    n_chromas = 4  # Increased for better coverage in 2D

    for c_idx in range(n_chromas):
        chroma = 0.2 + 0.6 * c_idx / (n_chromas - 1)  # 0.2 to 0.8
        for h_idx in range(n_hues):
            hue = 2 * np.pi * h_idx / n_hues

            # Convert to 2D model space coordinates
            w_dim1 = chroma * np.cos(hue)
            w_dim2 = chroma * np.sin(hue)

            references.append([w_dim1, w_dim2])

    return np.array(references)


# =============================================================================
# Observer Model
# =============================================================================

class Observer:
    """
    Simulated observer with Gaussian perceptual noise.
    Works with model space inputs, converts to RGB for matrix operations.
    """

    def __init__(self, m_true: np.ndarray, noise_std: float = 0.02):
        """
        Args:
            m_true: True hidden transformation matrix [3, 3]
            noise_std: Standard deviation of Gaussian perceptual noise
        """
        self.m_true = m_true
        self.noise_std = noise_std

        # Initialize color transformations
        self.color_transform = ColorTransformations()

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
    
    def perceive(self, rgb: np.ndarray) -> np.ndarray:
        """
        Perceive a color with added noise.

        Args:
            rgb: Input RGB color [3] in [0, 1] range

        Returns:
            Perceived RGB color with noise [3] in [0, 1] range
        """
        noise = np.random.randn(3) * self.noise_std
        perceived = rgb + noise
        return np.clip(perceived, 0.0, 1.0)  # Clip to valid RGB range

    def get_target_appearance(self, model_coords: np.ndarray) -> np.ndarray:
        """
        Get the appearance of target T (through hidden matrix).

        Args:
            model_coords: Reference color in model space [2]

        Returns:
            T = M_true @ RGB(model_coords) (perceived with noise)
        """
        # Convert model space to RGB
        rgb = self.color_transform.model_space_to_rgb(model_coords)
        # Apply true transformation matrix
        t_true = self.m_true @ rgb
        # Clip to valid RGB range
        t_true = np.clip(t_true, 0.0, 1.0)
        return t_true  # self.perceive(t_true)
    
    def compare_2afc(
        self,
        model_coords: np.ndarray,
        baseline: np.ndarray,
        m_star: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Perform 2AFC: Is C (m_star @ RGB(model_coords)) closer to T than P (baseline @ RGB(model_coords))?

        Args:
            model_coords: Reference color in model space [2]
            baseline: Current baseline matrix [3, 3]
            m_star: Proposed comparison matrix [3, 3]

        Returns:
            Tuple of (c_chosen, dist_t_c, dist_t_p)
        """
        # Convert model space to RGB
        rgb = self.color_transform.model_space_to_rgb(model_coords)

        # Compute appearances (with independent noise samples)
        t_perceived = self.get_target_appearance(model_coords)
        p_color = baseline @ rgb
        c_color = m_star @ rgb

        # Clip to valid RGB range
        p_color = np.clip(p_color, 0.0, 1.0)
        c_color = np.clip(c_color, 0.0, 1.0)

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
# AEPsych Parameter Optimizer
# =============================================================================

class AEPsychParameterOptimizer:
    """
    AEPsych wrapper for optimizing matrix diagonal parameters.

    Uses EAVC acquisition for parameter optimization in 2D space (m00, m11).
    """

    def __init__(self, bounds: Tuple[float, float], n_initial_random: int = 5):
        """
        Initialize AEPsych for parameter optimization.

        Args:
            bounds: Parameter bounds [min, max] for matrix diagonal elements
            n_initial_random: Number of initial random samples
        """
        self.bounds = bounds
        self.n_initial_random = n_initial_random
        self.trial_count = 0

        # Observations storage
        self.X_observed = []  # Parameter values tried
        self.y_observed = []  # Outcomes (1=improvement, 0=no improvement)

        # Set up AEPsych configuration
        self._setup_aepsych()

    def _setup_aepsych(self):
        """Set up AEPsych server for 2D parameter optimization."""
        lb, ub = self.bounds

        # Configuration for parameter optimization using UCB
        # UCB is more robust than EAVC for parameter optimization
        config_str = f"""
[common]
parnames = [m00, m11]
stimuli_per_trial = 1
outcome_types = [binary]
strategy_names = [init_strat, opt_strat]

[m00]
par_type = continuous
lower_bound = {lb}
upper_bound = {ub}

[m11]
par_type = continuous
lower_bound = {lb}
upper_bound = {ub}

[init_strat]
min_asks = {self.n_initial_random}
generator = SobolGenerator

[opt_strat]
min_asks = 1000
refit_every = 10
generator = OptimizeAcqfGenerator
model = GPClassificationModel

[GPClassificationModel]
inducing_size = 20
mean_covar_factory = default_mean_covar_factory

[OptimizeAcqfGenerator]
restarts = 5
samps = 500
acqf = qLogNoisyExpectedImprovement

[qLogNoisyExpectedImprovement]
objective = ProbitObjective
"""

        # Create and initialize server
        self.aepsych_config = Config(config_str=config_str)
        self.server = AEPsychServer()

        setup_message = {
            "type": "setup",
            "message": {"config_str": config_str}
        }

        try:
            response = self.server.handle_request(setup_message)
            print(f"AEPsych parameter optimizer setup: {response}")
        except Exception as e:
            print(f"Warning: AEPsych setup failed: {e}")
            print("Falling back to random optimization")
            self.server = None

    def reset(self):
        """Reset the optimizer for a new reference."""
        self.trial_count = 0
        self.X_observed = []
        self.y_observed = []
        self._setup_aepsych()

    def propose(self) -> np.ndarray:
        """
        Get next parameter proposal from AEPsych.

        Returns:
            Array [m00, m11] for diagonal matrix elements
        """
        self.trial_count += 1

        # Fallback to random if AEPsych not available
        if self.server is None:
            return np.random.uniform(self.bounds[0], self.bounds[1], size=2)

        message = {"type": "ask", "message": {}}

        try:
            response = self.server.handle_request(message)

            if isinstance(response, dict) and "config" in response:
                config = response["config"]
                m00 = float(config.get("m00", [self.bounds[0]])[0])
                m11 = float(config.get("m11", [self.bounds[0]])[0])
                return np.array([m00, m11])
            else:
                # Fallback to random
                print(f"Warning: Unexpected AEPsych response: {response}")
                return np.random.uniform(self.bounds[0], self.bounds[1], size=2)

        except Exception as e:
            print(f"Error asking AEPsych for parameters: {e}")
            return np.random.uniform(self.bounds[0], self.bounds[1], size=2)

    def add_observation(self, params: np.ndarray, outcome: float):
        """
        Tell AEPsych the result of the parameter trial.

        Args:
            params: Parameter values [m00, m11]
            outcome: Binary outcome (1=improvement, 0=no improvement)
        """
        # Store locally
        self.X_observed.append(params.copy())
        self.y_observed.append(outcome)

        # Skip AEPsych if not available
        if self.server is None:
            return

        message = {
            "type": "tell",
            "message": {
                "config": {
                    "m00": [float(params[0])],
                    "m11": [float(params[1])]
                },
                "outcome": int(outcome)
            }
        }

        try:
            self.server.handle_request(message)
        except Exception as e:
            print(f"Warning: Failed to tell AEPsych: {e}")


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
    reference_xyz: np.ndarray  # Now stores 2D model coordinates
    matched_output: np.ndarray  # baseline_final @ RGB(reference_xyz)
    baseline_final: np.ndarray  # Final baseline matrix
    n_trials: int
    converged: bool
    accuracy_history: List[float] = field(default_factory=list)


def run_matching_session(
    observer: Observer,
    optimizer: AEPsychParameterOptimizer,
    reference_model: np.ndarray,
    config: SimulationConfig,
    initial_baseline: Optional[np.ndarray] = None
) -> MatchingResult:
    """
    Run matching session for a single reference color.

    Args:
        observer: Observer model
        optimizer: AEPsych parameter optimizer
        reference_model: Reference color in model space [2]
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
        # Get proposal from optimizer
        if hasattr(optimizer, 'propose'):  # AEPsych optimizer
            m_params = optimizer.propose()
        else:  # SimpleGP fallback
            m_params = optimizer.propose()

        # Build M_star matrix (diagonal with proposed parameters)
        m_star = np.diag([m_params[0], m_params[1], 1.0])  # m22 fixed at 1.0

        # Observer comparison
        c_chosen, dist_t_c, dist_t_p = observer.compare_2afc(
            reference_model, baseline, m_star
        )

        # Record outcome
        outcomes.append(c_chosen)
        if hasattr(optimizer, 'add_observation'):  # AEPsych optimizer
            optimizer.add_observation(m_params, float(c_chosen))
        else:  # SimpleGP fallback
            optimizer.add_observation(m_params, float(c_chosen))
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
    
    # Compute final matched output (convert model space to RGB, apply baseline, keep in RGB)
    reference_rgb = observer.color_transform.model_space_to_rgb(reference_model)
    matched_output = baseline @ reference_rgb
    matched_output = np.clip(matched_output, 0.0, 1.0)  # Ensure valid RGB

    return MatchingResult(
        reference_xyz=reference_model,  # Store model coordinates
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
        print(f"Estimated M: {self.m_estimated}")
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
    
    # Create AEPsych parameter optimizer (with fallback to SimpleGP)
    try:
        optimizer = AEPsychParameterOptimizer(bounds=config.m_bounds)
        print("Using AEPsych parameter optimizer")
        use_aepsych = True
    except Exception as e:
        print(f"AEPsych initialization failed: {e}")
        print("Falling back to SimpleGP")
        optimizer = SimpleGP(bounds=config.m_bounds, n_dims=2)  # 2D for m00, m11
        use_aepsych = False
    
    # Get reference colors
    references = get_munsell_references()
    print(f"Number of reference colors: {len(references)}")
    
    # Run matching for each reference
    matching_results: List[MatchingResult] = []
    
    for i, ref_xyz in enumerate(references):
        print(f"\nReference {i+1}/{len(references)}: {ref_xyz}")
        
        # Reset optimizer for new reference (if configured)
        if config.reset_gp_on_new_reference:
            if hasattr(optimizer, 'reset'):  # Both AEPsych and SimpleGP have reset
                optimizer.reset()
        
        # Initial baseline
        if config.reset_baseline_on_new_reference or i == 0:
            initial_baseline = None
        else:
            # Use previous final baseline as starting point
            initial_baseline = matching_results[-1].baseline_final
        
        # Run matching
        result = run_matching_session(
            observer, optimizer, ref_xyz, config, initial_baseline
        )
        matching_results.append(result)
        pygame.time.wait(1500)
        status = "✓" if result.converged else "✗"
        print(f"  {status} Trials: {result.n_trials}, Converged: {result.converged}")
    
    # Collect matched pairs for regression
    # X: True RGB colors (from model space via color transformation)
    # Y: Matched RGB outputs (what the observer chose as matching)
    X = np.array([observer.color_transform.model_space_to_rgb(r.reference_xyz)
                  for r in matching_results])  # [N, 3] RGB_true
    Y = np.array([r.matched_output for r in matching_results])  # [N, 3] RGB_matched

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

    # Get reference colors in model space
    ref_model = np.array([r.reference_xyz for r in results.matching_results])
    Y_matched = np.array([r.matched_output for r in results.matching_results])

    # Convert model space to true RGB, then apply true transformation
    color_transform = ColorTransformations()
    X_rgb = np.array([color_transform.model_space_to_rgb(model_coords) for model_coords in ref_model])
    Y_true = X_rgb @ results.m_true.T
    
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
        use_full_matrix=True
    )
    
    # Run simulation
    results = run_simulation(config)
    
    # Print summary
    results.print_summary()
    
    # Plot results
    plot_results(results, save_path=args.save)


if __name__ == "__main__":
    main()