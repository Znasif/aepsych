#!/usr/bin/env python3
"""
Simulation for inferring hidden color transformation matrix M.

Setup:
- T (Target): Inaccessible region showing M_true × XYZ (observer sees, we don't know M_true)
- P (Probe): Accessible region showing baseline × XYZ (starts as M_standard, advances when C wins)
- C (Comparison): Accessible region showing M_star × XYZ (AEPsych proposal)

The standard observer uses M_standard to convert XYZ → linear sRGB.
A non-standard observer uses M_true (unknown) which we're trying to learn.

Per-reference flow:
1. Fix reference XYZ (from Munsell chips)
2. AEPsych proposes M_star
3. Observer 2AFC: "Is C or P closer to T?"
4. If C picked → baseline = M_star, get new proposal
5. Repeat until convergence (~50% accuracy)
6. Record (XYZ, baseline_final × XYZ) as matched pair

Final regression: M_est = argmin Σᵢ ‖M × XYZᵢ - Yᵢ‖²
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import pygame
import sys
from pathlib import Path

# colour-science for proper color conversions
import colour

# Suppress colour-science warnings about out-of-gamut colors
colour.utilities.filter_warnings(colour_usage_warnings=False)

sys.path.insert(0, str(Path(__file__).parent / "src"))


# =============================================================================
# Color Science Constants
# =============================================================================

# Standard XYZ to linear sRGB matrix (D65 illuminant)
M_STANDARD_XYZ_TO_SRGB = np.array([
    [ 3.2406255, -1.5372080, -0.4986286],
    [-0.9689307,  1.8757561,  0.0415175],
    [ 0.0557101, -0.2040211,  1.0569959]
])

# Inverse: linear sRGB to XYZ
M_STANDARD_SRGB_TO_XYZ = np.linalg.inv(M_STANDARD_XYZ_TO_SRGB)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimulationConfig:
    """Configuration for the simulation."""
    # True hidden matrix deviation from standard
    # M_true = M_deviation @ M_standard, or directly specified
    # These are multiplicative factors on the standard matrix rows
    m_true_row_scales: Tuple[float, float, float] = (1.08, 0.95, 1.12)
    
    # If True, use full 3x3 deviation matrix; if False, just scale rows
    use_full_matrix: bool = False
    
    # Full 3x3 deviation matrix (used if use_full_matrix=True)
    m_true_deviation: Optional[np.ndarray] = None
    
    # Observer noise (Gaussian std in each channel, in linear RGB space)
    observer_noise_std: float = 0.015
    
    # AEPsych GP bounds for matrix elements (as multipliers on M_standard)
    m_bounds: Tuple[float, float] = (0.8, 1.25)
    
    # Convergence: window of trials to assess accuracy
    convergence_window: int = 8
    convergence_threshold: float = 0.10  # Below this deviation from 0.5 = converged
    
    # Maximum trials per reference before forced termination
    max_trials_per_reference: int = 50
    
    # Minimum trials before checking convergence
    min_trials_per_reference: int = 15
    
    # Whether to reset GP when moving to new reference
    reset_gp_on_new_reference: bool = True
    
    # Whether to reset baseline when moving to new reference  
    reset_baseline_on_new_reference: bool = True
    
    # Munsell sampling parameters
    munsell_values: Tuple[int, ...] = (4, 6, 8)
    munsell_chromas: Tuple[int, ...] = (4, 8)
    munsell_hue_prefixes: Tuple[float, ...] = (5.0,)


# =============================================================================
# Munsell Color Chips (reference colors in CIE XYZ)
# =============================================================================

def get_munsell_references_xyz(config: SimulationConfig) -> Tuple[np.ndarray, List[str]]:
    """
    Generate Munsell reference colors in CIE XYZ.
    
    Args:
        config: Simulation configuration with Munsell sampling parameters
        
    Returns:
        Tuple of (XYZ array [N, 3], list of Munsell specifications)
    """
    references = []
    specs = []
    
    # Munsell hues (10 major hues)
    hue_names = ['R', 'YR', 'Y', 'GY', 'G', 'BG', 'B', 'PB', 'P', 'RP']
    
    for hue_name in hue_names:
        for prefix in config.munsell_hue_prefixes:
            for value in config.munsell_values:
                for chroma in config.munsell_chromas:
                    munsell_spec = f"{prefix}{hue_name} {value}/{chroma}"
                    try:
                        # Convert Munsell to xyY, then to XYZ
                        xyY = colour.munsell_colour_to_xyY(munsell_spec)
                        XYZ = colour.xyY_to_XYZ(xyY)
                        
                        # Check if it's within a reasonable range
                        if np.all(np.isfinite(XYZ)) and np.all(XYZ >= 0):
                            references.append(XYZ)
                            specs.append(munsell_spec)
                    except (ValueError, KeyError):
                        # Some Munsell combinations don't exist (out of gamut)
                        continue
    
    return np.array(references), specs


def filter_displayable_references(
    xyz_refs: np.ndarray,
    specs: List[str],
    m_matrix: np.ndarray
) -> Tuple[np.ndarray, List[str]]:
    """
    Filter references to only those displayable (within sRGB gamut after transformation).
    
    Args:
        xyz_refs: XYZ reference colors [N, 3]
        specs: Munsell specifications
        m_matrix: Transformation matrix (XYZ to linear RGB)
        
    Returns:
        Filtered (XYZ array, specs list)
    """
    filtered_xyz = []
    filtered_specs = []
    
    for xyz, spec in zip(xyz_refs, specs):
        rgb_linear = m_matrix @ xyz
        # Allow small negative values (will be clipped) but reject very out-of-gamut
        if np.all(rgb_linear > -0.1) and np.all(rgb_linear < 1.1):
            filtered_xyz.append(xyz)
            filtered_specs.append(spec)
    
    return np.array(filtered_xyz), filtered_specs


# =============================================================================
# Color Utilities
# =============================================================================

def linear_to_srgb(rgb_linear: np.ndarray) -> np.ndarray:
    """
    Apply sRGB gamma encoding to linear RGB.
    
    Args:
        rgb_linear: Linear RGB values [3] or [N, 3]
        
    Returns:
        sRGB encoded values (same shape)
    """
    rgb_linear = np.asarray(rgb_linear)
    
    # sRGB transfer function
    mask = rgb_linear <= 0.0031308
    rgb_srgb = np.where(
        mask,
        12.92 * rgb_linear,
        1.055 * np.power(np.clip(rgb_linear, 0.0031308, None), 1/2.4) - 0.055
    )
    
    return rgb_srgb


def srgb_to_linear(rgb_srgb: np.ndarray) -> np.ndarray:
    """
    Remove sRGB gamma encoding to get linear RGB.
    
    Args:
        rgb_srgb: sRGB encoded values [3] or [N, 3]
        
    Returns:
        Linear RGB values (same shape)
    """
    rgb_srgb = np.asarray(rgb_srgb)
    
    mask = rgb_srgb <= 0.04045
    rgb_linear = np.where(
        mask,
        rgb_srgb / 12.92,
        np.power((rgb_srgb + 0.055) / 1.055, 2.4)
    )
    
    return rgb_linear


# =============================================================================
# Observer Model
# =============================================================================

class Observer:
    """
    Simulated observer with non-standard color transformation and Gaussian perceptual noise.
    
    The observer sees colors through their personal M_true matrix (XYZ → their linear RGB),
    which differs from the standard M_standard.
    """
    
    def __init__(self, m_true: np.ndarray, noise_std: float = 0.015):
        """
        Args:
            m_true: Observer's true hidden transformation matrix XYZ → linear RGB [3, 3]
            noise_std: Standard deviation of Gaussian perceptual noise (in linear RGB)
        """
        self.m_true = m_true
        self.noise_std = noise_std
        
        # Initialize pygame display
        pygame.init()
        self.screen_width = 1920
        self.screen_height = 1080
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Color Discrimination - 2AFC")
        self.clock = pygame.time.Clock()
        
        # Stimulus parameters (in pixels)
        ppd = 50  # pixels per degree of visual angle
        self.stimulus_size_px = int(2.0 * ppd)
        self.stimulus_spacing_px = int(3.0 * ppd)
        self._calculate_positions()
    
    def _calculate_positions(self):
        """Calculate stimulus positions in triangular arrangement."""
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        radius = self.stimulus_spacing_px
        
        # Equilateral triangle arrangement
        # Top vertex (Target), bottom-left (Probe), bottom-right (Comparison)
        self.stimulus_positions = [
            (int(center_x), int(center_y - radius)),  # Top: Target
            (int(center_x - radius * 0.866), int(center_y + radius * 0.5)),  # Bottom-left: Probe
            (int(center_x + radius * 0.866), int(center_y + radius * 0.5)),  # Bottom-right: Comparison
        ]
        
        # Labels for display
        self.position_labels = ['T (Target)', 'P (Probe)', 'C (Comparison)']
    
    def perceive(self, rgb_linear: np.ndarray) -> np.ndarray:
        """
        Perceive a color with added perceptual noise.
        
        Args:
            rgb_linear: Input linear RGB color [3]
            
        Returns:
            Perceived color with noise [3]
        """
        noise = np.random.randn(3) * self.noise_std
        return rgb_linear + noise
    
    def get_target_appearance(self, xyz: np.ndarray) -> np.ndarray:
        """
        Get the appearance of target T through the observer's hidden matrix.
        
        Args:
            xyz: Reference color in XYZ [3]
            
        Returns:
            T = M_true @ xyz (in observer's linear RGB space)
        """
        return self.m_true @ xyz
    
    def _rgb_linear_to_display(self, rgb_linear: np.ndarray) -> Tuple[int, int, int]:
        """
        Convert linear RGB to display values (0-255 sRGB).
        
        Args:
            rgb_linear: Linear RGB [3]
            
        Returns:
            Tuple of (R, G, B) integers 0-255
        """
        # Clip to valid range
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
        
        # Apply sRGB gamma
        rgb_srgb = linear_to_srgb(rgb_linear)
        
        # Convert to 8-bit
        rgb_int = np.clip(rgb_srgb * 255, 0, 255).astype(int)
        
        return tuple(rgb_int)
    
    def compare_2afc(
        self,
        xyz: np.ndarray,
        baseline: np.ndarray,
        m_star: np.ndarray
    ) -> Tuple[bool, float, float]:
        """
        Perform 2AFC: Is C (m_star @ xyz) closer to T than P (baseline @ xyz)?
        
        All computations in linear RGB space; display is gamma-corrected.
        
        Args:
            xyz: Reference color in XYZ [3]
            baseline: Current baseline matrix XYZ → linear RGB [3, 3]
            m_star: Proposed comparison matrix XYZ → linear RGB [3, 3]
            
        Returns:
            Tuple of (c_chosen, dist_t_c, dist_t_p)
        """
        # Compute appearances in linear RGB (with independent noise samples)
        t_linear = self.get_target_appearance(xyz)  # What observer truly sees
        p_linear = baseline @ xyz  # Probe appearance
        c_linear = m_star @ xyz  # Comparison appearance
        
        # Add perceptual noise
        t_perceived = t_linear  # Target is the "truth" for this observer
        p_perceived = self.perceive(p_linear)
        c_perceived = self.perceive(c_linear)
        
        # Perceptual distances (in linear RGB space)
        dist_t_p = np.linalg.norm(t_perceived - p_perceived)
        dist_t_c = np.linalg.norm(t_perceived - c_perceived)
        
        # Display the stimuli
        self._display_stimuli(t_linear, p_linear, c_linear)
        
        # Observer chooses the closer one
        c_chosen = dist_t_c < dist_t_p
        
        pygame.time.wait(10)
        
        return c_chosen, dist_t_c, dist_t_p
    
    def _display_stimuli(
        self,
        t_linear: np.ndarray,
        p_linear: np.ndarray,
        c_linear: np.ndarray
    ):
        """Display the three stimuli on screen."""
        # Gray background (D65 white point at ~18% reflectance)
        bg_gray = int(0.18 ** (1/2.2) * 255)  # Gamma-corrected mid-gray
        self.screen.fill((bg_gray, bg_gray, bg_gray))
        
        # Convert to display colors
        colors = [
            self._rgb_linear_to_display(t_linear),
            self._rgb_linear_to_display(p_linear),
            self._rgb_linear_to_display(c_linear)
        ]
        
        # Draw stimuli
        for pos, color in zip(self.stimulus_positions, colors):
            pygame.draw.circle(self.screen, color, pos, self.stimulus_size_px)
        
        # Optional: draw labels (comment out for actual experiment)
        # font = pygame.font.Font(None, 24)
        # for pos, label in zip(self.stimulus_positions, self.position_labels):
        #     text = font.render(label, True, (200, 200, 200))
        #     self.screen.blit(text, (pos[0] - 40, pos[1] + self.stimulus_size_px + 10))
        
        pygame.display.flip()
    
    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()


# =============================================================================
# Simple GP-like Acquisition (AEPsych-style)
# =============================================================================

class SimpleGP:
    """
    Simple GP surrogate for M_star proposals.
    
    Uses UCB acquisition for exploration-exploitation.
    Proposes row scaling factors for the transformation matrix.
    """
    
    def __init__(
        self,
        bounds: Tuple[float, float],
        n_dims: int = 3,
        length_scale: float = 0.1,
        noise_var: float = 0.2,
        n_initial_random: int = 5
    ):
        self.bounds = bounds
        self.n_dims = n_dims
        self.length_scale = length_scale
        self.noise_var = noise_var
        self.n_initial_random = n_initial_random
        
        self.X: List[np.ndarray] = []  # Observations: M_star params (row scales)
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
        Propose next M_star parameters (row scaling factors).
        
        Uses random sampling initially, then UCB.
        
        Returns:
            Row scaling factors [s0, s1, s2] where M_star = diag(s) @ M_standard
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
                perturbed = best_x + np.random.randn(self.n_dims) * 0.03
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
    munsell_spec: str
    matched_output: np.ndarray  # baseline_final @ reference_xyz (linear RGB)
    baseline_final: np.ndarray  # Final baseline matrix
    n_trials: int
    converged: bool
    accuracy_history: List[float] = field(default_factory=list)


def build_matrix_from_scales(scales: np.ndarray, base_matrix: np.ndarray) -> np.ndarray:
    """
    Build transformation matrix from row scaling factors.
    
    Args:
        scales: Row scaling factors [3]
        base_matrix: Base matrix to scale [3, 3]
        
    Returns:
        Scaled matrix [3, 3]
    """
    return np.diag(scales) @ base_matrix


def run_matching_session(
    observer: Observer,
    gp: SimpleGP,
    reference_xyz: np.ndarray,
    munsell_spec: str,
    config: SimulationConfig,
    initial_baseline: Optional[np.ndarray] = None
) -> MatchingResult:
    """
    Run matching session for a single reference color.
    
    Args:
        observer: Observer model
        gp: GP for proposals
        reference_xyz: Reference color in XYZ [3]
        munsell_spec: Munsell specification string
        config: Simulation configuration
        initial_baseline: Starting baseline matrix (default: M_standard)
        
    Returns:
        MatchingResult with matched output
    """
    # Initialize baseline to standard observer matrix
    if initial_baseline is None:
        baseline = M_STANDARD_XYZ_TO_SRGB.copy()
    else:
        baseline = initial_baseline.copy()
    
    # Track outcomes for convergence
    outcomes: List[bool] = []
    accuracy_history: List[float] = []
    baseline_advances: int = 0
    
    n_trials = 0
    converged = False
    
    while n_trials < config.max_trials_per_reference:
        # Get proposal from GP (row scaling factors)
        scales = gp.propose()
        
        # Build M_star matrix
        m_star = build_matrix_from_scales(scales, M_STANDARD_XYZ_TO_SRGB)
        
        # Observer comparison
        c_chosen, dist_t_c, dist_t_p = observer.compare_2afc(
            reference_xyz, baseline, m_star
        )
        
        # Record outcome
        outcomes.append(c_chosen)
        gp.add_observation(scales, float(c_chosen))
        n_trials += 1
        
        # Update baseline if C was chosen (it was closer to target)
        if c_chosen:
            baseline = m_star.copy()
            baseline_advances += 1
        
        # Check convergence
        if n_trials >= config.min_trials_per_reference and baseline_advances >= 1:
            recent = outcomes[-config.convergence_window:]
            accuracy = np.mean(recent)
            accuracy_history.append(accuracy)
            
            # Converged when accuracy is near 50% (can't improve further)
            deviation_from_chance = abs(accuracy - 0.5)
            if deviation_from_chance < config.convergence_threshold:
                if baseline_advances >= 2 or n_trials >= config.min_trials_per_reference + 5:
                    converged = True
                    break
    
    # Compute final matched output
    matched_output = baseline @ reference_xyz
    
    return MatchingResult(
        reference_xyz=reference_xyz,
        munsell_spec=munsell_spec,
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
        
        print(f"\nTrue M (observer's XYZ→RGB):\n{self.m_true}")
        print(f"\nEstimated M:\n{self.m_estimated}")
        
        # Compare to standard
        print(f"\nStandard M (XYZ→sRGB):\n{M_STANDARD_XYZ_TO_SRGB}")
        
        # Error metrics
        frobenius_error = np.linalg.norm(self.m_true - self.m_estimated, 'fro')
        relative_error = frobenius_error / np.linalg.norm(self.m_true, 'fro')
        print(f"\nFrobenius error: {frobenius_error:.4f}")
        print(f"Relative error: {relative_error:.2%}")
        
        # Row-wise comparison (scaling factors)
        print("\nRow scaling comparison (estimated/standard vs true/standard):")
        for i in range(3):
            true_scale = np.linalg.norm(self.m_true[i]) / np.linalg.norm(M_STANDARD_XYZ_TO_SRGB[i])
            est_scale = np.linalg.norm(self.m_estimated[i]) / np.linalg.norm(M_STANDARD_XYZ_TO_SRGB[i])
            print(f"  Row {i}: true={true_scale:.3f}, estimated={est_scale:.3f}")
        
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
    if config.use_full_matrix and config.m_true_deviation is not None:
        m_true = config.m_true_deviation @ M_STANDARD_XYZ_TO_SRGB
    else:
        # Scale rows of standard matrix
        m_true = build_matrix_from_scales(
            np.array(config.m_true_row_scales),
            M_STANDARD_XYZ_TO_SRGB
        )
    
    print(f"\nTrue M (observer's hidden matrix):\n{m_true}")
    print(f"\nRow scales vs standard: {config.m_true_row_scales}")
    print(f"Observer noise std: {config.observer_noise_std}")
    
    # Create observer
    observer = Observer(m_true, noise_std=config.observer_noise_std)
    
    # Create GP
    gp = SimpleGP(bounds=config.m_bounds, n_dims=3)
    
    # Get reference colors from Munsell system
    print("\nLoading Munsell reference colors...")
    xyz_refs, specs = get_munsell_references_xyz(config)
    print(f"Found {len(xyz_refs)} valid Munsell colors")
    
    # Filter to displayable colors (within gamut for both standard and true observer)
    xyz_refs, specs = filter_displayable_references(xyz_refs, specs, M_STANDARD_XYZ_TO_SRGB)
    xyz_refs, specs = filter_displayable_references(xyz_refs, specs, m_true)
    print(f"After gamut filtering: {len(xyz_refs)} colors")
    
    if len(xyz_refs) == 0:
        raise ValueError("No displayable reference colors found! Check M_true bounds.")
    
    # Run matching for each reference
    matching_results: List[MatchingResult] = []
    
    try:
        for i, (ref_xyz, spec) in enumerate(zip(xyz_refs, specs)):
            print(f"\nReference {i+1}/{len(xyz_refs)}: {spec}")
            print(f"  XYZ: {ref_xyz}")
            
            # Reset GP for new reference (if configured)
            if config.reset_gp_on_new_reference:
                gp.reset()
            
            # Initial baseline
            if config.reset_baseline_on_new_reference or i == 0:
                initial_baseline = None
            else:
                initial_baseline = matching_results[-1].baseline_final
            
            # Run matching
            result = run_matching_session(
                observer, gp, ref_xyz, spec, config, initial_baseline
            )
            matching_results.append(result)
            
            status = "✓" if result.converged else "✗"
            print(f"  {status} Trials: {result.n_trials}, Converged: {result.converged}")
            
            # Brief pause between references
            pygame.time.wait(500)
            
    finally:
        observer.cleanup()
    
    # Collect matched pairs for regression
    X = np.array([r.reference_xyz for r in matching_results])  # [N, 3] XYZ inputs
    Y = np.array([r.matched_output for r in matching_results])  # [N, 3] RGB outputs
    
    # Least squares regression: Y ≈ X @ M.T
    # Solve for M such that Y.T ≈ M @ X.T
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
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    
    # Panel 1: Matrix heatmap comparison
    ax1 = axes[0, 0]
    
    diff = results.m_estimated - results.m_true
    im = ax1.imshow(diff, cmap='RdBu', vmin=-0.5, vmax=0.5)
    ax1.set_xticks([0, 1, 2])
    ax1.set_yticks([0, 1, 2])
    ax1.set_xticklabels(['X', 'Y', 'Z'])
    ax1.set_yticklabels(['R', 'G', 'B'])
    ax1.set_title('Estimation Error (M_est - M_true)')
    plt.colorbar(im, ax=ax1)
    
    # Add text annotations
    for i in range(3):
        for j in range(3):
            ax1.text(j, i, f'{diff[i,j]:.3f}', ha='center', va='center', fontsize=10)
    
    # Panel 2: Matched vs True outputs
    ax2 = axes[0, 1]
    
    X = np.array([r.reference_xyz for r in results.matching_results])
    Y_matched = np.array([r.matched_output for r in results.matching_results])
    Y_true = X @ results.m_true.T
    
    for ch, (color, label) in enumerate(zip(['red', 'green', 'blue'], ['R', 'G', 'B'])):
        ax2.scatter(Y_true[:, ch], Y_matched[:, ch], c=color, alpha=0.6, label=label, s=30)
    
    lims = [
        min(Y_true.min(), Y_matched.min()) - 0.05,
        max(Y_true.max(), Y_matched.max()) + 0.05
    ]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='Perfect match')
    ax2.set_xlabel('True output (M_true @ XYZ)')
    ax2.set_ylabel('Matched output')
    ax2.set_title('Matched vs True Outputs (linear RGB)')
    ax2.legend()
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect('equal')
    
    # Panel 3: Trials per reference
    ax3 = axes[1, 0]
    trials = [r.n_trials for r in results.matching_results]
    converged = [r.converged for r in results.matching_results]
    colors = ['forestgreen' if c else 'tomato' for c in converged]
    ax3.bar(range(len(trials)), trials, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax3.set_xlabel('Reference index')
    ax3.set_ylabel('Number of trials')
    ax3.set_title('Trials per Reference (green=converged, red=timeout)')
    ax3.axhline(y=np.mean(trials), color='blue', linestyle='--', linewidth=2, label=f'Mean: {np.mean(trials):.1f}')
    ax3.legend()
    
    # Panel 4: Accuracy histories
    ax4 = axes[1, 1]
    for i, r in enumerate(results.matching_results):
        if r.accuracy_history:
            ax4.plot(r.accuracy_history, alpha=0.3, color='steelblue', linewidth=1)
    
    ax4.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance (50%)')
    ax4.axhspan(0.4, 0.6, alpha=0.2, color='green', label='Convergence zone')
    ax4.set_xlabel('Trial (after minimum trials)')
    ax4.set_ylabel('Recent accuracy (C chosen rate)')
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
    parser.add_argument('--noise', type=float, default=0.015, help='Observer noise std (linear RGB)')
    parser.add_argument('--scale-r', type=float, default=1.08, help='True M row 0 scale (R channel)')
    parser.add_argument('--scale-g', type=float, default=0.95, help='True M row 1 scale (G channel)')
    parser.add_argument('--scale-b', type=float, default=1.12, help='True M row 2 scale (B channel)')
    parser.add_argument('--max-trials', type=int, default=50, help='Max trials per reference')
    parser.add_argument('--min-trials', type=int, default=15, help='Min trials before convergence check')
    parser.add_argument('--save', type=str, default=None, help='Path to save figure')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    # Configure simulation
    config = SimulationConfig(
        m_true_row_scales=(args.scale_r, args.scale_g, args.scale_b),
        observer_noise_std=args.noise,
        max_trials_per_reference=args.max_trials,
        min_trials_per_reference=args.min_trials,
        use_full_matrix=False,
        # Munsell sampling - can adjust for more/fewer references
        munsell_values=(4, 5, 6, 7, 8),
        munsell_chromas=(4, 6, 8),
        munsell_hue_prefixes=(2.5, 5.0, 7.5),
    )
    
    # Run simulation
    results = run_simulation(config)
    
    # Print summary
    results.print_summary()
    
    # Plot results
    plot_results(results, save_path=args.save)


if __name__ == "__main__":
    main()