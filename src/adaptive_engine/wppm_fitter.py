#!/usr/bin/env python3
"""
Wishart Process Psychophysical Model (WPPM) - Pure JAX Implementation

This module implements the semi-parametric Bayesian model for fitting
color discrimination data, based on Hong et al. (2025) bioRxiv.

No equinox dependency - uses pure JAX + optax for maximum compatibility.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
from jax.scipy.linalg import cholesky
from typing import NamedTuple, Dict, Any
from functools import partial
import optax


# =============================================================================
# Type Definitions
# =============================================================================

class TrialData(NamedTuple):
    """Trial data structure for WPPM fitting."""
    reference_coords: jnp.ndarray  # [n_trials, 2]
    comparison_coords: jnp.ndarray  # [n_trials, 2]
    responses: jnp.ndarray  # [n_trials] binary (1=correct, 0=incorrect)


class WPPMConfig(NamedTuple):
    """Configuration for WPPM model."""
    n_basis: int = 5  # Number of Chebyshev basis functions per dimension
    gamma: float = 3e-4  # Prior amplitude parameter (Eq. 16)
    epsilon: float = 0.5  # Prior decay rate (Eq. 16)
    n_mc_samples: int = 2000  # Monte Carlo samples for observer model
    coord_range: tuple = (-1.0, 1.0)  # Coordinate space bounds
    bandwidth: float = 1.0  # Bandwidth h for differentiable MC (Appendix 11)


class WPPMParams(NamedTuple):
    """Model parameters (just the weight matrix W)."""
    weights: jnp.ndarray  # Shape: [n_basis, n_basis, 2, 3]


# =============================================================================
# Save/Load Functionality
# =============================================================================

def save_wppm(
    filepath: str,
    params: WPPMParams,
    config: WPPMConfig,
    losses: jnp.ndarray = None,
    metadata: Dict[str, Any] = None
) -> None:
    """
    Save fitted WPPM parameters to disk.
    
    Args:
        filepath: Path to save file (.npz or .json)
        params: Fitted WPPMParams
        config: WPPMConfig used for fitting
        losses: Optional loss history from training
        metadata: Optional additional metadata (e.g., trial count, date)
    """
    import numpy as np
    
    if filepath.endswith('.npz'):
        # NumPy compressed format (recommended for large weights)
        save_dict = {
            'weights': np.array(params.weights),
            'n_basis': config.n_basis,
            'gamma': config.gamma,
            'epsilon': config.epsilon,
            'n_mc_samples': config.n_mc_samples,
            'coord_range_min': config.coord_range[0],
            'coord_range_max': config.coord_range[1],
            'bandwidth': config.bandwidth,
        }
        if losses is not None:
            save_dict['losses'] = np.array(losses)
        if metadata is not None:
            for k, v in metadata.items():
                save_dict[f'meta_{k}'] = v
        np.savez_compressed(filepath, **save_dict)
        
    elif filepath.endswith('.json'):
        # JSON format (human-readable, portable)
        import json
        save_dict = {
            'params': {
                'weights': np.array(params.weights).tolist(),
                'shape': list(params.weights.shape),
            },
            'config': {
                'n_basis': config.n_basis,
                'gamma': config.gamma,
                'epsilon': config.epsilon,
                'n_mc_samples': config.n_mc_samples,
                'coord_range': list(config.coord_range),
                'bandwidth': config.bandwidth,
            },
        }
        if losses is not None:
            save_dict['losses'] = np.array(losses).tolist()
        if metadata is not None:
            save_dict['metadata'] = metadata
        with open(filepath, 'w') as f:
            json.dump(save_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .npz or .json")
    
    print(f"Saved WPPM model to {filepath}")
    print(f"  - Weights shape: {params.weights.shape} ({params.weights.size} parameters)")


def load_wppm(filepath: str) -> tuple:
    """
    Load fitted WPPM parameters from disk.
    
    Args:
        filepath: Path to saved file (.npz or .json)
        
    Returns:
        Tuple of (WPPMParams, WPPMConfig, losses, metadata)
        losses and metadata may be None if not saved
    """
    import numpy as np
    
    if filepath.endswith('.npz'):
        data = np.load(filepath, allow_pickle=True)
        
        weights = jnp.array(data['weights'])
        params = WPPMParams(weights=weights)
        
        config = WPPMConfig(
            n_basis=int(data['n_basis']),
            gamma=float(data['gamma']),
            epsilon=float(data['epsilon']),
            n_mc_samples=int(data['n_mc_samples']),
            coord_range=(float(data['coord_range_min']), float(data['coord_range_max'])),
            bandwidth=float(data['bandwidth']),
        )
        
        losses = jnp.array(data['losses']) if 'losses' in data else None
        
        # Extract metadata
        metadata = {}
        for key in data.files:
            if key.startswith('meta_'):
                metadata[key[5:]] = data[key].item() if data[key].ndim == 0 else data[key]
        metadata = metadata if metadata else None
        
    elif filepath.endswith('.json'):
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        weights = jnp.array(data['params']['weights'])
        params = WPPMParams(weights=weights)
        
        cfg = data['config']
        config = WPPMConfig(
            n_basis=cfg['n_basis'],
            gamma=cfg['gamma'],
            epsilon=cfg['epsilon'],
            n_mc_samples=cfg['n_mc_samples'],
            coord_range=tuple(cfg['coord_range']),
            bandwidth=cfg['bandwidth'],
        )
        
        losses = jnp.array(data['losses']) if 'losses' in data else None
        metadata = data.get('metadata', None)
        
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .npz or .json")
    
    print(f"Loaded WPPM model from {filepath}")
    print(f"  - Weights shape: {params.weights.shape} ({params.weights.size} parameters)")
    
    return params, config, losses, metadata


# =============================================================================
# Chebyshev Polynomial Basis (Equations 9-12)
# =============================================================================

def chebyshev_basis_1d(x: float, n_basis: int) -> jnp.ndarray:
    """
    Compute 1D Chebyshev polynomials T_0(x) to T_{n-1}(x) using recurrence.
    
    Equations 9-11 from paper:
        T_0(x) = 1
        T_1(x) = x
        T_{i+1}(x) = 2x·T_i(x) - T_{i-1}(x)
    
    Args:
        x: Input value (should be in [-1, 1] for stability)
        n_basis: Number of basis functions
        
    Returns:
        Array of shape [n_basis] with T_0(x), T_1(x), ..., T_{n-1}(x)
    """
    def recurrence(carry, _):
        t_prev2, t_prev1 = carry
        t_next = 2 * x * t_prev1 - t_prev2
        return (t_prev1, t_next), t_next
    
    # Initial values: T_0 = 1, T_1 = x
    init = (jnp.array(1.0), jnp.array(x))
    
    # Use scan for higher-order terms
    _, higher_order = jax.lax.scan(recurrence, init, None, length=max(0, n_basis - 2))
    
    # Concatenate [T_0, T_1, T_2, ..., T_{n-1}]
    base = jnp.array([1.0, x])
    return jnp.concatenate([base, higher_order])[:n_basis]


def chebyshev_basis_2d(coords: jnp.ndarray, n_basis: int) -> jnp.ndarray:
    """
    Compute 2D tensor product Chebyshev basis (Equation 12).
    
    φ_{i,j}(x) = T_i(x_{dim1}) · T_j(x_{dim2})
    
    Args:
        coords: Coordinates [x_{dim1}, x_{dim2}] in [-1, 1]²
        n_basis: Number of basis functions per dimension
        
    Returns:
        Array of shape [n_basis, n_basis] with φ_{i,j}(coords)
    """
    t1 = chebyshev_basis_1d(coords[0], n_basis)  # [n_basis]
    t2 = chebyshev_basis_1d(coords[1], n_basis)  # [n_basis]
    return jnp.outer(t1, t2)  # [n_basis, n_basis]


# =============================================================================
# Covariance Field (Equations 13-14)
# =============================================================================

def compute_covariance(coords: jnp.ndarray, weights: jnp.ndarray, n_basis: int) -> jnp.ndarray:
    """
    Compute covariance matrix Σ(x) at given coordinates.
    
    Equation 13: U_{k,l}(x) = Σᵢ Σⱼ W_{i,j,k,l} · φ_{i,j}(x)
    Equation 14: Σ(x) = U_{k,l}(x) · U_{k,l}(x)ᵀ
    
    Args:
        coords: Spatial coordinates [x_{dim1}, x_{dim2}] in [-1, 1]²
        weights: Weight tensor W ∈ ℝ^{n_basis × n_basis × 2 × 3}
        n_basis: Number of basis functions per dimension
        
    Returns:
        2×2 positive semi-definite covariance matrix
    """
    # Compute 2D Chebyshev basis values: [n_basis, n_basis]
    phi = chebyshev_basis_2d(coords, n_basis)
    
    # Compute overcomplete representation U ∈ ℝ^{2×3} (Equation 13)
    # U_{k,l} = Σᵢⱼ W_{i,j,k,l} · φ_{i,j}
    U = jnp.einsum('ij,ijkl->kl', phi, weights)
    
    # Σ = U @ U.T ensures positive semi-definiteness (Equation 14)
    cov = U @ U.T
    
    # Add small regularization for numerical stability
    cov = cov + 1e-8 * jnp.eye(2)
    
    return cov


def compute_covariance_batch(coords_batch: jnp.ndarray, weights: jnp.ndarray, n_basis: int) -> jnp.ndarray:
    """Vectorized covariance computation for batch of coordinates."""
    return vmap(lambda c: compute_covariance(c, weights, n_basis))(coords_batch)


# =============================================================================
# Prior Distribution (Equations 15-16)
# =============================================================================

def compute_prior_variance(n_basis: int, gamma: float, epsilon: float) -> jnp.ndarray:
    """
    Compute prior variance η_{i+j} for each weight (Equation 16).
    
    η_{i+j} = γ · ε^{i+j}
    
    Args:
        n_basis: Number of basis functions per dimension
        gamma: Overall amplitude (γ = 3×10⁻⁴ in paper)
        epsilon: Decay rate (ε = 0.5 in paper)
        
    Returns:
        Variance array of shape [n_basis, n_basis]
    """
    i_idx = jnp.arange(n_basis)[:, None]
    j_idx = jnp.arange(n_basis)[None, :]
    order = i_idx + j_idx
    variance = gamma * (epsilon ** order)
    return variance


def log_prior(weights: jnp.ndarray, config: WPPMConfig) -> float:
    """
    Compute log prior probability of weights (Equation 15).
    
    W_{i,j,k,l} ~ N(0, η_{i+j})
    
    Args:
        weights: Weight tensor [n_basis, n_basis, 2, 3]
        config: Model configuration
        
    Returns:
        Log prior probability (scalar)
    """
    variance = compute_prior_variance(config.n_basis, config.gamma, config.epsilon)
    
    # Broadcast variance to all output dimensions [n_basis, n_basis, 1, 1]
    variance_4d = variance[:, :, None, None]
    
    # Gaussian log prior: -0.5 * w²/σ² - 0.5 * log(2πσ²)
    log_norm = -0.5 * jnp.log(2 * jnp.pi * variance_4d)
    log_exp = -0.5 * weights ** 2 / variance_4d
    
    return jnp.sum(log_norm + log_exp)


# =============================================================================
# Observer Model - 3AFC Task (Equations 1-8)
# =============================================================================

def simulate_3afc_trial(
    key: jr.PRNGKey,
    ref_coords: jnp.ndarray,
    comp_coords: jnp.ndarray,
    weights: jnp.ndarray,
    n_basis: int,
    n_samples: int,
    bandwidth: float
) -> float:
    """
    Simulate 3AFC trial using differentiable Monte Carlo (Appendix 11).
    
    Observer model (Equations 1-8):
    - Equations 1-3: Internal representations z ~ N(stimulus, Σ(stimulus))
    - Equation 4: Decision rule based on Mahalanobis distances
    - Equations 5-7: Mahalanobis distance definitions
    - Equation 8: Weighted covariance S = (2/3)Σ(x₀) + (1/3)Σ(x₀+Δ)
    
    Uses smooth sigmoid approximation (Appendix 11, Eq. S22) for differentiability.
    
    Args:
        key: JAX random key
        ref_coords: Reference stimulus coordinates [2]
        comp_coords: Comparison stimulus coordinates [2]
        weights: Weight tensor W
        n_basis: Number of basis functions
        n_samples: Number of Monte Carlo samples
        bandwidth: Smoothing bandwidth h for differentiable estimator
        
    Returns:
        Estimated probability of correct response (differentiable)
    """
    # Get covariance matrices at stimulus locations
    cov_ref = compute_covariance(ref_coords, weights, n_basis)
    cov_comp = compute_covariance(comp_coords, weights, n_basis)
    
    # Weighted average covariance for Mahalanobis distance (Equation 8)
    # S = (2/3)·Σ(x₀) + (1/3)·Σ(x₀ + Δ)
    S = (2.0 / 3.0) * cov_ref + (1.0 / 3.0) * cov_comp
    S_inv = jnp.linalg.inv(S)
    
    # Cholesky decompositions for sampling
    L_ref = cholesky(cov_ref, lower=True)
    L_comp = cholesky(cov_comp, lower=True)
    
    # Generate standard normal samples
    eps = jr.normal(key, (n_samples, 3, 2))  # [n_samples, 3 stimuli, 2 dims]
    
    # Internal representations (Equations 1-3)
    # z₀ ~ N(x₀, Σ(x₀)), z'₀ ~ N(x₀, Σ(x₀)), z₁ ~ N(x₀+Δ, Σ(x₀+Δ))
    z0 = ref_coords + eps[:, 0, :] @ L_ref.T
    z0_prime = ref_coords + eps[:, 1, :] @ L_ref.T
    z1 = comp_coords + eps[:, 2, :] @ L_comp.T
    
    # Squared Mahalanobis distances (Equations 5-7)
    # d²_M(a, b) = (a-b)ᵀ S⁻¹ (a-b)
    def mahalanobis_sq(a, b):
        diff = a - b
        return jnp.sum(diff * (diff @ S_inv.T), axis=-1)
    
    d_01 = mahalanobis_sq(z0, z0_prime)  # d²_M(z₀, z'₀)
    d_0c = mahalanobis_sq(z0, z1)        # d²_M(z₀, z₁)
    d_1c = mahalanobis_sq(z0_prime, z1)  # d²_M(z'₀, z₁)
    
    # Decision variable (Equation 4)
    # Correct if: d²_M(z₀, z'₀) - min(d²_M(z₀, z₁), d²_M(z'₀, z₁)) < 0
    v = d_01 - jnp.minimum(d_0c, d_1c)
    
    # Differentiable Monte Carlo estimator (Appendix 11, Eq. S22)
    # I_{v_i}(0) = σ(-v_i / h) smoothly approximates 1[v_i < 0]
    smooth_correct = jax.nn.sigmoid(-v / bandwidth)
    
    return jnp.mean(smooth_correct)


# =============================================================================
# Likelihood and Loss Functions (Equations 17-18)
# =============================================================================

def compute_log_likelihood(
    weights: jnp.ndarray,
    trial_data: TrialData,
    key: jr.PRNGKey,
    config: WPPMConfig
) -> float:
    """
    Compute log likelihood of trial data (Equation 17).
    
    p_r(y₁,...,y_R | W) = Σᵣ [yᵣ·log(pᵣ) + (1-yᵣ)·log(1-pᵣ)]
    
    Args:
        weights: Weight tensor W
        trial_data: Trial data (coords and responses)
        key: JAX random key
        config: Model configuration
        
    Returns:
        Log likelihood (scalar)
    """
    n_trials = trial_data.responses.shape[0]
    keys = jr.split(key, n_trials)
    
    # Vectorized probability computation (Equation 18)
    def compute_p_correct(key, ref, comp):
        return simulate_3afc_trial(
            key, ref, comp, weights, 
            config.n_basis, config.n_mc_samples, config.bandwidth
        )
    
    p_correct = vmap(compute_p_correct)(
        keys,
        trial_data.reference_coords,
        trial_data.comparison_coords
    )
    
    # Clip for numerical stability
    eps = 1e-6
    p_correct = jnp.clip(p_correct, eps, 1 - eps)
    
    # Bernoulli log likelihood
    responses = trial_data.responses.astype(jnp.float32)
    log_lik = responses * jnp.log(p_correct) + (1 - responses) * jnp.log(1 - p_correct)
    
    return jnp.sum(log_lik)


def loss_fn(
    weights: jnp.ndarray,
    trial_data: TrialData,
    key: jr.PRNGKey,
    config: WPPMConfig
) -> float:
    """
    Negative log posterior (loss for MAP estimation).
    
    Combines likelihood (Eq. 17) and prior (Eq. 15-16).
    
    Args:
        weights: Weight tensor W
        trial_data: Trial data
        key: JAX random key
        config: Model configuration
        
    Returns:
        Negative log posterior (to minimize)
    """
    log_lik = compute_log_likelihood(weights, trial_data, key, config)
    log_p = log_prior(weights, config)
    
    return -(log_lik + log_p)


# =============================================================================
# Model Initialization
# =============================================================================

def init_params(key: jr.PRNGKey, config: WPPMConfig, init_scale: float = 0.1) -> WPPMParams:
    """
    Initialize model parameters.
    
    Args:
        key: JAX random key
        config: Model configuration
        init_scale: Scale for weight initialization
        
    Returns:
        Initialized WPPMParams
    """
    shape = (config.n_basis, config.n_basis, 2, 3)
    weights = jr.normal(key, shape) * init_scale
    return WPPMParams(weights=weights)


# =============================================================================
# Training Loop
# =============================================================================

def fit_wppm(
    trial_data: TrialData,
    config: WPPMConfig = WPPMConfig(),
    n_iterations: int = 1000,
    learning_rate: float = 1e-3,
    key: jr.PRNGKey = jr.PRNGKey(42),
    verbose: bool = True,
    print_every: int = 100
) -> tuple[WPPMParams, list]:
    """
    Fit WPPM model to trial data using MAP estimation.
    
    Args:
        trial_data: Trial data with coordinates and responses
        config: Model configuration
        n_iterations: Number of optimization iterations
        learning_rate: Learning rate for Adam
        key: JAX random key
        verbose: Whether to print progress
        print_every: Print frequency
        
    Returns:
        Tuple of (fitted parameters, loss history)
    """
    if verbose:
        print(f"Fitting WPPM to {trial_data.responses.shape[0]} trials...")
        print(f"Config: n_basis={config.n_basis}, n_mc={config.n_mc_samples}, bandwidth={config.bandwidth}")
    
    # Initialize
    key, init_key = jr.split(key)
    params = init_params(init_key, config)
    
    # Setup optimizer
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params.weights)
    
    # Loss and gradient function
    @jax.value_and_grad
    def loss_and_grad(weights, key):
        return loss_fn(weights, trial_data, key, config)
    
    loss_history = []
    
    for i in range(n_iterations):
        key, subkey = jr.split(key)
        
        loss, grads = loss_and_grad(params.weights, subkey)
        updates, opt_state = optimizer.update(grads, opt_state, params.weights)
        new_weights = optax.apply_updates(params.weights, updates)
        params = WPPMParams(weights=new_weights)
        
        loss_history.append(float(loss))
        
        if verbose and (i + 1) % print_every == 0:
            print(f"  Step {i+1}/{n_iterations}, Loss: {loss:.4f}")
    
    if verbose:
        print(f"Optimization complete. Final loss: {loss_history[-1]:.4f}")
    
    return params, loss_history


# =============================================================================
# Evaluation Functions
# =============================================================================

def predict_percent_correct(
    params: WPPMParams,
    ref_coords: jnp.ndarray,
    comp_coords: jnp.ndarray,
    config: WPPMConfig,
    key: jr.PRNGKey = jr.PRNGKey(0)
) -> float:
    """
    Predict percent correct for a stimulus pair.
    
    Args:
        params: Fitted model parameters
        ref_coords: Reference stimulus coordinates
        comp_coords: Comparison stimulus coordinates
        config: Model configuration
        key: JAX random key
        
    Returns:
        Predicted percent correct
    """
    return simulate_3afc_trial(
        key, ref_coords, comp_coords, params.weights,
        config.n_basis, config.n_mc_samples, config.bandwidth
    )


def get_covariance_at(params: WPPMParams, coords: jnp.ndarray, config: WPPMConfig) -> jnp.ndarray:
    """
    Get covariance matrix at a specific location.
    
    Args:
        params: Model parameters
        coords: Spatial coordinates [2]
        config: Model configuration
        
    Returns:
        2×2 covariance matrix
    """
    return compute_covariance(coords, params.weights, config.n_basis)


def predict_threshold(
    params: WPPMParams,
    center_coords: jnp.ndarray,
    direction: jnp.ndarray,
    config: WPPMConfig,
    target_accuracy: float = 0.75,
    key: jr.PRNGKey = jr.PRNGKey(0),
    n_search_points: int = 50,
    max_delta: float = 0.5
) -> float:
    """
    Find discrimination threshold along a direction.
    
    Args:
        params: Fitted model parameters
        center_coords: Center point in stimulus space
        direction: Direction vector for threshold search
        config: Model configuration
        target_accuracy: Target percent correct (default 75%)
        key: JAX random key
        n_search_points: Points to evaluate
        max_delta: Maximum distance to search
        
    Returns:
        Threshold distance for target accuracy
    """
    # Normalize direction
    direction = direction / jnp.linalg.norm(direction)
    
    # Search along direction
    deltas = jnp.linspace(0.01, max_delta, n_search_points)
    keys = jr.split(key, n_search_points)
    
    def eval_delta(key, delta):
        comp = center_coords + delta * direction
        return simulate_3afc_trial(
            key, center_coords, comp, params.weights,
            config.n_basis, config.n_mc_samples, config.bandwidth
        )
    
    accuracies = vmap(eval_delta)(keys, deltas)
    
    # Find threshold via interpolation
    above_target = accuracies >= target_accuracy
    idx = jnp.argmax(above_target)
    
    if idx == 0:
        return deltas[0]
    
    # Linear interpolation
    acc_low = accuracies[idx - 1]
    acc_high = accuracies[idx]
    delta_low = deltas[idx - 1]
    delta_high = deltas[idx]
    
    t = (target_accuracy - acc_low) / (acc_high - acc_low + 1e-8)
    threshold = delta_low + t * (delta_high - delta_low)
    
    return threshold


# =============================================================================
# Synthetic Data Generation
# =============================================================================

def create_synthetic_data(
    n_trials: int = 500,
    key: jr.PRNGKey = jr.PRNGKey(123)
) -> TrialData:
    """
    Create synthetic trial data for testing.
    
    Args:
        n_trials: Number of trials
        key: JAX random key
        
    Returns:
        TrialData with synthetic trials
    """
    key1, key2, key3, key4 = jr.split(key, 4)
    
    # Random reference points in [-0.8, 0.8]²
    ref_coords = jr.uniform(key1, (n_trials, 2), minval=-0.8, maxval=0.8)
    
    # Random comparison offsets
    angles = jr.uniform(key2, (n_trials,)) * 2 * jnp.pi
    deltas = jr.uniform(key3, (n_trials,)) * 0.2 + 0.05
    offsets = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=1) * deltas[:, None]
    comp_coords = ref_coords + offsets
    
    # Synthetic responses (higher accuracy for larger deltas)
    base_accuracy = 0.5 + deltas * 2
    base_accuracy = jnp.clip(base_accuracy, 0.5, 0.95)
    responses = (jr.uniform(key4, (n_trials,)) < base_accuracy).astype(jnp.int32)
    
    return TrialData(
        reference_coords=ref_coords,
        comparison_coords=comp_coords,
        responses=responses
    )


# =============================================================================
# Main Demo
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("WPPM Pure JAX Implementation - Demo")
    print("=" * 60)
    
    # Create synthetic data
    print("\nCreating synthetic trial data...")
    trial_data = create_synthetic_data(n_trials=100)
    print(f"  Generated {trial_data.responses.shape[0]} trials")
    print(f"  Accuracy: {trial_data.responses.mean():.1%}")
    
    # Configure model
    config = WPPMConfig(
        n_basis=4,
        n_mc_samples=300,
        bandwidth=1.0,
        gamma=3e-4,
        epsilon=0.5
    )
    
    # Fit model
    print("\nFitting WPPM...")
    params, losses = fit_wppm(
        trial_data,
        config=config,
        n_iterations=50,
        learning_rate=1e-2,
        verbose=True,
        print_every=10
    )
    
    # Save fitted model
    save_wppm(
        "wppm_fitted.npz",
        params,
        config,
        losses=losses,
        metadata={'n_trials': len(trial_data.responses), 'demo': True}
    )
    
    # Evaluate
    print("\nEvaluating fitted model...")
    test_point = jnp.array([0.0, 0.0])
    cov = get_covariance_at(params, test_point, config)
    print(f"  Covariance at origin:\n{cov}")
    
    # Threshold
    threshold = predict_threshold(
        params,
        center_coords=test_point,
        direction=jnp.array([1.0, 0.0]),
        config=config,
        target_accuracy=0.75
    )
    print(f"  75% threshold along x-axis: {threshold:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo complete!")