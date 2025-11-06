#!/usr/bin/env python3
"""
Wishart Process Psychophysical Model (WPPM) implementation.

This module implements the semi-parametric Bayesian model for fitting
color discrimination data from the AEPsych adaptive trials.
"""

import jax
import jax.numpy as jnp
import jax.random as random
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro import handlers
from numpyro.infer import MCMC, NUTS
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from adaptive_engine.utils.transformations import ColorTransformations

class WPPMObserver:
    """Observer model for 3AFC color discrimination task."""

    def __init__(self, n_samples: int = 1000):
        """
        Initialize the observer model.

        Args:
            n_samples: Number of Monte Carlo samples for decision simulation
        """
        self.n_samples = n_samples
        self.transform = ColorTransformations()

    def simulate_trial(self, reference_coords: jnp.ndarray,
                      comparison_coords: jnp.ndarray,
                      cov_matrix: jnp.ndarray,
                      key: random.PRNGKey) -> float:
        """
        Simulate a single 3AFC trial using Monte Carlo sampling.

        Args:
            reference_coords: Reference stimulus coordinates [x, y]
            comparison_coords: Comparison stimulus coordinates [x, y]
            cov_matrix: 2x2 covariance matrix for internal noise
            key: JAX random key

        Returns:
            Probability of correct response (0-1)
        """
        # In 3AFC, we present two identical references and one comparison
        # The observer needs to identify which stimulus is different

        # Generate noisy observations of the three stimuli
        # Stimuli: [ref1, ref2, comp] where ref1 == ref2 != comp

        stimuli_coords = jnp.stack([
            reference_coords,    # Reference 1
            reference_coords,    # Reference 2 (identical)
            comparison_coords    # Comparison (different)
        ])

        # Add internal noise to each stimulus observation
        # Each stimulus observation ~ MVN(stimulus_coords[i], cov_matrix)
        keys = random.split(key, 3)
        noisy_observations = jnp.stack([
            dist.MultivariateNormal(stimuli_coords[0], cov_matrix).sample(keys[0]),
            dist.MultivariateNormal(stimuli_coords[1], cov_matrix).sample(keys[1]),
            dist.MultivariateNormal(stimuli_coords[2], cov_matrix).sample(keys[2])
        ])

        # Observer decision rule: identify the stimulus most distant from others
        # using Mahalanobis distance with the inverse covariance matrix
        inv_cov = jnp.linalg.inv(cov_matrix)

        # Compute pairwise Mahalanobis distances (JAX-compatible)
        # Pairs: (0,1), (0,2), (1,2)
        diffs = jnp.array([
            noisy_observations[0] - noisy_observations[1],  # pair 0-1
            noisy_observations[0] - noisy_observations[2],  # pair 0-2
            noisy_observations[1] - noisy_observations[2],  # pair 1-2
        ])

        mahalanobis_dists = jnp.sqrt(jnp.sum(diffs * jnp.dot(inv_cov, diffs.T).T, axis=1))

        # Find the pair with maximum distance (these are most likely the references)
        max_dist_idx = jnp.argmax(mahalanobis_dists)

        # Map back to stimulus indices
        # pair 0: indices (0,1), pair 1: indices (0,2), pair 2: indices (1,2)
        pair_indices = jnp.array([[0, 1], [0, 2], [1, 2]])
        ref_pair = pair_indices[max_dist_idx]

        # Find the stimulus not in the reference pair (should be the comparison at index 2)
        # Since we know the pairs are [0,1], [0,2], [1,2], we can determine this directly
        comp_idx = jnp.where(max_dist_idx == 0, 2, jnp.where(max_dist_idx == 1, 1, 0))

        # Decision is correct if comp_idx == 2 (the comparison is in position 2)
        correct = (comp_idx == 2)
        return correct

    def predict_percent_correct(self, reference_coords: jnp.ndarray,
                               comparison_coords: jnp.ndarray,
                               cov_matrix: jnp.ndarray) -> float:
        """
        Predict percent correct for a stimulus pair using Monte Carlo simulation.

        Args:
            reference_coords: Reference stimulus coordinates [x, y]
            comparison_coords: Comparison stimulus coordinates [x, y]
            cov_matrix: 2x2 covariance matrix for internal noise

        Returns:
            Predicted percent correct (0-1)
        """
        key = random.PRNGKey(42)  # Fixed seed for reproducibility

        # Run Monte Carlo simulation
        correct_count = 0
        for i in range(self.n_samples):
            trial_key = random.fold_in(key, i)
            correct = self.simulate_trial(reference_coords, comparison_coords,
                                        cov_matrix, trial_key)
            correct_count += correct

        return correct_count / self.n_samples

class ChebyshevCovarianceField:
    """Chebyshev polynomial-based covariance matrix field for WPPM."""

    def __init__(self, n_basis_per_dim: int = 5):
        """
        Initialize the Chebyshev covariance field.

        Args:
            n_basis_per_dim: Number of basis functions per dimension (default 5, giving 25 total)
        """
        self.n_basis_per_dim = n_basis_per_dim  # i, j ∈ {0, 1, ..., 4}
        self.total_basis = n_basis_per_dim * n_basis_per_dim  # 25 total 2D basis functions

        # Prior parameters (from paper)
        self.gamma = 3e-4  # Overall amplitude of variance
        self.epsilon = 0.5  # Rate of exponential decay with polynomial order

    def chebyshev_polynomial(self, x: float, order: int) -> float:
        """
        Compute Chebyshev polynomial of given order at point x.

        Args:
            x: Input value (should be in [-1, 1] for optimal conditioning)
            order: Polynomial order (0, 1, 2, ...)

        Returns:
            T_order(x)
        """
        if order == 0:
            return 1.0
        elif order == 1:
            return x
        else:
            # Recursive definition: T_{n+1}(x) = 2x*T_n(x) - T_{n-1}(x)
            t_prev2 = 1.0  # T_0
            t_prev1 = x     # T_1

            for n in range(2, order + 1):
                t_current = 2 * x * t_prev1 - t_prev2
                t_prev2 = t_prev1
                t_prev1 = t_current

            return t_prev1

    def compute_2d_basis_functions(self, coords: jnp.ndarray) -> jnp.ndarray:
        """
        Compute 2D Chebyshev basis functions at given coordinates.

        Args:
            coords: Coordinate [x, y] in model space (should be scaled to [-1, 1])

        Returns:
            Array of basis function values [n_basis_per_dim, n_basis_per_dim]
        """
        x_dim1, x_dim2 = coords[0], coords[1]

        basis_values = jnp.zeros((self.n_basis_per_dim, self.n_basis_per_dim))

        for i in range(self.n_basis_per_dim):
            t_i = self.chebyshev_polynomial(x_dim1, i)
            for j in range(self.n_basis_per_dim):
                t_j = self.chebyshev_polynomial(x_dim2, j)
                basis_values = basis_values.at[i, j].set(t_i * t_j)

        return basis_values

    def compute_overcomplete_representation(self, coords: jnp.ndarray,
                                          weight_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Compute overcomplete representation U_{k,l}(x) from weighted basis functions.

        Args:
            coords: Coordinate [x, y] in model space
            weight_matrix: Weight matrix W ∈ ℝ[5×5×2×3]

        Returns:
            U_{k,l} ∈ ℝ[2×3] for the covariance matrix components
        """
        # Get 2D basis function values φ_{i,j}(x)
        basis_2d = self.compute_2d_basis_functions(coords)

        # U_{k,l}(x) = Σ_{i=0}^4 Σ_{j=0}^4 W_{i,j,k,l} * φ_{i,j}(x)
        # where k ∈ {0,1} (output component), l ∈ {0,1,2} (matrix element type)
        u_kl = jnp.zeros((2, 3))

        for k in range(2):  # k ∈ {0,1} for output components
            for l in range(3):  # l ∈ {0,1,2} for matrix elements
                weighted_sum = 0.0
                for i in range(self.n_basis_per_dim):
                    for j in range(self.n_basis_per_dim):
                        weighted_sum += weight_matrix[i, j, k, l] * basis_2d[i, j]
                u_kl = u_kl.at[k, l].set(weighted_sum)

        return u_kl

    def compute_covariance_matrix(self, coords: jnp.ndarray,
                                weight_matrix: jnp.ndarray) -> jnp.ndarray:
        """
        Compute covariance matrix Σ(x) at given coordinates.

        The covariance matrix is constructed as:
        Σ(x) = U_{k,l}(x) · U_{k,l}(x)^T

        This ensures the matrix is symmetric and positive semi-definite.

        Args:
            coords: Coordinate [x, y] in model space (should be in [-1, 1] range)
            weight_matrix: Weight matrix W ∈ ℝ[5×5×2×3]

        Returns:
            2×2 covariance matrix
        """
        # Get overcomplete representation U_{k,l}
        u_kl = self.compute_overcomplete_representation(coords, weight_matrix)

        # Extract components for covariance matrix construction
        # U has shape [2, 3], where:
        # U[0, :] corresponds to first row: [σ²_dim1, σ_dim1_dim2, extra_param]
        # U[1, :] corresponds to second row: [σ_dim1_dim2, σ²_dim2, extra_param]

        # For positive semi-definite matrix, we use the first two columns
        # Σ = U[:, :2] @ U[:, :2].T
        u_reduced = u_kl[:, :2]  # Shape: [2, 2]

        # Compute Σ(x) = U · U^T
        cov_matrix = u_reduced @ u_reduced.T

        # Ensure positive definiteness with small regularization
        cov_matrix = cov_matrix + 1e-6 * jnp.eye(2)

        return cov_matrix

    def get_weight_prior_variance(self, i: int, j: int) -> float:
        """
        Get prior variance for weight W_{i,j,k,l} based on polynomial order.

        η_{i+j} = γ · ε^{i+j}

        Args:
            i: First dimension basis function index
            j: Second dimension basis function index

        Returns:
            Prior variance η_{i+j}
        """
        polynomial_order = i + j
        return self.gamma * (self.epsilon ** polynomial_order)

class WPPMModel:
    """Wishart Process Psychophysical Model implementation."""

    def __init__(self, n_basis_per_dim: int = 5, observer_samples: int = 1):
        """
        Initialize the WPPM model.

        Args:
            n_basis_per_dim: Number of basis functions per dimension (default 5, giving 25 total)
            observer_samples: Number of Monte Carlo samples for observer model
        """
        self.covariance_field = ChebyshevCovarianceField(n_basis_per_dim=n_basis_per_dim)
        self.observer = WPPMObserver(n_samples=observer_samples)

        # Model parameters
        self.n_basis_per_dim = n_basis_per_dim
        self.weight_matrix_shape = (n_basis_per_dim, n_basis_per_dim, 2, 3)  # 5×5×2×3
        self.total_params = n_basis_per_dim * n_basis_per_dim * 2 * 3  # 150 parameters total

    def numpyro_model(self, trial_data: Dict[str, jnp.ndarray]):
        """
        NumPyro model definition for WPPM.

        Args:
            trial_data: Dictionary with 'reference_coords', 'comparison_coords', 'responses'
        """
        # Extract trial data
        reference_coords = trial_data['reference_coords']  # [n_trials, 2]
        comparison_coords = trial_data['comparison_coords']  # [n_trials, 2]
        responses = trial_data['responses']  # [n_trials] (0 or 1)

        n_trials = reference_coords.shape[0]

        # Sample weight matrix W ∈ ℝ[5×5×2×3] from prior
        # Prior: W_{i,j,k,l} ∼ Normal(0, η_{i+j}) where η_{i+j} = γ · ε^{i+j}
        weight_matrix = jnp.zeros(self.weight_matrix_shape)

        for i in range(self.n_basis_per_dim):
            for j in range(self.n_basis_per_dim):
                # Get prior variance for this polynomial order
                prior_var = self.covariance_field.get_weight_prior_variance(i, j)
                prior_std = jnp.sqrt(prior_var)

                for k in range(2):  # output components
                    for l in range(3):  # matrix element types
                        param_name = f"W_{i}_{j}_{k}_{l}"
                        weight_value = numpyro.sample(
                            param_name,
                            dist.Normal(0, prior_std)
                        )
                        weight_matrix = weight_matrix.at[i, j, k, l].set(weight_value)

        # For each trial, compute the covariance matrix and predict performance
        for i in range(n_trials):
            ref_coords = reference_coords[i]
            comp_coords = comparison_coords[i]

            # Get covariance matrix for this location (average of ref and comp locations)
            avg_coords = (ref_coords + comp_coords) / 2
            cov_matrix = self.covariance_field.compute_covariance_matrix(
                avg_coords, weight_matrix
            )

            # Predict percent correct using observer model
            p_correct = self.observer.predict_percent_correct(
                ref_coords, comp_coords, cov_matrix
            )

            # Likelihood: Bernoulli response
            numpyro.sample(
                f"response_{i}",
                dist.Bernoulli(probs=p_correct),
                obs=responses[i]
            )

    def fit_model(self, trial_data: pd.DataFrame,
                  n_iterations: int = 1000, learning_rate: float = 1e-3) -> Dict[str, Any]:
        """
        Fit the WPPM to trial data using MAP estimation with SGD.

        Args:
            trial_data: DataFrame with trial data
            n_iterations: Number of SGD iterations
            learning_rate: Learning rate for SGD

        Returns:
            Dictionary with fitted MAP parameters and diagnostics
        """
        return self.fit_model_map(trial_data, n_iterations, learning_rate)

    def predict_psychometric_field(self, fitted_result: Dict[str, Any],
                                 grid_coords: jnp.ndarray) -> jnp.ndarray:
        """
        Predict discrimination performance across the stimulus space.

        Args:
            fitted_result: MAP fitted model result (contains 'weight_matrix')
            grid_coords: Grid of coordinates to evaluate [n_points, 2]

        Returns:
            Predicted percent correct for each grid point [n_points]
        """
        # Get weight matrix from MAP result
        weight_matrix = fitted_result['weight_matrix']
        n_points = grid_coords.shape[0]

        predictions = []

        for coord_idx in range(n_points):
            coords = grid_coords[coord_idx]

            # For prediction, we need reference vs comparison
            # For now, use a small perturbation along x-axis
            ref_coords = coords
            comp_coords = coords + jnp.array([0.1, 0.0])

            cov_matrix = self.covariance_field.compute_covariance_matrix(
                coords, weight_matrix
            )

            p_correct = self.observer.predict_percent_correct(
                ref_coords, comp_coords, cov_matrix
            )

            predictions.append(p_correct)

        return jnp.array(predictions)

    def log_posterior(self, weight_matrix: jnp.ndarray, trial_data: Dict[str, jnp.ndarray]) -> float:
        """
        Compute the log posterior probability for MAP estimation.

        Args:
            weight_matrix: Weight matrix W ∈ ℝ[5×5×2×3]
            trial_data: Dictionary with trial data

        Returns:
            Log posterior probability (to be maximized)
        """
        # Extract trial data
        reference_coords = trial_data['reference_coords']
        comparison_coords = trial_data['comparison_coords']
        responses = trial_data['responses']

        n_trials = reference_coords.shape[0]

        # Compute log likelihood
        log_likelihood = 0.0

        for i in range(n_trials):
            ref_coords = reference_coords[i]
            comp_coords = comparison_coords[i]
            response = responses[i]

            # Get covariance matrix for this trial location
            avg_coords = (ref_coords + comp_coords) / 2
            cov_matrix = self.covariance_field.compute_covariance_matrix(avg_coords, weight_matrix)

            # Predict percent correct using Monte Carlo simulation
            p_correct = self.observer.predict_percent_correct(ref_coords, comp_coords, cov_matrix)

            # Add to log likelihood (Bernoulli: response*log(p) + (1-response)*log(1-p))
            # Add small epsilon to avoid log(0) and clip probabilities
            eps = 1e-6
            p_correct = jnp.clip(p_correct, eps, 1 - eps)
            log_likelihood += response * jnp.log(p_correct) + (1 - response) * jnp.log(1 - p_correct)

        # Compute log prior
        log_prior = 0.0

        for i in range(self.n_basis_per_dim):
            for j in range(self.n_basis_per_dim):
                # Get prior variance for this polynomial order
                prior_var = self.covariance_field.get_weight_prior_variance(i, j)
                prior_std = jnp.sqrt(prior_var)

                for k in range(2):
                    for l in range(3):
                        weight_value = weight_matrix[i, j, k, l]
                        # Normal log prior: -0.5 * (weight/prior_std)^2 - 0.5*log(2*pi*prior_std^2)
                        log_prior += -0.5 * (weight_value / prior_std) ** 2 - 0.5 * jnp.log(2 * jnp.pi * prior_var)

        # Add L2 regularization for numerical stability
        l2_reg = 1e-4 * jnp.sum(weight_matrix ** 2)

        return log_likelihood + log_prior - l2_reg

    def fit_model_map(self, trial_data: pd.DataFrame,
                     n_iterations: int = 1000, learning_rate: float = 1e-4) -> Dict[str, Any]:
        """
        Fit the WPPM to trial data using MAP estimation with SGD.

        Args:
            trial_data: DataFrame with trial data
            n_iterations: Number of SGD iterations
            learning_rate: Learning rate for SGD

        Returns:
            Dictionary with fitted MAP parameters and diagnostics
        """
        print("Preparing data for WPPM MAP fitting...")

        # Extract relevant data from DataFrame
        ref_coords = []
        comp_coords = []
        responses = []

        for _, trial in trial_data.iterrows():
            # Convert RGB to model space
            ref_rgb = [trial['ref_rgb_r'], trial['ref_rgb_g'], trial['ref_rgb_b']]
            comp_rgb = [trial['comp_rgb_r'], trial['comp_rgb_g'], trial['comp_rgb_b']]

            ref_model = self.observer.transform.rgb_to_model_space(ref_rgb)
            comp_model = self.observer.transform.rgb_to_model_space(comp_rgb)

            ref_coords.append(ref_model)
            comp_coords.append(comp_model)
            responses.append(trial['response'])

        # Convert to JAX arrays
        trial_dict = {
            'reference_coords': jnp.array(ref_coords),
            'comparison_coords': jnp.array(comp_coords),
            'responses': jnp.array(responses, dtype=jnp.int32)
        }

        print(f"Fitting WPPM MAP to {len(responses)} trials...")

        # Initialize weight matrix randomly
        rng_key = random.PRNGKey(42)
        weight_matrix = random.normal(rng_key, shape=self.weight_matrix_shape) * 0.1

        # Define the loss function (negative log posterior for minimization)
        def loss_fn(weight_matrix):
            return -self.log_posterior(weight_matrix, trial_dict)

        # Set up gradient descent
        from jax import grad

        grad_fn = grad(loss_fn)

        # Optimization loop
        for iteration in range(n_iterations):
            if iteration % 2 == 0:
                current_loss = loss_fn(weight_matrix)
                current_log_post = -current_loss
                print(f"Iteration {iteration}: Log Posterior = {current_log_post:.4f}")

            # Compute gradient
            grads = grad_fn(weight_matrix)

            # Check for NaN gradients
            if jnp.any(jnp.isnan(grads)):
                print(f"Warning: NaN gradients detected at iteration {iteration}")
                break

            # Clip gradients to prevent explosion
            grads = jnp.clip(grads, -10.0, 10.0)

            # Update weights
            weight_matrix = weight_matrix - learning_rate * grads

            # Check for NaN weights
            if jnp.any(jnp.isnan(weight_matrix)):
                print(f"Warning: NaN weights detected at iteration {iteration}")
                break

        print("MAP optimization completed.")

        # Compute final diagnostics
        final_log_posterior = self.log_posterior(weight_matrix, trial_dict)

        return {
            'weight_matrix': weight_matrix,
            'log_posterior': final_log_posterior,
            'n_iterations': n_iterations,
            'learning_rate': learning_rate,
            'model_config': {
                'n_basis_per_dim': self.n_basis_per_dim,
                'observer_samples': self.observer.n_samples
            }
        }

def load_trial_data(data_path: str) -> pd.DataFrame:
    """
    Load trial data for WPPM fitting.

    Args:
        data_path: Path to trial data CSV

    Returns:
        DataFrame with trial data
    """
    df = pd.read_csv(data_path)

    # Add response column (for now, assume all are correct for testing)
    # In real usage, this would come from the actual participant responses
    if 'response' not in df.columns:
        df['response'] = np.random.choice([0, 1], size=len(df), p=[0.2, 0.8])

    return df

def fit_wppm_to_data(trial_data_path: str, output_path: str = None, n_basis_per_dim: int = 5,
                    n_iterations: int = 1000, learning_rate: float = 1e-4) -> Dict[str, Any]:
    """
    Fit WPPM to trial data and save results.

    Args:
        trial_data_path: Path to trial data CSV
        output_path: Path to save fitted model
        n_basis_per_dim: Number of basis functions per dimension (default 5)
        n_iterations: Number of SGD iterations (default 1000)
        learning_rate: Learning rate for SGD (default 1e-4)

    Returns:
        Dictionary with fitted model results
    """
    print("Loading trial data...")
    trial_data = load_trial_data(trial_data_path)

    print("Initializing WPPM...")
    wppm = WPPMModel(n_basis_per_dim=n_basis_per_dim, observer_samples=500)

    print("Fitting WPPM model...")
    fit_results = wppm.fit_model(trial_data, n_iterations=n_iterations, learning_rate=learning_rate)

    # Save results
    if output_path is None:
        output_path = Path(trial_data_path).parent / "wppm_fit_results.json"

    # Convert JAX arrays to lists for JSON serialization
    serializable_results = {
        'weight_matrix': fit_results['weight_matrix'].tolist(),
        'log_posterior': float(fit_results['log_posterior']),
        'n_iterations': fit_results['n_iterations'],
        'learning_rate': fit_results['learning_rate'],
        'model_config': fit_results['model_config']
    }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"WPPM fitting completed. Results saved to {output_path}")

    return fit_results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fit WPPM to trial data")
    parser.add_argument("--data", required=True, help="Path to trial data CSV")
    parser.add_argument("--output", help="Path to save fitted model")
    parser.add_argument("--n_basis_per_dim", type=int, default=5, help="Number of basis functions per dimension")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of SGD iterations")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for SGD")

    args = parser.parse_args()

    fit_wppm_to_data(args.data, args.output, args.n_basis_per_dim, args.iterations, args.learning_rate)
