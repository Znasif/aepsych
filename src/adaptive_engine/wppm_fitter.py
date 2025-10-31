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

class WishartProcess:
    """Wishart process implementation for smooth covariance field."""

    def __init__(self, n_basis: int = 10, lengthscale: float = 1.0):
        """
        Initialize the Wishart process.

        Args:
            n_basis: Number of basis functions
            lengthscale: Lengthscale parameter for RBF kernel
        """
        self.n_basis = n_basis
        self.lengthscale = lengthscale

        # Create basis function centers (fixed for reproducibility)
        np.random.seed(42)
        self.centers = np.random.uniform(-1, 1, (n_basis, 2))

    def rbf_kernel(self, x1: jnp.ndarray, x2: jnp.ndarray) -> float:
        """Radial basis function kernel."""
        diff = x1 - x2
        return jnp.exp(-jnp.sum(diff**2) / (2 * self.lengthscale**2))

    def compute_covariance_matrix(self, coords: jnp.ndarray,
                                basis_weights: jnp.ndarray) -> jnp.ndarray:
        """
        Compute covariance matrix at given coordinates using basis function expansion.

        Args:
            coords: Coordinate [x, y] in model space
            basis_weights: Weights for basis functions (n_basis × 3 parameters:
                          scale + 2 shape parameters for each basis)

        Returns:
            2x2 covariance matrix
        """
        # Each basis function contributes a scaled Wishart component
        cov_sum = jnp.zeros((2, 2))

        for i in range(self.n_basis):
            # RBF weight for this basis function at this location
            weight = self.rbf_kernel(coords, self.centers[i])

            # Extract parameters for this basis (3 parameters per basis)
            start_idx = i * 3
            scale_param = basis_weights[start_idx]
            shape_param1 = basis_weights[start_idx + 1]
            shape_param2 = basis_weights[start_idx + 2]

            # Create a simple covariance matrix from these parameters
            # This is a simplified Wishart-like construction
            cov_component = jnp.array([
                [jnp.exp(scale_param + shape_param1), shape_param2],
                [shape_param2, jnp.exp(scale_param - shape_param1)]
            ])

            cov_sum += weight * cov_component

        # Ensure positive definiteness by adding a small regularization
        cov_sum += 0.01 * jnp.eye(2)

        return cov_sum

class WPPMModel:
    """Wishart Process Psychophysical Model implementation."""

    def __init__(self, n_basis: int = 10, observer_samples: int = 100):
        """
        Initialize the WPPM model.

        Args:
            n_basis: Number of basis functions for Wishart process
            observer_samples: Number of Monte Carlo samples for observer model
        """
        self.wishart_process = WishartProcess(n_basis=n_basis)
        self.observer = WPPMObserver(n_samples=observer_samples)

        # Model parameters
        self.n_basis = n_basis
        self.total_params = n_basis * 3  # 3 parameters per basis function

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

        # Sample basis function weights from prior
        # Each weight has a normal prior
        basis_weights = numpyro.sample(
            "basis_weights",
            dist.Normal(0, 1).expand([self.total_params])
        )

        # For each trial, compute the covariance matrix and predict performance
        for i in range(n_trials):
            ref_coords = reference_coords[i]
            comp_coords = comparison_coords[i]

            # Get covariance matrix for this location (average of ref and comp locations)
            avg_coords = (ref_coords + comp_coords) / 2
            cov_matrix = self.wishart_process.compute_covariance_matrix(
                avg_coords, basis_weights
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
                  n_samples: int = 1000, n_warmup: int = 500) -> Dict[str, Any]:
        """
        Fit the WPPM to trial data using MCMC.

        Args:
            trial_data: DataFrame with trial data
            n_samples: Number of MCMC samples
            n_warmup: Number of warmup samples

        Returns:
            Dictionary with fitted parameters and diagnostics
        """
        print("Preparing data for WPPM fitting...")

        # Extract relevant data from DataFrame
        # Convert RGB to model space coordinates
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
            responses.append(trial['response'])  # Assuming 1 for correct, 0 for incorrect

        # Convert to JAX arrays
        trial_dict = {
            'reference_coords': jnp.array(ref_coords),
            'comparison_coords': jnp.array(comp_coords),
            'responses': jnp.array(responses, dtype=jnp.int32)
        }

        print(f"Fitting WPPM to {len(responses)} trials...")

        # Set up MCMC
        kernel = NUTS(self.numpyro_model)
        mcmc = MCMC(kernel, num_warmup=n_warmup, num_samples=n_samples,
                   progress_bar=True)

        # Run MCMC
        rng_key = random.PRNGKey(0)
        mcmc.run(rng_key, trial_dict)

        # Extract results
        samples = mcmc.get_samples()
        diagnostics = {
            'summary': mcmc.print_summary(),
            'r_hat': None,  # Could compute Gelman-Rubin statistic
            'n_eff': None   # Could compute effective sample size
        }

        return {
            'samples': samples,
            'diagnostics': diagnostics,
            'mcmc': mcmc
        }

    def predict_psychometric_field(self, fitted_samples: Dict[str, jnp.ndarray],
                                 grid_coords: jnp.ndarray) -> jnp.ndarray:
        """
        Predict discrimination performance across the stimulus space.

        Args:
            fitted_samples: MCMC samples from fitted model
            grid_coords: Grid of coordinates to evaluate [n_points, 2]

        Returns:
            Predicted percent correct for each grid point [n_samples, n_points]
        """
        n_samples = fitted_samples['basis_weights'].shape[0]
        n_points = grid_coords.shape[0]

        predictions = []

        for sample_idx in range(min(100, n_samples)):  # Use subset for speed
            basis_weights = fitted_samples['basis_weights'][sample_idx]

            sample_predictions = []
            for coord_idx in range(n_points):
                coords = grid_coords[coord_idx]

                # For prediction, we need reference vs comparison
                # For now, use a small perturbation along x-axis
                ref_coords = coords
                comp_coords = coords + jnp.array([0.1, 0.0])

                cov_matrix = self.wishart_process.compute_covariance_matrix(
                    coords, basis_weights
                )

                p_correct = self.observer.predict_percent_correct(
                    ref_coords, comp_coords, cov_matrix
                )

                sample_predictions.append(p_correct)

            predictions.append(sample_predictions)

        return jnp.array(predictions)

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

def fit_wppm_to_data(trial_data_path: str, output_path: str = None) -> Dict[str, Any]:
    """
    Fit WPPM to trial data and save results.

    Args:
        trial_data_path: Path to trial data CSV
        output_path: Path to save fitted model

    Returns:
        Dictionary with fitted model results
    """
    print("Loading trial data...")
    trial_data = load_trial_data(trial_data_path)

    print("Initializing WPPM...")
    wppm = WPPMModel(n_basis=10, observer_samples=500)

    print("Fitting WPPM model...")
    fit_results = wppm.fit_model(trial_data, n_samples=500, n_warmup=200)

    # Save results
    if output_path is None:
        output_path = Path(trial_data_path).parent / "wppm_fit_results.json"

    # Convert JAX arrays to lists for JSON serialization
    serializable_results = {
        'samples': {k: v.tolist() for k, v in fit_results['samples'].items()},
        'diagnostics': fit_results['diagnostics'],
        'model_config': {
            'n_basis': wppm.n_basis,
            'observer_samples': wppm.observer.n_samples
        }
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
    parser.add_argument("--n_basis", type=int, default=10, help="Number of basis functions")
    parser.add_argument("--samples", type=int, default=500, help="Number of MCMC samples")

    args = parser.parse_args()

    fit_wppm_to_data(args.data, args.output)
