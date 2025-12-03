#!/usr/bin/env python3
"""
AEPsych wrapper for adaptive stimulus selection in 3AFC color discrimination.

Based on Hong et al. (2025) methodology:
- 4D selection: reference (x₀) AND comparison (x₁) coordinates
- Probit-Bernoulli GP with RBF kernel
- EAVC acquisition for 66.7% threshold estimation
- 900 Sobol initialization + adaptive trials
- GP updated every 20 trials
"""

import numpy as np
import torch
from aepsych.server import AEPsychServer
from aepsych.config import Config
from typing import Tuple, List, Optional, Dict, Any
import json
import os
from pathlib import Path


class PsychophysicalAEPsych:
    """
    AEPsych wrapper for 4D adaptive stimulus selection.
    
    Selects both reference stimulus x₀ and comparison stimulus x₁ in model space.
    Uses probit-Bernoulli GP with RBF kernel for threshold estimation at 66.7% correct.
    """

    def __init__(self, config_path: str = None, participant_id: str = "P01"):
        """
        Initialize AEPsych for the psychophysical experiment.

        Args:
            config_path: Path to experiment configuration
            participant_id: Participant identifier
        """
        self.participant_id = participant_id
        self.trial_count = 0

        # Load experiment configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "experiment_params.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # AEPsych configuration from paper
        adaptive_config = self.config.get('adaptive', {})
        self.total_trials = adaptive_config.get('total_trials', 6000)
        self.sobol_trials = adaptive_config.get('sobol_trials', 900)
        self.adaptive_trials = adaptive_config.get('adaptive_trials', 5100)
        self.update_frequency = adaptive_config.get('update_frequency', 20)

        # Model space bounds (same for reference and comparison)
        stimuli_config = self.config.get('stimuli', {})
        self.model_bounds = stimuli_config.get('model_space_bounds', [-0.7, 0.7])

        # Target threshold (66.7% for 3AFC where chance is 33.3%)
        self.target_threshold = adaptive_config.get('target_threshold', 0.667)

        # Initialize AEPsych server
        self._setup_aepsych()

        # Trial history
        self.trial_history = []

    def _setup_aepsych(self):
        """
        Set up AEPsych server with 4D configuration.
        
        Parameters:
        - x0_dim1, x0_dim2: Reference stimulus coordinates
        - x1_dim1, x1_dim2: Comparison stimulus coordinates
        """
        lb = self.model_bounds[0]
        ub = self.model_bounds[1]

        # 4D configuration: reference (2D) + comparison (2D)
        # Note: acqf must be in OptimizeAcqfGenerator section (not in strategy)
        config_str = f"""
[common]
parnames = [x0_dim1, x0_dim2, x1_dim1, x1_dim2]
stimuli_per_trial = 1
outcome_types = [binary]
target = {self.target_threshold}
strategy_names = [init_strat, opt_strat]

[x0_dim1]
par_type = continuous
lower_bound = {lb}
upper_bound = {ub}

[x0_dim2]
par_type = continuous
lower_bound = {lb}
upper_bound = {ub}

[x1_dim1]
par_type = continuous
lower_bound = {lb}
upper_bound = {ub}

[x1_dim2]
par_type = continuous
lower_bound = {lb}
upper_bound = {ub}

[init_strat]
min_asks = {self.sobol_trials}
generator = SobolGenerator

[opt_strat]
min_asks = {self.adaptive_trials}
refit_every = {self.update_frequency}
generator = OptimizeAcqfGenerator
model = GPClassificationModel

[GPClassificationModel]
inducing_size = 100
mean_covar_factory = default_mean_covar_factory

[OptimizeAcqfGenerator]
restarts = 10
samps = 1000
acqf = MCLevelSetEstimation

[MCLevelSetEstimation]
beta = 3.84
objective = ProbitObjective
target = {self.target_threshold}
"""

        # Create config object
        self.aepsych_config = Config(config_str=config_str)

        # Initialize server
        self.server = AEPsychServer()

        # Send setup message
        setup_message = {
            "type": "setup",
            "message": {"config_str": config_str}
        }

        response = self.server.handle_request(setup_message)
        print(f"AEPsych setup response: {response}")

    def get_next_trial(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """
        Get the next stimulus pair for testing.

        Returns:
            Tuple of (reference_coords, comparison_coords, trial_type)
            - reference_coords: [x0_dim1, x0_dim2] in model space
            - comparison_coords: [x1_dim1, x1_dim2] in model space
            - trial_type: "SOBOL", "ADAPTIVE", or "FALLBACK"
        """
        self.trial_count += 1

        message = {
            "type": "ask",
            "message": {}
        }

        try:
            response = self.server.handle_request(message)

            # Parse 4D response
            if isinstance(response, dict) and "config" in response:
                config = response["config"]
                
                # Extract coordinates
                x0_dim1 = float(config.get("x0_dim1", [0])[0])
                x0_dim2 = float(config.get("x0_dim2", [0])[0])
                x1_dim1 = float(config.get("x1_dim1", [0])[0])
                x1_dim2 = float(config.get("x1_dim2", [0])[0])

                ref_coords = np.array([x0_dim1, x0_dim2])
                comp_coords = np.array([x1_dim1, x1_dim2])

                trial_type = "ADAPTIVE" if self.trial_count > self.sobol_trials else "SOBOL"

            else:
                # Fallback to random
                print(f"Warning: Unexpected AEPsych response: {response}")
                ref_coords, comp_coords = self._generate_fallback_trial()
                trial_type = "FALLBACK"

        except Exception as e:
            print(f"Error asking AEPsych for trial: {e}")
            ref_coords, comp_coords = self._generate_fallback_trial()
            trial_type = "FALLBACK"

        return ref_coords, comp_coords, trial_type

    def _generate_fallback_trial(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate fallback trial using Sobol'-like sampling.
        
        From paper: "fallback trials were Sobol'-sampled with the difference 
        vector Δ scaled by one of three factors (2/8, 3/8, or 4/8)"
        """
        lb, ub = self.model_bounds

        # Random reference
        ref_coords = np.random.uniform(lb, ub, size=2)

        # Random direction
        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])

        # Random magnitude from {2/8, 3/8, 4/8} * range
        range_size = ub - lb
        scale_factor = np.random.choice([2/8, 3/8, 4/8])
        magnitude = scale_factor * range_size

        # Comparison = reference + offset
        offset = magnitude * direction
        comp_coords = ref_coords + offset

        # Clip to bounds
        comp_coords = np.clip(comp_coords, lb, ub)

        return ref_coords, comp_coords

    def tell_trial_result(
        self, 
        ref_coords: np.ndarray, 
        comp_coords: np.ndarray, 
        response: int, 
        trial_type: str
    ):
        """
        Tell AEPsych the result of a trial.

        Args:
            ref_coords: Reference stimulus coordinates [x0_dim1, x0_dim2]
            comp_coords: Comparison stimulus coordinates [x1_dim1, x1_dim2]
            response: Participant response (1=correct, 0=incorrect)
            trial_type: Type of trial
        """
        # Tell AEPsych about all trials
        message = {
            "type": "tell",
            "message": {
                "config": {
                    "x0_dim1": [float(ref_coords[0])],
                    "x0_dim2": [float(ref_coords[1])],
                    "x1_dim1": [float(comp_coords[0])],
                    "x1_dim2": [float(comp_coords[1])]
                },
                "outcome": int(response)
            }
        }

        try:
            self.server.handle_request(message)
        except Exception as e:
            print(f"Warning: Failed to tell AEPsych: {e}")

        # Store trial history
        self.trial_history.append({
            'trial': self.trial_count,
            'ref_coords': ref_coords.tolist(),
            'comp_coords': comp_coords.tolist(),
            'delta': (comp_coords - ref_coords).tolist(),
            'response': response,
            'trial_type': trial_type
        })

    def get_trial_data_for_wppm(self) -> Dict[str, np.ndarray]:
        """
        Get trial data formatted for WPPM fitting.
        
        Returns:
            Dictionary with:
            - reference_coords: [n_trials, 2]
            - comparison_coords: [n_trials, 2]
            - responses: [n_trials]
        """
        if not self.trial_history:
            return None

        ref_coords = np.array([t['ref_coords'] for t in self.trial_history])
        comp_coords = np.array([t['comp_coords'] for t in self.trial_history])
        responses = np.array([t['response'] for t in self.trial_history])

        return {
            'reference_coords': ref_coords,
            'comparison_coords': comp_coords,
            'responses': responses
        }

    def save_trial_history(self, filepath: str):
        """Save trial history to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump({
                'participant_id': self.participant_id,
                'n_trials': len(self.trial_history),
                'config': {
                    'sobol_trials': self.sobol_trials,
                    'adaptive_trials': self.adaptive_trials,
                    'target_threshold': self.target_threshold,
                    'model_bounds': self.model_bounds
                },
                'trials': self.trial_history
            }, f, indent=2)

    def load_trial_history(self, filepath: str):
        """Load trial history from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        self.trial_history = data['trials']
        self.trial_count = len(self.trial_history)

    def get_trial_history(self) -> List[Dict[str, Any]]:
        """Get the history of all trials."""
        return self.trial_history.copy()

    def reset(self):
        """Reset the AEPsych experiment."""
        self.trial_count = 0
        self.trial_history = []
        self._setup_aepsych()


# =============================================================================
# Alternative: Parameterization with reference + offset (Δ)
# =============================================================================

class PsychophysicalAEPsychDelta:
    """
    Alternative AEPsych wrapper using reference + offset parameterization.
    
    Parameters: x₀ (2D reference) + Δ (2D offset)
    Comparison computed as: x₁ = x₀ + Δ
    
    This may be more natural for threshold estimation since Δ directly
    represents the discrimination difficulty.
    """

    def __init__(self, config_path: str = None, participant_id: str = "P01"):
        self.participant_id = participant_id
        self.trial_count = 0

        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "experiment_params.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        adaptive_config = self.config.get('adaptive', {})
        self.sobol_trials = adaptive_config.get('sobol_trials', 900)
        self.adaptive_trials = adaptive_config.get('adaptive_trials', 5100)
        self.update_frequency = adaptive_config.get('update_frequency', 20)

        stimuli_config = self.config.get('stimuli', {})
        self.model_bounds = stimuli_config.get('model_space_bounds', [-0.7, 0.7])
        
        # Delta bounds (offset magnitude)
        self.delta_bounds = adaptive_config.get('delta_bounds', [-0.3, 0.3])

        self.target_threshold = adaptive_config.get('target_threshold', 0.667)

        self._setup_aepsych()
        self.trial_history = []

    def _setup_aepsych(self):
        """Set up AEPsych with reference + delta parameterization."""
        ref_lb, ref_ub = self.model_bounds
        delta_lb, delta_ub = self.delta_bounds

        config_str = f"""
[common]
parnames = [x0_dim1, x0_dim2, delta_dim1, delta_dim2]
stimuli_per_trial = 1
outcome_types = [binary]
target = {self.target_threshold}
strategy_names = [init_strat, opt_strat]

[x0_dim1]
par_type = continuous
lower_bound = {ref_lb}
upper_bound = {ref_ub}

[x0_dim2]
par_type = continuous
lower_bound = {ref_lb}
upper_bound = {ref_ub}

[delta_dim1]
par_type = continuous
lower_bound = {delta_lb}
upper_bound = {delta_ub}

[delta_dim2]
par_type = continuous
lower_bound = {delta_lb}
upper_bound = {delta_ub}

[init_strat]
min_asks = {self.sobol_trials}
generator = SobolGenerator

[opt_strat]
min_asks = {self.adaptive_trials}
refit_every = {self.update_frequency}
generator = OptimizeAcqfGenerator
model = GPClassificationModel

[GPClassificationModel]
inducing_size = 100
mean_covar_factory = default_mean_covar_factory

[OptimizeAcqfGenerator]
restarts = 10
samps = 1000
acqf = MCLevelSetEstimation

[MCLevelSetEstimation]
beta = 3.84
objective = ProbitObjective
target = {self.target_threshold}
"""

        self.aepsych_config = Config(config_str=config_str)
        self.server = AEPsychServer()

        setup_message = {
            "type": "setup",
            "message": {"config_str": config_str}
        }
        response = self.server.handle_request(setup_message)
        print(f"AEPsych (delta param) setup: {response}")

    def get_next_trial(self) -> Tuple[np.ndarray, np.ndarray, str]:
        """Get next trial with reference + offset -> comparison."""
        self.trial_count += 1

        message = {"type": "ask", "message": {}}

        try:
            response = self.server.handle_request(message)

            if isinstance(response, dict) and "config" in response:
                config = response["config"]

                x0_dim1 = float(config.get("x0_dim1", [0])[0])
                x0_dim2 = float(config.get("x0_dim2", [0])[0])
                delta_dim1 = float(config.get("delta_dim1", [0])[0])
                delta_dim2 = float(config.get("delta_dim2", [0])[0])

                ref_coords = np.array([x0_dim1, x0_dim2])
                delta = np.array([delta_dim1, delta_dim2])
                comp_coords = ref_coords + delta

                # Clip comparison to bounds
                comp_coords = np.clip(comp_coords, self.model_bounds[0], self.model_bounds[1])

                trial_type = "ADAPTIVE" if self.trial_count > self.sobol_trials else "SOBOL"
            else:
                ref_coords, comp_coords = self._generate_fallback_trial()
                trial_type = "FALLBACK"

        except Exception as e:
            print(f"Error: {e}")
            ref_coords, comp_coords = self._generate_fallback_trial()
            trial_type = "FALLBACK"

        return ref_coords, comp_coords, trial_type

    def _generate_fallback_trial(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate fallback trial."""
        lb, ub = self.model_bounds
        ref_coords = np.random.uniform(lb, ub, size=2)

        angle = np.random.uniform(0, 2 * np.pi)
        direction = np.array([np.cos(angle), np.sin(angle)])
        scale_factor = np.random.choice([2/8, 3/8, 4/8])
        magnitude = scale_factor * (ub - lb)

        comp_coords = ref_coords + magnitude * direction
        comp_coords = np.clip(comp_coords, lb, ub)

        return ref_coords, comp_coords

    def tell_trial_result(
        self,
        ref_coords: np.ndarray,
        comp_coords: np.ndarray,
        response: int,
        trial_type: str
    ):
        """Tell AEPsych the result."""
        delta = comp_coords - ref_coords

        message = {
            "type": "tell",
            "message": {
                "config": {
                    "x0_dim1": [float(ref_coords[0])],
                    "x0_dim2": [float(ref_coords[1])],
                    "delta_dim1": [float(delta[0])],
                    "delta_dim2": [float(delta[1])]
                },
                "outcome": int(response)
            }
        }

        try:
            self.server.handle_request(message)
        except Exception as e:
            print(f"Warning: {e}")

        self.trial_history.append({
            'trial': self.trial_count,
            'ref_coords': ref_coords.tolist(),
            'comp_coords': comp_coords.tolist(),
            'delta': delta.tolist(),
            'response': response,
            'trial_type': trial_type
        })

    def get_trial_data_for_wppm(self) -> Dict[str, np.ndarray]:
        """Get data formatted for WPPM."""
        if not self.trial_history:
            return None

        return {
            'reference_coords': np.array([t['ref_coords'] for t in self.trial_history]),
            'comparison_coords': np.array([t['comp_coords'] for t in self.trial_history]),
            'responses': np.array([t['response'] for t in self.trial_history])
        }

    def get_trial_history(self) -> List[Dict[str, Any]]:
        return self.trial_history.copy()

    def reset(self):
        self.trial_count = 0
        self.trial_history = []
        self._setup_aepsych()


# =============================================================================
# Test
# =============================================================================

def test_aepsych_wrapper():
    """Test the corrected AEPsych wrapper."""
    print("Testing 4D AEPsych wrapper...")
    print("=" * 50)

    # Create mock config
    mock_config = {
        'adaptive': {
            'total_trials': 100,
            'sobol_trials': 20,
            'adaptive_trials': 80,
            'update_frequency': 10,
            'target_threshold': 0.667
        },
        'stimuli': {
            'model_space_bounds': [-0.7, 0.7]
        }
    }

    # Save mock config
    config_path = Path("/tmp/test_config.json")
    with open(config_path, 'w') as f:
        json.dump(mock_config, f)

    try:
        wrapper = PsychophysicalAEPsych(config_path=str(config_path))

        print("\nRunning 5 test trials...")
        for i in range(5):
            ref, comp, trial_type = wrapper.get_next_trial()
            delta = comp - ref
            delta_mag = np.linalg.norm(delta)

            print(f"Trial {i+1} ({trial_type}):")
            print(f"  Reference: [{ref[0]:.3f}, {ref[1]:.3f}]")
            print(f"  Comparison: [{comp[0]:.3f}, {comp[1]:.3f}]")
            print(f"  Delta: [{delta[0]:.3f}, {delta[1]:.3f}] (mag: {delta_mag:.3f})")

            # Simulate response
            response = np.random.randint(0, 2)
            wrapper.tell_trial_result(ref, comp, response, trial_type)

        # Get WPPM-formatted data
        wppm_data = wrapper.get_trial_data_for_wppm()
        print(f"\nWPPM data shapes:")
        print(f"  reference_coords: {wppm_data['reference_coords'].shape}")
        print(f"  comparison_coords: {wppm_data['comparison_coords'].shape}")
        print(f"  responses: {wppm_data['responses'].shape}")

        print("\nTest completed successfully!")

    except Exception as e:
        print(f"Test failed (expected if AEPsych not installed): {e}")
        print("\nTesting fallback generation...")

        # Test fallback
        wrapper = PsychophysicalAEPsych.__new__(PsychophysicalAEPsych)
        wrapper.model_bounds = [-0.7, 0.7]

        for i in range(3):
            ref, comp = wrapper._generate_fallback_trial()
            delta = comp - ref
            print(f"Fallback {i+1}: ref={ref}, delta_mag={np.linalg.norm(delta):.3f}")


if __name__ == "__main__":
    test_aepsych_wrapper()