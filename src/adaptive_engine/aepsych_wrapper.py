#!/usr/bin/env python3
"""
AEPsych wrapper for adaptive stimulus selection in the psychophysical experiment.
"""

import numpy as np
import torch
from aepsych.server import AEPsychServer
from aepsych.config import Config
from aepsych.strategy import Strategy
from typing import Tuple, List, Optional, Dict, Any
import json
import os
from pathlib import Path

class PsychophysicalAEPsych:
    """Wrapper class for AEPsych integration with psychophysical color discrimination."""

    def __init__(self, config_path: str = None, participant_id: str = "P01"):
        """
        Initialize AEPsych for the psychophysical experiment.

        Args:
            config_path: Path to experiment configuration
            participant_id: Participant identifier
        """
        self.participant_id = participant_id
        self.trial_count = 0
        self.sobol_trials_completed = 0
        self.adaptive_trials_completed = 0

        # Load experiment configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "experiment_params.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # AEPsych configuration
        adaptive_config = self.config['adaptive']
        self.total_trials = adaptive_config['total_trials']
        self.sobol_trials = adaptive_config['sobol_trials']
        self.adaptive_trials = adaptive_config['adaptive_trials']
        self.update_frequency = adaptive_config['update_frequency']

        # Model space bounds
        stimuli_config = self.config['stimuli']
        self.model_bounds = stimuli_config['model_space_bounds']

        # Initialize AEPsych server
        self._setup_aepsych()

        # Trial history
        self.trial_history = []

    def _setup_aepsych(self):
        """Set up AEPsych server and strategy."""
        # Create AEPsych configuration in INI format
        config_str = f"""
[common]
parnames = [x, y]
lb = [{self.model_bounds[0]}, {self.model_bounds[0]}]
ub = [{self.model_bounds[1]}, {self.model_bounds[1]}]
stimuli_per_trial = 1
outcome_types = [binary]
target = 0.5
strategy_names = [init_strat, opt_strat]
acqf = MCLevelSetEstimation
model = GPClassificationModel

[init_strat]
min_asks = {self.sobol_trials}
generator = SobolGenerator
min_total_outcome_occurrences = 0

[opt_strat]
min_asks = {self.adaptive_trials}
refit_every = {self.update_frequency}
generator = OptimizeAcqfGenerator
min_total_outcome_occurrences = 0

[GPClassificationModel]
inducing_size = 100
mean_covar_factory = default_mean_covar_factory

[OptimizeAcqfGenerator]
restarts = 10
samps = 1000

[MCLevelSetEstimation]
beta = 3.84
objective = ProbitObjective
"""

        # Create config object
        self.aepsych_config = Config(config_str=config_str)

        # Initialize server
        self.server = AEPsychServer()

        # Send setup message to initialize the experiment
        setup_message = {
            "type": "setup",
            "message": {"config_str": config_str}
        }

        response = self.server.handle_request(setup_message)
        print(f"AEPsych setup response: {response}")

    def should_use_adaptive(self) -> bool:
        """
        Determine if we should use adaptive sampling or continue with Sobol.

        Returns:
            True if adaptive sampling should be used
        """
        return self.trial_count >= self.sobol_trials

    def get_next_trial(self) -> Tuple[np.ndarray, str]:
        """
        Get the next stimulus pair for testing.

        Returns:
            Tuple of (model_space_coordinates, trial_type)
        """
        self.trial_count += 1

        # Ask AEPsych for next trial - it handles strategy switching internally
        message = {
            "type": "ask",
            "message": {}
        }

        try:
            response = self.server.handle_request(message)

            if isinstance(response, dict) and "config" in response and "x" in response["config"] and "y" in response["config"]:
                coords = np.array([
                    float(response["config"]["x"][0]),
                    float(response["config"]["y"][0])
                ])

                # Determine trial type based on strategy
                trial_type = "ADAPTIVE" if self.should_use_adaptive() else "SOBOL"
            else:
                # Fallback to random if AEPsych fails
                print(f"Warning: Unexpected AEPsych response structure: {response}")
                coords = np.array([
                    np.random.uniform(self.model_bounds[0], self.model_bounds[1]),
                    np.random.uniform(self.model_bounds[0], self.model_bounds[1])
                ])
                trial_type = "FALLBACK"

        except Exception as e:
            print(f"Error asking AEPsych for trial: {e}")
            # Fallback to random
            coords = np.array([
                np.random.uniform(self.model_bounds[0], self.model_bounds[1]),
                np.random.uniform(self.model_bounds[0], self.model_bounds[1])
            ])
            trial_type = "FALLBACK"

        return coords, trial_type

    def tell_trial_result(self, coords: np.ndarray, response: int, trial_type: str):
        """
        Tell AEPsych the result of a trial.

        Args:
            coords: Model space coordinates of the stimulus pair
            response: Participant response (0 or 1 for correct/incorrect)
            trial_type: Type of trial ("SOBOL" or "ADAPTIVE")
        """
        # Tell AEPsych about all trials (both Sobol and adaptive)
        if trial_type in ["SOBOL", "ADAPTIVE"]:
            message = {
                "type": "tell",
                "message": {
                    "config": {
                        "x": [float(coords[0])],
                        "y": [float(coords[1])]
                    },
                    "outcome": int(response)
                }
            }

            try:
                self.server.handle_request(message)
            except Exception as e:
                print(f"Warning: Failed to tell AEPsych about trial result: {e}")

        # Store trial history
        self.trial_history.append({
            'trial': self.trial_count,
            'coords': coords.tolist(),
            'response': response,
            'trial_type': trial_type
        })

    def save_model_state(self, filepath: str):
        """
        Save the current AEPsych model state.

        Args:
            filepath: Path to save the model
        """
        try:
            # Get model state from server
            message = {"type": "get_model", "version": "0.1"}
            response = self.server.handle_request(message)

            # Save to file
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            with open(filepath, 'w') as f:
                json.dump(response, f, indent=2)

        except Exception as e:
            print(f"Warning: Failed to save model state: {e}")

    def load_model_state(self, filepath: str):
        """
        Load AEPsych model state from file.

        Args:
            filepath: Path to load the model from
        """
        try:
            with open(filepath, 'r') as f:
                model_state = json.load(f)

            # Load into server
            message = {
                "type": "set_model",
                "model": model_state,
                "version": "0.1"
            }

            self.server.handle_request(message)

        except Exception as e:
            print(f"Warning: Failed to load model state: {e}")

    def get_trial_history(self) -> List[Dict[str, Any]]:
        """
        Get the history of all trials.

        Returns:
            List of trial dictionaries
        """
        return self.trial_history.copy()

    def reset(self):
        """Reset the AEPsych experiment."""
        self.trial_count = 0
        self.sobol_trials_completed = 0
        self.adaptive_trials_completed = 0
        self.trial_history = []
        self._setup_aepsych()

# Simplified version for testing without full AEPsych complexity
class SimpleAEPsychSimulator:
    """Simplified AEPsych simulator for initial testing."""

    def __init__(self, config_path: str = None):
        """Initialize simple AEPsych simulator."""
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "experiment_params.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        adaptive_config = self.config['adaptive']
        self.total_trials = adaptive_config['total_trials']
        self.sobol_trials = adaptive_config['sobol_trials']

        stimuli_config = self.config['stimuli']
        self.model_bounds = stimuli_config['model_space_bounds']

        self.trial_count = 0
        self.trial_history = []

    def get_next_trial(self) -> Tuple[np.ndarray, str]:
        """Get next trial coordinates."""
        self.trial_count += 1

        # Simple random sampling for now
        x = np.random.uniform(self.model_bounds[0], self.model_bounds[1])
        y = np.random.uniform(self.model_bounds[0], self.model_bounds[1])
        coords = np.array([x, y])

        trial_type = "SOBOL" if self.trial_count <= self.sobol_trials else "ADAPTIVE"

        return coords, trial_type

    def tell_trial_result(self, coords: np.ndarray, response: int, trial_type: str):
        """Record trial result."""
        self.trial_history.append({
            'trial': self.trial_count,
            'coords': coords.tolist(),
            'response': response,
            'trial_type': trial_type
        })

    def get_trial_history(self) -> List[Dict[str, Any]]:
        """Get trial history."""
        return self.trial_history.copy()

# Test functions
def test_aepsych_wrapper():
    """Test the AEPsych wrapper."""
    print("Testing AEPsych wrapper...")

    # Test simple simulator
    simulator = SimpleAEPsychSimulator()

    for i in range(5):
        coords, trial_type = simulator.get_next_trial()
        print(f"Trial {i+1}: {trial_type} at {coords}")

        # Simulate random response
        response = np.random.randint(0, 2)
        simulator.tell_trial_result(coords, response, trial_type)

    history = simulator.get_trial_history()
    print(f"Completed {len(history)} trials")

    print("AEPsych wrapper test completed!")

if __name__ == "__main__":
    test_aepsych_wrapper()