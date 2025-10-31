#!/usr/bin/env python3
"""
Basic integration test combining stimulus presentation with AEPsych.
Tests the fundamental trial loop without the dual-computer setup.
"""

import sys
import os
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from adaptive_engine.simple_stimulus import SimpleStimulusPresentation
from adaptive_engine.aepsych_wrapper import PsychophysicalAEPsych
from adaptive_engine.utils.transformations import ColorTransformations
import json

class BasicIntegrationTest:
    """Test the basic integration of stimulus presentation and adaptive sampling."""

    def __init__(self):
        """Initialize the integration test."""
        self.config_path = Path(__file__).parent / "config" / "experiment_params.json"

        # Initialize components
        self.stimulus = SimpleStimulusPresentation(self.config_path)
        self.aepsych = PsychophysicalAEPsych(self.config_path)
        self.transform = ColorTransformations()

        # Results storage
        self.results = []

    def run_trial_loop(self, num_trials: int = 10):
        """
        Run the basic trial loop.
        Every third trial allows manual user interaction.

        Args:
            num_trials: Number of trials to run
        """
        print("Starting basic integration test...")
        print(f"Running {num_trials} trials")
        print("Note: Every 3rd trial will require manual user input")
        print("-" * 50)

        for trial_num in range(1, num_trials + 1):
            print(f"\nTrial {trial_num}/{num_trials}")

            # 1. Get next trial from AEPsych
            model_coords, trial_type = self.aepsych.get_next_trial()
            print(f"AEPsych suggests: {trial_type} at model coords {model_coords}")

            # 2. Convert to RGB stimuli
            # For simplicity, use the model coords to create a reference stimulus
            # and generate a comparison stimulus
            ref_rgb = self.transform.model_space_to_rgb(model_coords)
            stimuli_rgb, correct_index = self.transform.create_trial_stimuli(ref_rgb)

            print(f"Stimuli RGB: {[f'[{r:.2f},{g:.2f},{b:.2f}]' for r,g,b in stimuli_rgb]}")
            print(f"Correct response: {correct_index + 1} (stimulus {correct_index})")

            # 3. Present stimuli and get response
            if trial_num % 3 == 0 and self._is_interactive_environment():
                # Manual interaction every third trial (only if in interactive environment)
                print("\n--- MANUAL INTERACTION REQUIRED ---")
                print("Three colored circles will be displayed.")
                print("Press the number key (1, 2, or 3) to select which circle you think is different.")
                print("The trial will start automatically in 2 seconds...")

                # Brief pause to let user prepare
                import time
                time.sleep(2.0)

                try:
                    # Present the actual stimuli and get user response
                    user_response, user_correct, response_time = self.stimulus.run_trial()
                    simulated_response = user_response
                    simulated_correct = user_correct
                    simulated_rt = response_time

                    print(f"User response: {user_response + 1}, Correct: {user_correct}, RT: {response_time:.2f}s")
                except Exception as e:
                    print(f"Manual interaction failed: {e}. Falling back to simulation.")
                    simulated_response = self._simulate_participant_response(correct_index)
                    simulated_correct = (simulated_response == correct_index)
                    simulated_rt = 0.5 + np.random.random() * 0.5
                    print(f"Simulated response: {simulated_response + 1}, Correct: {simulated_correct}")

            else:
                # Automatic simulation for other trials or non-interactive environments
                simulated_response = self._simulate_participant_response(correct_index)
                simulated_correct = (simulated_response == correct_index)
                simulated_rt = 0.5 + np.random.random() * 0.5  # 0.5-1.0 seconds

                if trial_num % 3 == 0:
                    print("Simulated response (manual trial in non-interactive environment):", end=" ")
                else:
                    print("Simulated response:", end=" ")
                print(f"{simulated_response + 1}, Correct: {simulated_correct}")

            # 4. Tell AEPsych the result
            # Convert response to binary (1 for correct, 0 for incorrect)
            binary_response = 1 if simulated_correct else 0
            self.aepsych.tell_trial_result(model_coords, binary_response, trial_type)

            # 5. Store results
            was_manual_attempt = trial_num % 3 == 0
            actual_interaction_type = 'manual' if (was_manual_attempt and self._is_interactive_environment()) else 'simulated'

            trial_result = {
                'trial': trial_num,
                'model_coords': model_coords.tolist(),
                'stimuli_rgb': stimuli_rgb,
                'correct_index': correct_index,
                'response': simulated_response,
                'correct': simulated_correct,
                'response_time': simulated_rt,
                'trial_type': trial_type,
                'interaction_type': actual_interaction_type
            }
            self.results.append(trial_result)

        print("\n" + "="*50)
        print("Integration test completed!")

    def _is_interactive_environment(self) -> bool:
        """
        Check if we're in an interactive environment where manual input is possible.

        Returns:
            True if interactive environment, False otherwise
        """
        import os
        import sys

        # Check if we have a display (for GUI applications)
        display_available = os.environ.get('DISPLAY') is not None or os.name == 'nt'

        # Check if we're running in a terminal that supports interaction
        # This is a simple heuristic - in production you'd want more sophisticated detection
        is_interactive = (
            sys.stdout.isatty() and  # Running in a TTY
            display_available and    # Display available for GUI
            not os.environ.get('CI') # Not running in CI environment
        )

        return is_interactive

    def _simulate_participant_response(self, correct_index: int) -> int:
        """
        Simulate participant response with some noise.

        Args:
            correct_index: Index of the correct stimulus

        Returns:
            Simulated response index
        """
        import numpy as np

        # 80% correct rate
        if np.random.random() < 0.8:
            return correct_index
        else:
            # Random incorrect response
            return np.random.choice([i for i in range(3) if i != correct_index])

    def save_results(self):
        """Save test results."""
        results_dir = Path(__file__).parent / "data" / "P01" / "raw"
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = results_dir / "integration_test_results.json"

        # Convert numpy types to Python types for JSON serialization
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if hasattr(value, 'tolist'):  # numpy array
                    serializable_result[key] = value.tolist()
                elif hasattr(value, 'item'):  # numpy scalar
                    serializable_result[key] = value.item()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)

        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)

        print(f"Results saved to {results_file}")

        # Print summary
        correct_trials = sum(1 for r in self.results if r['correct'])
        total_trials = len(self.results)
        accuracy = correct_trials / total_trials * 100

        # Separate statistics for manual vs simulated trials
        manual_results = [r for r in self.results if r['interaction_type'] == 'manual']
        simulated_results = [r for r in self.results if r['interaction_type'] == 'simulated']

        manual_correct = sum(1 for r in manual_results if r['correct'])
        simulated_correct = sum(1 for r in simulated_results if r['correct'])

        print(f"\nSummary:")
        print(f"Total trials: {total_trials}")
        print(f"Correct: {correct_trials}")
        print(f"Accuracy: {accuracy:.1f}%")

        if manual_results:
            manual_accuracy = manual_correct / len(manual_results) * 100
            print(f"\nManual trials ({len(manual_results)}): {manual_correct} correct ({manual_accuracy:.1f}%)")

        if simulated_results:
            simulated_accuracy = simulated_correct / len(simulated_results) * 100
            print(f"Simulated trials ({len(simulated_results)}): {simulated_correct} correct ({simulated_accuracy:.1f}%)")

def main():
    """Run the basic integration test."""
    import numpy as np

    try:
        test = BasicIntegrationTest()
        test.run_trial_loop(num_trials=21)  # Use 21 to get 7 manual trials (3, 6, 9, 12, 15, 18, 21)
        test.save_results()

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\nBasic integration test finished")

if __name__ == "__main__":
    main()
