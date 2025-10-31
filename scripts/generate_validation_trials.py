#!/usr/bin/env python3
"""
Generate validation trials using Method of Constant Stimuli (MOCS).

This script creates 6,000 validation trials across 25 reference points,
with 12 comparison stimulus levels for each condition. These trials
are used to independently validate the WPPM predictions.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from adaptive_engine.utils.transformations import ColorTransformations

class MOCSGenerator:
    """Generate validation trials using Method of Constant Stimuli."""

    def __init__(self, config_path: str = None, participant_id: str = "P01"):
        """
        Initialize MOCS generator.

        Args:
            config_path: Path to experiment configuration
            participant_id: Participant identifier
        """
        self.participant_id = participant_id

        # Load experiment configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "experiment_params.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Validation parameters from config
        validation_config = self.config['validation']
        self.total_trials = validation_config['total_trials']  # 6000
        self.reference_points = validation_config['reference_points']  # 25
        self.comparison_levels = validation_config['comparison_levels']  # 12

        # Model space bounds
        stimuli_config = self.config['stimuli']
        self.model_bounds = stimuli_config['model_space_bounds']

        # Initialize color transformations
        self.transform = ColorTransformations()

        # Calculate trials per condition: 6000 total / (25 refs × 12 levels) = 20 trials per condition
        total_conditions = self.reference_points * self.comparison_levels  # 25 × 12 = 300
        self.trials_per_condition = self.total_trials // total_conditions  # 6000 // 300 = 20
        print(f"Generating {self.total_trials} validation trials:")
        print(f"  {self.reference_points} reference points")
        print(f"  {self.comparison_levels} comparison levels per reference")
        print(f"  {self.trials_per_condition} trials per condition")

    def generate_reference_points(self) -> np.ndarray:
        """
        Generate 25 reference points in model space.

        Uses Sobol sampling for good space-filling properties.

        Returns:
            Array of shape (25, 2) with reference coordinates
        """
        # For reproducibility, use fixed seed
        np.random.seed(42)

        # Generate Sobol-like sampling in the model space
        # Since scipy.stats.qmc is not available in all environments,
        # we'll use a simple grid + jitter approach

        # Create a 5x5 grid
        n_per_dim = 5  # sqrt(25) = 5
        coords_1d = np.linspace(self.model_bounds[0], self.model_bounds[1], n_per_dim)

        references = []
        for x in coords_1d:
            for y in coords_1d:
                # Add small jitter to avoid exact grid points
                jitter_x = np.random.uniform(-0.1, 0.1)
                jitter_y = np.random.uniform(-0.1, 0.1)
                ref_x = np.clip(x + jitter_x, self.model_bounds[0], self.model_bounds[1])
                ref_y = np.clip(y + jitter_y, self.model_bounds[0], self.model_bounds[1])
                references.append([ref_x, ref_y])

        return np.array(references)

    def generate_chromatic_directions(self, n_directions: int = 4) -> np.ndarray:
        """
        Generate chromatic directions around each reference.

        For each reference point, we sample directions in color space.
        The paper mentions testing along specific chromatic directions.

        Args:
            n_directions: Number of directions to test per reference

        Returns:
            Array of direction angles in radians
        """
        # For each reference, we'll test a few chromatic directions
        # These should be chosen to cover the color discrimination space well

        # Use fixed angles for reproducibility, but randomize slightly
        base_angles = np.linspace(0, 2*np.pi, n_directions, endpoint=False)

        # Since we have 12 comparison levels total, and we're generating 4 directions,
        # we'll have 3 levels per direction
        self.levels_per_direction = self.comparison_levels // n_directions

        return base_angles

    def generate_comparison_levels(self, reference: np.ndarray, direction: float,
                                 n_levels: int) -> List[float]:
        """
        Generate comparison stimulus levels along a chromatic direction.

        Args:
            reference: Reference point coordinates [x, y]
            direction: Direction angle in radians
            n_levels: Number of comparison levels to generate

        Returns:
            List of perturbation magnitudes (in model space units)
        """
        # Generate perturbation levels for this direction
        # Based on pilot data mentioned in the paper, we want a range from
        # clearly discriminable to near-threshold differences

        min_perturbation = 0.02  # Small perturbation (hard trials)
        max_perturbation = 0.3   # Large perturbation (easy trials)

        # Use evenly spaced levels from negative to positive
        levels = np.linspace(-max_perturbation, max_perturbation, n_levels)

        return levels.tolist()

    def create_trial_dataframe(self) -> pd.DataFrame:
        """
        Create a DataFrame with all validation trials.

        Returns:
            DataFrame with columns: participant_id, reference_x, reference_y,
            comparison_x, comparison_y, direction, level_index, trial_type
        """
        trials_data = []

        # Generate reference points
        references = self.generate_reference_points()
        directions = self.generate_chromatic_directions()

        trial_index = 0

        for ref_idx, reference in enumerate(references):
            print(f"Generating trials for reference {ref_idx + 1}/{len(references)}")

            for dir_idx, direction in enumerate(directions):
                # Generate comparison levels for this direction
                levels = self.generate_comparison_levels(reference, direction,
                                                       self.levels_per_direction)

                for level_idx, level in enumerate(levels):
                    # Create comparison stimulus by perturbing along the direction
                    perturbation = np.array([level * np.cos(direction),
                                           level * np.sin(direction)])

                    comparison = reference + perturbation

                    # Ensure comparison stays within bounds
                    comparison = np.clip(comparison, self.model_bounds[0], self.model_bounds[1])

                    # Generate multiple trials per condition (for psychometric function fitting)
                    for trial_rep in range(self.trials_per_condition):
                        # Randomize which stimulus is presented where (reference vs comparison)
                        # and which position gets which stimulus
                        stimulus_positions = np.random.permutation(3)
                        ref_position = stimulus_positions[0]  # First two positions get reference
                        comp_position = stimulus_positions[2]  # Last position gets comparison

                        # Convert to RGB for storage
                        ref_rgb = self.transform.model_space_to_rgb(reference)
                        comp_rgb = self.transform.model_space_to_rgb(comparison)

                        trial_data = {
                            'participant_id': self.participant_id,
                            'session_index': 1,  # Will be set when used
                            'trial_index': trial_index,
                            'trial_type': 'VALIDATION',
                            'reference_x': reference[0],
                            'reference_y': reference[1],
                            'comparison_x': comparison[0],
                            'comparison_y': comparison[1],
                            'ref_rgb_r': ref_rgb[0],
                            'ref_rgb_g': ref_rgb[1],
                            'ref_rgb_b': ref_rgb[2],
                            'comp_rgb_r': comp_rgb[0],
                            'comp_rgb_g': comp_rgb[1],
                            'comp_rgb_b': comp_rgb[2],
                            'direction': direction,
                            'level_magnitude': level,
                            'reference_position': ref_position,
                            'comparison_position': comp_position,
                            'ref_idx': ref_idx,
                            'dir_idx': dir_idx,
                            'level_idx': level_idx,
                            'trial_rep': trial_rep
                        }

                        trials_data.append(trial_data)
                        trial_index += 1

        return pd.DataFrame(trials_data)

    def save_validation_trials(self, output_path: str = None) -> str:
        """
        Generate and save validation trials to CSV.

        Args:
            output_path: Path to save the CSV file

        Returns:
            Path to the saved file
        """
        if output_path is None:
            output_path = (Path(__file__).parent.parent /
                          "config" / "participants" /
                          f"{self.participant_id}_validation_trials.csv")

        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate trials
        print("Generating validation trials...")
        trials_df = self.create_trial_dataframe()

        # Shuffle trials for randomization
        trials_df = trials_df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Save to CSV
        trials_df.to_csv(output_path, index=False)

        print(f"Saved {len(trials_df)} validation trials to {output_path}")
        print(f"Trials per reference point: {len(trials_df) // self.reference_points}")

        return str(output_path)

def main():
    """Main function to generate validation trials."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate MOCS validation trials")
    parser.add_argument("--participant", default="P01", help="Participant ID")
    parser.add_argument("--output", help="Output CSV path")
    parser.add_argument("--config", help="Path to experiment config JSON")

    args = parser.parse_args()

    # Generate validation trials
    generator = MOCSGenerator(
        config_path=args.config,
        participant_id=args.participant
    )

    output_path = generator.save_validation_trials(args.output)

    print(f"\nValidation trials generated successfully!")
    print(f"File: {output_path}")

if __name__ == "__main__":
    main()
