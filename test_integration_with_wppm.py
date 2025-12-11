#!/usr/bin/env python3
"""
Integration test combining stimulus presentation with AEPsych and WPPM fitting.

This test implements the full pipeline from Hong et al. (2025):
1. AEPsych selects 4D trial parameters (reference + comparison in model space)
2. Color transformations convert to RGB for display
3. 3AFC task presented to participant
4. Response fed back to AEPsych
5. Post-hoc WPPM model fitting to collected data
"""

import sys
import os
from pathlib import Path
import numpy as np
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import components
from adaptive_engine.color import ColorTransformations
from adaptive_engine.aepsych_wrapper import PsychophysicalAEPsychDelta

# Try to import WPPM components
try:
    import jax.numpy as jnp
    from adaptive_engine.wppm_fitter import (
        fit_wppm, WPPMConfig, TrialData, 
        get_covariance_at, predict_threshold, save_wppm
    )
    WPPM_AVAILABLE = True
except ImportError as e:
    print(f"Warning: WPPM dependencies not available: {e}")
    print("WPPM fitting will be skipped.")
    WPPM_AVAILABLE = False


class IntegrationTestWithWPPM:
    """
    Full integration test: AEPsych adaptive sampling -> WPPM fitting.
    
    Follows the architecture from Hong et al. (2025):
    - Phase 1: Sobol initialization (900 trials)
    - Phase 2: Adaptive sampling via AEPsych (5100 trials)
    - Phase 3: Post-hoc WPPM model fitting
    """

    def __init__(self, config_path: str = None, use_display: bool = False):
        """
        Initialize components.
        
        Args:
            config_path: Path to experiment configuration
            use_display: If True, use pygame for stimulus presentation
        """
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "experiment_params.json"
        
        self.config_path = Path(config_path)
        self.use_display = use_display
        
        # Load config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize color transformations (from Appendix 1)
        self.transform = ColorTransformations()
        
        # Initialize AEPsych wrapper (4D: reference + comparison)
        self.aepsych = PsychophysicalAEPsychDelta(str(self.config_path))
        
        # Initialize stimulus presentation if using display
        self.stimulus = None
        if self.use_display:
            try:
                from adaptive_engine.simple_stimulus import StimulusPresentation
                self.stimulus = StimulusPresentation(str(self.config_path))
            except ImportError:
                print("Warning: Could not import StimulusPresentation, falling back to simulation")
                self.use_display = False
        
        # Results storage
        self.trial_results = []
        
    def run_trial_loop(self, num_trials: int = 100, simulate: bool = True):
        """
        Run the trial collection loop.
        
        Args:
            num_trials: Number of trials to run
            simulate: If True, simulate participant responses (ignored if use_display=True)
        """
        # Override simulate if using display
        if self.use_display and self.stimulus is not None:
            simulate = False
            
        print(f"Starting trial collection ({num_trials} trials)...")
        print(f"Mode: {'Display' if self.use_display else 'Simulation'}")
        print("-" * 50)
        
        for trial_num in range(1, num_trials + 1):
            # Check if we should continue (for display mode)
            if self.use_display and self.stimulus is not None and not self.stimulus.running:
                print("Experiment terminated by user")
                break
                
            # 1. Get next trial from AEPsych (4D: ref + comp in model space)
            ref_model, comp_model, trial_type = self.aepsych.get_next_trial()
            
            # 2. Convert model space to RGB
            ref_rgb = self.transform.model_space_to_rgb(ref_model)
            comp_rgb = self.transform.model_space_to_rgb(comp_model)
            
            # 3. Random position for comparison
            comparison_index = np.random.randint(0, 3)
            
            # 4. Get response (display or simulation)
            if self.use_display and self.stimulus is not None:
                # Present actual stimuli via pygame
                response, is_correct, rt = self.stimulus.run_trial_from_pipeline(
                    ref_rgb=ref_rgb,
                    comp_rgb=comp_rgb,
                    comparison_index=comparison_index,
                    show_feedback=True
                )
                
                if response is None:
                    # Quit or timeout
                    if not self.stimulus.running:
                        break
                    # Timeout - record as incorrect
                    is_correct = False
                    rt = None
            else:
                # Simulate response
                response, is_correct, rt = self._simulate_response(
                    ref_model, comp_model, comparison_index
                )
            
            # 5. Tell AEPsych the result
            binary_response = 1 if is_correct else 0
            self.aepsych.tell_trial_result(ref_model, comp_model, binary_response, trial_type)
            
            # 6. Build stimulus list for storage
            stimuli_rgb = []
            for i in range(3):
                if i == comparison_index:
                    stimuli_rgb.append(comp_rgb.copy())
                else:
                    stimuli_rgb.append(ref_rgb.copy())
            
            # 7. Store result
            self.trial_results.append({
                'trial': trial_num,
                'trial_type': trial_type,
                'ref_model': ref_model.tolist(),
                'comp_model': comp_model.tolist(),
                'ref_rgb': ref_rgb.tolist(),
                'comp_rgb': comp_rgb.tolist(),
                'stimuli_rgb': [s.tolist() for s in stimuli_rgb],
                'comparison_index': comparison_index,
                'response': response,
                'correct': is_correct,
                'response_time': rt
            })
            
            # Progress update
            if trial_num % 10 == 0 or trial_num == 1:
                accuracy = sum(1 for r in self.trial_results if r['correct']) / len(self.trial_results)
                print(f"Trial {trial_num}/{num_trials} ({trial_type}): "
                      f"accuracy = {accuracy:.1%}")
        
        # Cleanup display if used
        if self.use_display and self.stimulus is not None:
            self.stimulus.cleanup()
        
        print("\n" + "=" * 50)
        print("Trial collection complete!")
        self._print_summary()
    
    def _simulate_response(
        self, 
        ref_model: np.ndarray, 
        comp_model: np.ndarray,
        correct_index: int
    ) -> tuple:
        """
        Simulate participant response based on stimulus difficulty.
        
        Uses a simple model where accuracy depends on distance in model space.
        """
        # Compute difficulty (smaller distance = harder)
        distance = np.linalg.norm(comp_model - ref_model)
        
        # Psychometric function: probability correct increases with distance
        # At distance=0: p=0.333 (chance for 3AFC)
        # At distance=0.5: p≈0.8
        # At distance=1.0: p≈0.95
        slope = 4.0
        threshold = 0.15
        p_correct = 0.333 + 0.667 * (1 - np.exp(-slope * (distance / threshold)))
        p_correct = np.clip(p_correct, 0.333, 0.99)
        
        # Simulate response
        is_correct = np.random.random() < p_correct
        
        if is_correct:
            response = correct_index
        else:
            # Random incorrect response
            wrong_indices = [i for i in range(3) if i != correct_index]
            response = np.random.choice(wrong_indices)
        
        # Simulate response time (faster for easier trials)
        rt = 0.5 + 0.5 * np.exp(-2 * distance) + 0.1 * np.random.random()
        
        return response, is_correct, rt
    
    def _print_summary(self):
        """Print summary statistics."""
        n_trials = len(self.trial_results)
        n_correct = sum(1 for r in self.trial_results if r['correct'])
        
        print(f"\nSummary:")
        print(f"  Total trials: {n_trials}")
        print(f"  Correct: {n_correct} ({100*n_correct/n_trials:.1f}%)")
        
        # By trial type
        for trial_type in ['SOBOL', 'ADAPTIVE', 'FALLBACK']:
            trials = [r for r in self.trial_results if r['trial_type'] == trial_type]
            if trials:
                correct = sum(1 for r in trials if r['correct'])
                print(f"  {trial_type}: {len(trials)} trials, {correct} correct ({100*correct/len(trials):.1f}%)")
    
    def fit_wppm_model(self):
        """
        Fit WPPM model to collected trial data.
        
        This is Phase 3 from the architecture: post-hoc model fitting.
        """
        print("\n" + "=" * 50)
        print("PHASE 3: WPPM MODEL FITTING")
        print("=" * 50)
        
        if not WPPM_AVAILABLE:
            print("WPPM dependencies not available. Skipping.")
            print("Install JAX: pip install jax jaxlib")
            return None
        
        if not self.trial_results:
            print("No trial data to fit!")
            return None
        
        # Convert to WPPM format
        ref_coords = jnp.array([r['ref_model'] for r in self.trial_results])
        comp_coords = jnp.array([r['comp_model'] for r in self.trial_results])
        responses = jnp.array([1 if r['correct'] else 0 for r in self.trial_results])
        
        trial_data = TrialData(
            reference_coords=ref_coords,
            comparison_coords=comp_coords,
            responses=responses
        )
        
        print(f"\nFitting WPPM to {len(self.trial_results)} trials...")
        
        # Configure WPPM
        config = WPPMConfig(
            n_basis=5,
            n_mc_samples=500,
            bandwidth=1.0,
            gamma=3e-4,
            epsilon=0.5
        )
        
        # Fit model
        import jax.random as jr
        params, losses = fit_wppm(
            trial_data,
            config=config,
            n_iterations=200,
            learning_rate=1e-2,
            key=jr.PRNGKey(42),
            verbose=True,
            print_every=50
        )
        
        # Evaluate fitted model
        print("\nEvaluating fitted model...")
        
        # Covariance at origin
        origin = jnp.array([0.0, 0.0])
        cov_origin = get_covariance_at(params, origin, config)
        print(f"  Covariance at origin:\n{cov_origin}")
        
        # Threshold along cardinal directions
        for direction, name in [([1, 0], "x-axis"), ([0, 1], "y-axis")]:
            threshold = predict_threshold(
                params, origin, jnp.array(direction), config,
                target_accuracy=0.667  # 66.7% as in paper
            )
            print(f"  66.7% threshold along {name}: {threshold:.4f}")
        
        return {'params': params, 'losses': losses, 'config': config}
    
    def save_results(self, output_dir: str = None):
        """Save trial results and WPPM fit."""
        if output_dir is None:
            output_dir = Path(__file__).parent / "data" / "results"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save trial data
        trial_file = output_dir / "trial_results.json"
        with open(trial_file, 'w') as f:
            json.dump(self.trial_results, f, indent=2)
        print(f"Saved trial results to {trial_file}")
        
        # Save as CSV for WPPM
        csv_file = output_dir / "trial_data.csv"
        with open(csv_file, 'w') as f:
            f.write("ref_x,ref_y,comp_x,comp_y,response\n")
            for r in self.trial_results:
                f.write(f"{r['ref_model'][0]},{r['ref_model'][1]},"
                       f"{r['comp_model'][0]},{r['comp_model'][1]},"
                       f"{1 if r['correct'] else 0}\n")
        print(f"Saved CSV data to {csv_file}")
    
    def get_wppm_trial_data(self):
        """Get trial data formatted for WPPM fitting."""
        return self.aepsych.get_trial_data_for_wppm()


def main():
    """Run the integration test."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AEPsych + WPPM Integration Test")
    parser.add_argument('--display', action='store_true', 
                       help='Use pygame display for stimulus presentation')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of trials to run (default: 50)')
    parser.add_argument('--skip-wppm', action='store_true',
                       help='Skip WPPM model fitting')
    args = parser.parse_args()
    
    print("=" * 60)
    print("Integration Test: AEPsych + Color Transforms + WPPM")
    print("=" * 60)
    
    # Create mock config if needed
    config_dir = Path(__file__).parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / "experiment_params.json"
    
    if not config_file.exists():
        mock_config = {
            'adaptive': {
                'total_trials': 100,
                'sobol_trials': 20,
                'adaptive_trials': 80,
                'update_frequency': 10,
                'target_threshold': 0.667
            },
            'stimuli': {
                'model_space_bounds': [-0.7, 0.7],
                'stimulus_size_deg': 2.0,
                'stimulus_spacing_deg': 5.0
            },
            'display': {
                'width_px': 1920,
                'height_px': 1080,
                'background_rgb': [0.5, 0.5, 0.5],
                'pixels_per_degree': 50
            },
            'timing': {
                'fixation_ms': 500,
                'stimulus_duration_ms': 200,
                'response_timeout_ms': 5000,
                'iti_ms': 500,
                'feedback_ms': 200
            }
        }
        with open(config_file, 'w') as f:
            json.dump(mock_config, f, indent=2)
        print(f"Created mock config at {config_file}")
    
    try:
        # Initialize test
        test = IntegrationTestWithWPPM(str(config_file), use_display=args.display)
        
        # Run trials
        test.run_trial_loop(num_trials=args.trials, simulate=not args.display)
        
        # # Fit WPPM (unless skipped)
        if not args.skip_wppm:
            wppm_results = test.fit_wppm_model()
            save_wppm(
                "wppm_fitted.npz",
                wppm_results['params'],
                wppm_results['config'],
                losses=wppm_results['losses'],
                metadata={'n_trials': len(test.trial_results), 'demo': True}
            )
        
        # # Save results
        test.save_results()
        
        print("\n" + "=" * 60)
        print("Integration test completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()