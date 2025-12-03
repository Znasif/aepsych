#!/usr/bin/env python3
"""
Stimulus presentation for 3AFC color discrimination task.

Displays three colored circles in triangular arrangement.
Accepts RGB stimuli from the AEPsych/WPPM pipeline or generates random stimuli for testing.

Based on Hong et al. (2025) experimental paradigm:
- 3AFC oddity task: two identical reference stimuli + one comparison
- Participant identifies the "odd one out"
- Response via keyboard (1, 2, 3) or configurable input
"""

import pygame
import json
import random
import time
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional


class StimulusPresentation:
    """
    3AFC stimulus presentation for color discrimination experiments.
    
    Can operate in two modes:
    1. Pipeline mode: Accept RGB stimuli from AEPsych/color transformation pipeline
    2. Demo mode: Generate random stimuli for testing
    """

    def __init__(self, config_path: str = None, fullscreen: bool = False):
        """
        Initialize the stimulus presentation system.
        
        Args:
            config_path: Path to experiment configuration JSON
            fullscreen: Whether to run in fullscreen mode
        """
        pygame.init()

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "experiment_params.json"
        
        self.config_path = Path(config_path)
        
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        else:
            # Use defaults if config doesn't exist
            self.config = self._default_config()

        # Display settings
        display_config = self.config.get('display', {})
        self.screen_width = display_config.get('width_px', 1920)
        self.screen_height = display_config.get('height_px', 1080)
        
        # Background color (default mid-gray as adapting point)
        bg_rgb = display_config.get('background_rgb', [0.5, 0.5, 0.5])
        self.background_color = tuple(int(c * 255) for c in bg_rgb)

        # Stimulus settings
        stimulus_config = self.config.get('stimuli', {})
        self.stimulus_size_deg = stimulus_config.get('stimulus_size_deg', 2.0)
        self.stimulus_spacing_deg = stimulus_config.get('stimulus_spacing_deg', 5.0)
        
        # Convert degrees to pixels (approximate, should use monitor calibration)
        ppd = display_config.get('pixels_per_degree', 50)  # pixels per degree
        self.stimulus_size_px = int(self.stimulus_size_deg * ppd)
        self.stimulus_spacing_px = int(self.stimulus_spacing_deg * ppd)

        # Timing settings (in milliseconds)
        timing_config = self.config.get('timing', {})
        self.fixation_ms = timing_config.get('fixation_ms', 500)
        self.stimulus_duration_ms = timing_config.get('stimulus_duration_ms', 200)
        self.response_timeout_ms = timing_config.get('response_timeout_ms', 5000)
        self.iti_ms = timing_config.get('iti_ms', 500)
        self.feedback_ms = timing_config.get('feedback_ms', 200)

        # Initialize display
        if fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            info = pygame.display.Info()
            self.screen_width = info.current_w
            self.screen_height = info.current_h
        else:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        
        pygame.display.set_caption("Color Discrimination - 3AFC")
        self.clock = pygame.time.Clock()

        # Calculate stimulus positions (triangular arrangement)
        self._calculate_positions()

        # State
        self.running = True
        self.current_trial = 0

    def _default_config(self) -> dict:
        """Return default configuration."""
        return {
            'display': {
                'width_px': 1920,
                'height_px': 1080,
                'background_rgb': [0.5, 0.5, 0.5],
                'pixels_per_degree': 50
            },
            'stimuli': {
                'num_stimuli': 3,
                'stimulus_size_deg': 2.0,
                'stimulus_spacing_deg': 5.0
            },
            'timing': {
                'fixation_ms': 500,
                'stimulus_duration_ms': 200,
                'response_timeout_ms': 5000,
                'iti_ms': 500,
                'feedback_ms': 200
            }
        }

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

    def _rgb_float_to_int(self, rgb: np.ndarray) -> Tuple[int, int, int]:
        """Convert RGB from [0,1] float to [0,255] int."""
        rgb = np.asarray(rgb).flatten()
        rgb = np.clip(rgb, 0.0, 1.0)
        return tuple(int(c * 255) for c in rgb)

    def draw_fixation(self):
        """Draw fixation cross at screen center."""
        self.screen.fill(self.background_color)
        
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        size = 20
        thickness = 3
        
        # Draw cross in black (or could be white depending on background)
        cross_color = (0, 0, 0) if sum(self.background_color) > 384 else (255, 255, 255)
        
        pygame.draw.line(self.screen, cross_color, 
                        (center_x - size, center_y), 
                        (center_x + size, center_y), thickness)
        pygame.draw.line(self.screen, cross_color, 
                        (center_x, center_y - size), 
                        (center_x, center_y + size), thickness)
        
        pygame.display.flip()

    def draw_stimuli(self, stimuli_rgb: List[np.ndarray]):
        """
        Draw the three colored stimuli on screen.
        
        Args:
            stimuli_rgb: List of 3 RGB arrays, each in [0,1] range
        """
        self.screen.fill(self.background_color)

        for pos, rgb in zip(self.stimulus_positions, stimuli_rgb):
            color = self._rgb_float_to_int(rgb)
            pygame.draw.circle(self.screen, color, pos, self.stimulus_size_px)

        pygame.display.flip()

    def draw_blank(self):
        """Show blank screen (background only)."""
        self.screen.fill(self.background_color)
        pygame.display.flip()

    def draw_feedback(self, correct: bool):
        """
        Show feedback after response.
        
        Args:
            correct: Whether the response was correct
        """
        self.screen.fill(self.background_color)
        
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        
        if correct:
            # Green checkmark or circle
            pygame.draw.circle(self.screen, (0, 200, 0), (center_x, center_y), 30, 5)
        else:
            # Red X
            size = 25
            pygame.draw.line(self.screen, (200, 0, 0), 
                           (center_x - size, center_y - size),
                           (center_x + size, center_y + size), 5)
            pygame.draw.line(self.screen, (200, 0, 0), 
                           (center_x - size, center_y + size),
                           (center_x + size, center_y - size), 5)
        
        pygame.display.flip()

    def wait_for_response(self, timeout_ms: int = None) -> Tuple[Optional[int], Optional[float]]:
        """
        Wait for participant response.
        
        Args:
            timeout_ms: Maximum wait time in milliseconds
            
        Returns:
            Tuple of (response_index, response_time) or (None, None) if timeout/quit
        """
        if timeout_ms is None:
            timeout_ms = self.response_timeout_ms
            
        start_time = time.time()
        start_ticks = pygame.time.get_ticks()

        while pygame.time.get_ticks() - start_ticks < timeout_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None, None
                    
                elif event.type == pygame.KEYDOWN:
                    response = None
                    
                    # Number keys 1, 2, 3
                    if event.key == pygame.K_UP:
                        response = 0  # Top stimulus
                    elif event.key == pygame.K_LEFT:
                        response = 1  # Bottom-left
                    elif event.key == pygame.K_RIGHT:
                        response = 2  # Bottom-right
                    
                    # Numpad keys
                    elif event.key == pygame.K_KP1:
                        response = 0
                    elif event.key == pygame.K_KP2:
                        response = 1
                    elif event.key == pygame.K_KP3:
                        response = 2
                    
                    # Quit key
                    elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                        self.running = False
                        return None, None

                    if response is not None:
                        response_time = time.time() - start_time
                        return response, response_time

            # Small delay to prevent CPU spinning
            pygame.time.wait(1)

        # Timeout
        return None, None

    def run_trial(
        self, 
        stimuli_rgb: List[np.ndarray] = None,
        correct_index: int = None,
        show_feedback: bool = True
    ) -> Tuple[Optional[int], bool, Optional[float]]:
        """
        Run a single trial.
        
        Args:
            stimuli_rgb: List of 3 RGB arrays [0,1]. If None, generates random stimuli.
            correct_index: Index of the comparison stimulus (0, 1, or 2). 
                          Required if stimuli_rgb is provided.
            show_feedback: Whether to show correctness feedback
            
        Returns:
            Tuple of (response_index, is_correct, response_time)
        """
        self.current_trial += 1

        # Generate stimuli if not provided
        if stimuli_rgb is None:
            stimuli_rgb, correct_index = self._generate_random_stimuli()
        elif correct_index is None:
            raise ValueError("correct_index must be provided when stimuli_rgb is given")

        # Ensure stimuli are numpy arrays
        stimuli_rgb = [np.asarray(s) for s in stimuli_rgb]

        # === Trial sequence ===
        
        # 1. Fixation
        self.draw_fixation()
        pygame.time.wait(self.fixation_ms)
        
        # Check for quit during fixation
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_q):
                self.running = False
                return None, False, None

        # 2. Stimulus presentation
        self.draw_stimuli(stimuli_rgb)
        pygame.time.wait(self.stimulus_duration_ms)

        # 3. Blank screen during response window
        self.draw_blank()
        
        # 4. Wait for response
        response, response_time = self.wait_for_response()

        if response is None:
            # Timeout or quit
            is_correct = False
        else:
            is_correct = (response == correct_index)

        # 5. Feedback (optional)
        if show_feedback and response is not None:
            self.draw_feedback(is_correct)
            pygame.time.wait(self.feedback_ms)

        # 6. ITI
        self.draw_blank()
        pygame.time.wait(self.iti_ms)

        return response, is_correct, response_time

    def run_trial_from_pipeline(
        self,
        ref_rgb: np.ndarray,
        comp_rgb: np.ndarray,
        comparison_index: int = None,
        show_feedback: bool = True
    ) -> Tuple[Optional[int], bool, Optional[float]]:
        """
        Run a trial with stimuli from the AEPsych/color transformation pipeline.
        
        Args:
            ref_rgb: Reference stimulus RGB [0,1]
            comp_rgb: Comparison stimulus RGB [0,1]
            comparison_index: Position of comparison (0, 1, 2). Random if None.
            show_feedback: Whether to show feedback
            
        Returns:
            Tuple of (response_index, is_correct, response_time)
        """
        # Randomize comparison position if not specified
        if comparison_index is None:
            comparison_index = random.randint(0, 2)

        # Build stimulus list: two references + one comparison
        stimuli_rgb = []
        for i in range(3):
            if i == comparison_index:
                stimuli_rgb.append(np.asarray(comp_rgb))
            else:
                stimuli_rgb.append(np.asarray(ref_rgb))

        return self.run_trial(stimuli_rgb, comparison_index, show_feedback)

    def _generate_random_stimuli(self) -> Tuple[List[np.ndarray], int]:
        """
        Generate random stimuli for demo/testing mode.
        
        Returns:
            Tuple of (list of 3 RGB arrays, index of odd-one-out)
        """
        # Random reference color (avoid extremes for visibility)
        ref_rgb = np.random.uniform(0.3, 0.7, size=3)

        # Random comparison with perturbation
        perturbation = np.random.uniform(0.1, 0.3)
        direction = np.random.choice([-1, 1], size=3)
        comp_rgb = ref_rgb + perturbation * direction * np.random.choice([1, 0, 0], size=3, p=[0.6, 0.2, 0.2])
        comp_rgb = np.clip(comp_rgb, 0.0, 1.0)

        # Random position for comparison
        correct_index = random.randint(0, 2)

        # Build stimulus list
        stimuli = []
        for i in range(3):
            if i == correct_index:
                stimuli.append(comp_rgb)
            else:
                stimuli.append(ref_rgb.copy())

        return stimuli, correct_index

    def run_demo(self, num_trials: int = 10):
        """
        Run demo mode with random stimuli.
        
        Args:
            num_trials: Number of trials to run
        """
        print("=" * 50)
        print("3AFC Color Discrimination - Demo Mode")
        print("=" * 50)
        print("\nInstructions:")
        print("  - Three colored circles will appear")
        print("  - Two are identical, one is different")
        print("  - Press 1, 2, or 3 to select the different one")
        print("  - Press Q or ESC to quit")
        print("\nPositions: 1=Top, 2=Bottom-left, 3=Bottom-right")
        print("-" * 50)

        results = []

        for trial in range(1, num_trials + 1):
            if not self.running:
                break

            print(f"\nTrial {trial}/{num_trials}...")

            response, correct, rt = self.run_trial()

            if response is None:
                if not self.running:
                    print("Experiment terminated by user")
                else:
                    print("No response (timeout)")
                continue

            results.append({
                'trial': trial,
                'response': response,
                'correct': correct,
                'response_time': rt
            })

            status = "✓ Correct" if correct else "✗ Incorrect"
            print(f"{status} - RT: {rt:.3f}s")

        # Summary
        if results:
            n_correct = sum(1 for r in results if r['correct'])
            accuracy = 100 * n_correct / len(results)
            mean_rt = np.mean([r['response_time'] for r in results if r['response_time']])
            
            print("\n" + "=" * 50)
            print(f"Summary: {n_correct}/{len(results)} correct ({accuracy:.1f}%)")
            print(f"Mean RT: {mean_rt:.3f}s")

        return results

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()


def main():
    """Run demo of stimulus presentation."""
    stim = None
    try:
        stim = StimulusPresentation(fullscreen=False)
        stim.run_demo(num_trials=5)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if stim is not None:
            stim.cleanup()


if __name__ == "__main__":
    main()