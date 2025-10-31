#!/usr/bin/env python3
"""
Simple stimulus presentation for testing the psychophysical experiment.
Displays three colored circles in triangular arrangement for 3AFC oddity task.
"""

import pygame
import json
import random
import time
import sys
import os
from pathlib import Path

class SimpleStimulusPresentation:
    def __init__(self, config_path=None):
        """Initialize the stimulus presentation system."""
        pygame.init()

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "experiment_params.json"

        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Display settings
        display_config = self.config['display']
        self.screen_width = display_config['width_px']
        self.screen_height = display_config['height_px']
        self.background_color = [int(c * 255) for c in display_config['background_rgb']]

        # Stimulus settings
        stimulus_config = self.config['stimuli']
        self.num_stimuli = stimulus_config['num_stimuli']
        self.stimulus_size_px = int(stimulus_config['stimulus_size_deg'] * 50)  # Rough conversion
        self.stimulus_spacing_px = int(stimulus_config['stimulus_spacing_deg'] * 50)

        # Timing settings
        timing_config = self.config['timing']
        self.iti_ms = timing_config['iti_ms']
        self.stimulus_duration_ms = timing_config['stimulus_duration_ms']
        self.response_timeout_ms = timing_config['response_timeout_ms']

        # Initialize display
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Color Discrimination Test")
        self.clock = pygame.time.Clock()

        # Calculate stimulus positions (triangular arrangement)
        center_x = self.screen_width // 2
        center_y = self.screen_height // 2
        radius = self.stimulus_spacing_px

        # Position stimuli in triangle
        self.stimulus_positions = [
            (center_x, center_y - radius),  # Top
            (center_x - radius * 0.866, center_y + radius * 0.5),  # Bottom left
            (center_x + radius * 0.866, center_y + radius * 0.5),  # Bottom right
        ]

        self.running = True
        self.current_trial = 0

    def generate_random_colors(self):
        """Generate three RGB colors where two are identical and one differs."""
        # Generate base color
        base_r = random.uniform(0.2, 0.8)
        base_g = random.uniform(0.2, 0.8)
        base_b = random.uniform(0.2, 0.8)

        # Choose which stimulus will be different (0, 1, or 2)
        odd_one_out = random.randint(0, 2)

        # Generate small perturbation for the different stimulus
        perturbation = random.uniform(0.1, 0.3)
        direction = random.choice([-1, 1])

        colors = []
        for i in range(3):
            if i == odd_one_out:
                # Perturb one color component
                color_component = random.choice(['r', 'g', 'b'])
                if color_component == 'r':
                    r = min(1.0, max(0.0, base_r + perturbation * direction))
                    g, b = base_g, base_b
                elif color_component == 'g':
                    g = min(1.0, max(0.0, base_g + perturbation * direction))
                    r, b = base_r, base_b
                else:  # 'b'
                    b = min(1.0, max(0.0, base_b + perturbation * direction))
                    r, g = base_r, base_g
            else:
                r, g, b = base_r, base_g, base_b

            colors.append((int(r * 255), int(g * 255), int(b * 255)))

        return colors, odd_one_out

    def draw_stimuli(self, colors):
        """Draw the three colored stimuli on screen."""
        self.screen.fill(self.background_color)

        for i, (pos, color) in enumerate(zip(self.stimulus_positions, colors)):
            pygame.draw.circle(self.screen, color, pos, self.stimulus_size_px)

        pygame.display.flip()

    def show_blank_screen(self):
        """Show blank screen (ITI or response period)."""
        self.screen.fill(self.background_color)
        pygame.display.flip()

    def run_trial(self):
        """Run a single trial."""
        self.current_trial += 1

        # Generate stimuli
        colors, correct_response = self.generate_random_colors()

        # Show blank screen (ITI)
        self.show_blank_screen()
        pygame.time.wait(self.iti_ms)

        # Show stimuli
        start_time = time.time()
        self.draw_stimuli(colors)
        stimulus_start = pygame.time.get_ticks()

        response = None
        response_time = None

        # Wait for response or timeout
        while pygame.time.get_ticks() - stimulus_start < self.stimulus_duration_ms + self.response_timeout_ms:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    return None, None, None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_1:
                        response = 0  # Top stimulus
                    elif event.key == pygame.K_2:
                        response = 1  # Bottom left
                    elif event.key == pygame.K_3:
                        response = 2  # Bottom right
                    elif event.key == pygame.K_q:
                        self.running = False
                        return None, None, None

                    if response is not None:
                        response_time = time.time() - start_time
                        break

            if response is not None:
                break

        # Show blank screen after response
        self.show_blank_screen()

        # Determine if response was correct
        correct = (response == correct_response) if response is not None else False

        return response, correct, response_time

    def run_experiment(self, num_trials=10):
        """Run the experiment for specified number of trials."""
        print("Starting simple stimulus presentation test...")
        print(f"Press keys 1, 2, or 3 to respond (corresponding to top, bottom-left, bottom-right stimuli)")
        print("Press Q to quit")
        print("-" * 50)

        results = []

        for trial in range(num_trials):
            if not self.running:
                break

            print(f"Trial {trial + 1}/{num_trials}")

            response, correct, response_time = self.run_trial()

            if response is None:
                print("Experiment terminated")
                break

            results.append({
                'trial': trial + 1,
                'response': response,
                'correct': correct,
                'response_time': response_time
            })

            # Provide feedback
            if correct:
                print(f"✓ Correct! RT: {response_time:.2f}s")
            else:
                print(f"✗ Incorrect. RT: {response_time:.2f}s")
            # Short pause between trials
            pygame.time.wait(500)

        # Save results
        self.save_results(results)
        print("\nExperiment completed!")
        return results

    def save_results(self, results):
        """Save trial results to CSV."""
        import csv

        results_dir = Path(__file__).parent.parent.parent / "data" / "P01" / "raw"
        results_dir.mkdir(parents=True, exist_ok=True)

        filename = results_dir / "P01_S01_simple_test.csv"

        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['trial', 'response', 'correct', 'response_time']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"Results saved to {filename}")

    def cleanup(self):
        """Clean up pygame resources."""
        pygame.quit()

def main():
    """Main function to run the simple stimulus presentation."""
    try:
        stimulus = SimpleStimulusPresentation()

        # Run a short test
        num_trials = 5
        results = stimulus.run_experiment(num_trials)

        # Print summary
        if results:
            correct_trials = sum(1 for r in results if r['correct'])
            print(f"\nSummary: {correct_trials}/{len(results)} trials correct ({correct_trials/len(results)*100:.1f}%)")

    except KeyboardInterrupt:
        print("\nExperiment interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if 'stimulus' in locals():
            stimulus.cleanup()

if __name__ == "__main__":
    main()