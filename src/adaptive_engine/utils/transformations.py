#!/usr/bin/env python3
"""
Color space transformation utilities for the psychophysical experiment.
Handles conversion between RGB, model space, and other color representations.
"""

import numpy as np
from typing import Tuple, List, Union

class ColorTransformations:
    """Utilities for color space transformations."""

    def __init__(self):
        """Initialize transformation matrices and parameters."""
        # These would typically be loaded from calibration data
        # For now, using identity transformations for testing
        self.rgb_to_model_matrix = np.eye(2)  # 2D model space
        self.model_to_rgb_matrix = np.eye(2)
        self.model_space_bounds = [-1.0, 1.0]

    def rgb_to_model_space(self, rgb: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert RGB coordinates to model space coordinates.

        Args:
            rgb: RGB values as [r, g, b] in range [0, 1]

        Returns:
            Model space coordinates as numpy array
        """
        rgb = np.array(rgb)

        # Ensure RGB is in valid range
        rgb = np.clip(rgb, 0.0, 1.0)

        # For isoluminant plane, we work in RG space (relative to gray point)
        # This is a simplified transformation for testing
        gray_point = np.array([0.5, 0.5, 0.5])

        # Project onto isoluminant plane and transform to model space
        rg_chromaticity = rgb[:2] - gray_point[:2]  # Just use R and G for simplicity

        # Apply affine transformation to square bounds
        # This would normally use the calibrated transformation matrix
        model_coords = rg_chromaticity * 2.0  # Scale to roughly fill [-1, 1] range
        model_coords = np.clip(model_coords, self.model_space_bounds[0], self.model_space_bounds[1])

        return model_coords

    def model_space_to_rgb(self, model_coords: Union[List[float], np.ndarray]) -> np.ndarray:
        """
        Convert model space coordinates back to RGB.

        Args:
            model_coords: Model space coordinates as [x, y]

        Returns:
            RGB values as numpy array in range [0, 1]
        """
        model_coords = np.array(model_coords)

        # Ensure coordinates are in valid range
        model_coords = np.clip(model_coords, self.model_space_bounds[0], self.model_space_bounds[1])

        # Reverse the transformation
        rg_chromaticity = model_coords / 2.0
        gray_point = np.array([0.5, 0.5, 0.5])

        # Reconstruct RGB (simplified)
        rgb = gray_point.copy()
        rgb[:2] += rg_chromaticity

        # Ensure RGB is in valid range
        rgb = np.clip(rgb, 0.0, 1.0)

        return rgb

    def generate_random_model_space_point(self) -> np.ndarray:
        """
        Generate a random point in model space within bounds.

        Returns:
            Random model space coordinates as numpy array
        """
        x = np.random.uniform(self.model_space_bounds[0], self.model_space_bounds[1])
        y = np.random.uniform(self.model_space_bounds[0], self.model_space_bounds[1])
        return np.array([x, y])

    def generate_perturbed_stimulus(self, reference_rgb: Union[List[float], np.ndarray],
                                   perturbation_size: float = 0.1) -> np.ndarray:
        """
        Generate a comparison stimulus by perturbing a reference stimulus.

        Args:
            reference_rgb: Reference stimulus RGB values
            perturbation_size: Size of perturbation in model space

        Returns:
            Comparison stimulus RGB values
        """
        # Convert reference to model space
        ref_model = self.rgb_to_model_space(reference_rgb)

        # Generate random perturbation
        angle = np.random.uniform(0, 2 * np.pi)
        perturbation = np.array([
            perturbation_size * np.cos(angle),
            perturbation_size * np.sin(angle)
        ])

        # Apply perturbation
        comp_model = ref_model + perturbation

        # Ensure result stays within bounds
        comp_model = np.clip(comp_model, self.model_space_bounds[0], self.model_space_bounds[1])

        # Convert back to RGB
        comp_rgb = self.model_space_to_rgb(comp_model)

        return comp_rgb

    def create_trial_stimuli(self, reference_rgb: Union[List[float], np.ndarray] = None,
                           perturbation_size: float = 0.1) -> Tuple[List[List[float]], int]:
        """
        Create stimuli for a trial: two identical references and one comparison.

        Args:
            reference_rgb: RGB values for reference stimuli (random if None)
            perturbation_size: Size of perturbation for comparison stimulus

        Returns:
            Tuple of (list of RGB stimuli, index of odd stimulus)
        """
        if reference_rgb is None:
            # Generate random reference
            ref_model = self.generate_random_model_space_point()
            reference_rgb = self.model_space_to_rgb(ref_model)

        # Choose which stimulus will be the odd one out (0, 1, or 2)
        odd_index = np.random.randint(0, 3)

        # Create stimuli list
        stimuli_rgb = []
        for i in range(3):
            if i == odd_index:
                # Generate comparison stimulus
                comp_rgb = self.generate_perturbed_stimulus(reference_rgb, perturbation_size)
                stimuli_rgb.append(comp_rgb.tolist())
            else:
                # Use reference stimulus
                stimuli_rgb.append(reference_rgb.tolist() if isinstance(reference_rgb, np.ndarray) else reference_rgb)

        return stimuli_rgb, odd_index

    def validate_rgb_values(self, rgb: Union[List[float], np.ndarray]) -> bool:
        """
        Validate that RGB values are in valid range [0, 1].

        Args:
            rgb: RGB values to validate

        Returns:
            True if valid, False otherwise
        """
        rgb = np.array(rgb)
        return np.all((rgb >= 0.0) & (rgb <= 1.0))

    def apply_gamma_correction(self, rgb: Union[List[float], np.ndarray],
                             gamma: float = 2.2) -> np.ndarray:
        """
        Apply gamma correction to RGB values.

        Args:
            rgb: Linear RGB values
            gamma: Gamma value

        Returns:
            Gamma-corrected RGB values
        """
        rgb = np.array(rgb)
        return np.power(rgb, 1.0 / gamma)

# Test functions
def test_transformations():
    """Test the color transformation utilities."""
    transform = ColorTransformations()

    print("Testing color transformations...")

    # Test basic conversion
    test_rgb = [0.5, 0.5, 0.5]
    model_coords = transform.rgb_to_model_space(test_rgb)
    rgb_back = transform.model_space_to_rgb(model_coords)

    print(f"Original RGB: {test_rgb}")
    print(f"Model coords: {model_coords}")
    print(f"RGB back: {rgb_back}")

    # Test trial stimuli generation
    stimuli, odd_index = transform.create_trial_stimuli()
    print(f"Generated stimuli: {stimuli}")
    print(f"Odd stimulus index: {odd_index}")

    # Test perturbation
    ref_rgb = [0.5, 0.5, 0.5]
    comp_rgb = transform.generate_perturbed_stimulus(ref_rgb)
    print(f"Reference: {ref_rgb}")
    print(f"Perturbed: {comp_rgb}")

    print("Transformation tests completed!")

if __name__ == "__main__":
    test_transformations()