#!/usr/bin/env python3
"""
Color space transformations for the psychophysical experiment.

Based on Hong et al. (2025) Appendix 1: Colorimetric Transformations.

This module provides transformations between:
- Linear RGB space (monitor's native space)
- Model space (2D space used by WPPM, bounded by [-1, 1]²)
- DKL space (Derrington-Krauskopf-Lennie isoluminant color space)

The transformations are derived from the monitor's gamut boundary in the
isoluminant plane, as described in Appendix 1.1-1.3.
"""

import numpy as np
from typing import Tuple, List, Optional


class ColorTransformations:
    """
    Color space transformations between RGB, DKL, and model space.
    
    All transformations are affine and operate on the isoluminant plane
    with the adapting background at [0.5, 0.5, 0.5] in linear RGB.
    
    Transformation matrices from Table S2 of Hong et al. (2025).
    """
    
    # Transformation matrix from RGB to model space (Table S2)
    # [w_dim1, w_dim2, 1]^T = M_RGB_TO_W @ [R, G, B]^T
    M_RGB_TO_W = np.array([
        [1.556, -1.364, -0.192],
        [-0.444, -1.364, 1.808],
        [0.444, 1.364, 0.192]
    ])
    
    # Transformation matrix from DKL to model space (Table S2)
    # [w_dim1, w_dim2, 1]^T = M_DKL_TO_W @ [DKL_L-M, DKL_S, 0]^T
    M_DKL_TO_W = np.array([
        [6.724, 0.213, 0.000],
        [0.076, 1.221, 0.000],
        [0.000, 0.000, 1.000]
    ])
    
    # Corner vertices from Table S1 (for validation)
    CORNER_VERTICES = {
        1: {'dkl': (-0.123, -0.812, 0), 'lms': (0.145, 0.147, 0.016), 
            'rgb': (0.000, 0.733, 0.000), 'model': (-1, -1)},
        2: {'dkl': (0.175, -0.830, 0), 'lms': (0.164, 0.111, 0.014), 
            'rgb': (1.000, 0.407, 0.000), 'model': (1, -1)},
        3: {'dkl': (-0.175, 0.830, 0), 'lms': (0.142, 0.154, 0.152), 
            'rgb': (0.000, 0.593, 1.000), 'model': (-1, 1)},
        4: {'dkl': (0.123, 0.812, 0), 'lms': (0.160, 0.117, 0.150), 
            'rgb': (1.000, 0.267, 1.000), 'model': (1, 1)},
    }
    
    # Background RGB (adapting point)
    BACKGROUND_RGB = np.array([0.5, 0.5, 0.5])
    
    def __init__(self):
        """Initialize transformations and compute inverse matrices."""
        # Compute inverse of RGB to model space transformation
        self.M_W_TO_RGB = np.linalg.inv(self.M_RGB_TO_W)
        
        # Validate transformations against known corner vertices
        self._validate_transformations()
    
    def _validate_transformations(self):
        """Validate that transformations produce correct corner vertices."""
        for corner_id, vertex in self.CORNER_VERTICES.items():
            rgb = np.array(vertex['rgb'])
            expected_model = np.array(vertex['model'])
            
            # Test RGB -> Model
            computed_model = self.rgb_to_model_space(rgb)
            
            if not np.allclose(computed_model, expected_model, atol=0.01):
                print(f"Warning: Corner {corner_id} validation failed!")
                print(f"  Expected: {expected_model}, Got: {computed_model}")
    
    def rgb_to_model_space(self, rgb: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to 2D model space coordinates.
        
        Uses the affine transformation from Appendix 1.3 (Equation S7).
        
        Args:
            rgb: Linear RGB values [R, G, B] in range [0, 1]
            
        Returns:
            Model space coordinates [w_dim1, w_dim2] in range [-1, 1]
        """
        rgb = np.asarray(rgb).flatten()
        
        # Apply transformation: [w1, w2, 1]^T = M @ [R, G, B]^T
        homogeneous = self.M_RGB_TO_W @ rgb
        
        # Extract 2D coordinates (divide by homogeneous coordinate for safety)
        w_dim1 = homogeneous[0] / homogeneous[2]
        w_dim2 = homogeneous[1] / homogeneous[2]
        
        return np.array([w_dim1, w_dim2])
    
    def model_space_to_rgb(self, model_coords: np.ndarray) -> np.ndarray:
        """
        Convert 2D model space coordinates to linear RGB.
        
        Inverts the affine transformation from Appendix 1.3.
        
        Args:
            model_coords: Model space coordinates [w_dim1, w_dim2] in [-1, 1]²
            
        Returns:
            Linear RGB values [R, G, B] in range [0, 1]
        """
        model_coords = np.asarray(model_coords).flatten()
        
        # Create homogeneous coordinates [w1, w2, 1]
        homogeneous = np.array([model_coords[0], model_coords[1], 1.0])
        
        # Apply inverse transformation
        rgb = self.M_W_TO_RGB @ homogeneous
        
        # Clip to valid RGB range
        rgb = np.clip(rgb, 0.0, 1.0)
        
        return rgb
    
    def dkl_to_model_space(self, dkl: np.ndarray) -> np.ndarray:
        """
        Convert DKL coordinates to 2D model space.
        
        Uses the affine transformation from Appendix 1.2 (Equation S5).
        
        Args:
            dkl: DKL coordinates [L-M, S, Luminance] (Luminance should be 0 for isoluminant)
            
        Returns:
            Model space coordinates [w_dim1, w_dim2]
        """
        dkl = np.asarray(dkl).flatten()
        
        # Ensure we're on isoluminant plane
        dkl_homogeneous = np.array([dkl[0], dkl[1], 0.0])
        
        # Apply transformation
        homogeneous = self.M_DKL_TO_W @ dkl_homogeneous
        
        return np.array([homogeneous[0], homogeneous[1]])
    
    def create_comparison_stimulus(
        self, 
        reference_rgb: np.ndarray, 
        delta_model: np.ndarray
    ) -> np.ndarray:
        """
        Create comparison stimulus by adding offset in model space.
        
        Args:
            reference_rgb: Reference stimulus RGB values
            delta_model: Offset in model space [delta_w1, delta_w2]
            
        Returns:
            Comparison stimulus RGB values
        """
        # Convert reference to model space
        ref_model = self.rgb_to_model_space(reference_rgb)
        
        # Add offset
        comp_model = ref_model + np.asarray(delta_model)
        
        # Clip to model space bounds
        comp_model = np.clip(comp_model, -1.0, 1.0)
        
        # Convert back to RGB
        return self.model_space_to_rgb(comp_model)
    
    def create_trial_stimuli(
        self, 
        reference_rgb: np.ndarray,
        delta_magnitude: float = 0.1,
        delta_direction: Optional[float] = None
    ) -> Tuple[List[np.ndarray], int]:
        """
        Create 3AFC trial stimuli (two reference copies + one comparison).
        
        Args:
            reference_rgb: Reference stimulus RGB values
            delta_magnitude: Magnitude of offset in model space
            delta_direction: Direction of offset in radians (random if None)
            
        Returns:
            Tuple of (list of 3 RGB stimuli, index of comparison stimulus)
        """
        if delta_direction is None:
            delta_direction = np.random.uniform(0, 2 * np.pi)
        
        # Create offset vector in model space
        delta_model = delta_magnitude * np.array([
            np.cos(delta_direction),
            np.sin(delta_direction)
        ])
        
        # Create comparison stimulus
        comparison_rgb = self.create_comparison_stimulus(reference_rgb, delta_model)
        
        # Random position for comparison (0, 1, or 2)
        comparison_index = np.random.randint(0, 3)
        
        # Create stimulus list
        stimuli = []
        for i in range(3):
            if i == comparison_index:
                stimuli.append(comparison_rgb.copy())
            else:
                stimuli.append(reference_rgb.copy())
        
        return stimuli, comparison_index
    
    def get_chromatic_direction(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        """
        Get chromatic direction between two stimuli in model space.
        
        Args:
            rgb1: First RGB stimulus
            rgb2: Second RGB stimulus
            
        Returns:
            Direction in radians
        """
        model1 = self.rgb_to_model_space(rgb1)
        model2 = self.rgb_to_model_space(rgb2)
        
        delta = model2 - model1
        return np.arctan2(delta[1], delta[0])
    
    def get_chromatic_distance(self, rgb1: np.ndarray, rgb2: np.ndarray) -> float:
        """
        Get Euclidean distance between two stimuli in model space.
        
        Args:
            rgb1: First RGB stimulus
            rgb2: Second RGB stimulus
            
        Returns:
            Distance in model space units
        """
        model1 = self.rgb_to_model_space(rgb1)
        model2 = self.rgb_to_model_space(rgb2)
        
        return np.linalg.norm(model2 - model1)
    
    def is_within_gamut(self, model_coords: np.ndarray, tolerance: float = 0.01) -> bool:
        """
        Check if model space coordinates are within the monitor's gamut.
        
        The gamut in model space is approximately [-1, 1]² but the actual
        boundary depends on the monitor's RGB gamut.
        
        Args:
            model_coords: Model space coordinates
            tolerance: Tolerance for boundary check
            
        Returns:
            True if within gamut
        """
        model_coords = np.asarray(model_coords)
        
        # First check model space bounds
        if np.any(np.abs(model_coords) > 1.0 + tolerance):
            return False
        
        # Then check if resulting RGB is valid
        rgb = self.model_space_to_rgb(model_coords)
        return np.all(rgb >= -tolerance) and np.all(rgb <= 1.0 + tolerance)
    
    def sample_random_reference(self, margin: float = 0.2) -> np.ndarray:
        """
        Sample a random reference stimulus within the gamut.
        
        Args:
            margin: Margin from gamut boundary in model space units
            
        Returns:
            Random RGB stimulus
        """
        # Sample from center region of model space
        bounds = 1.0 - margin
        model_coords = np.random.uniform(-bounds, bounds, size=2)
        
        return self.model_space_to_rgb(model_coords)
    
    @staticmethod
    def linear_rgb_to_srgb(linear_rgb: np.ndarray) -> np.ndarray:
        """
        Convert linear RGB to sRGB (gamma-corrected) for display.
        
        Args:
            linear_rgb: Linear RGB values in [0, 1]
            
        Returns:
            sRGB values in [0, 1]
        """
        linear_rgb = np.asarray(linear_rgb)
        
        # sRGB gamma correction
        srgb = np.where(
            linear_rgb <= 0.0031308,
            12.92 * linear_rgb,
            1.055 * np.power(linear_rgb, 1/2.4) - 0.055
        )
        
        return np.clip(srgb, 0.0, 1.0)
    
    @staticmethod
    def srgb_to_linear_rgb(srgb: np.ndarray) -> np.ndarray:
        """
        Convert sRGB (gamma-corrected) to linear RGB.
        
        Args:
            srgb: sRGB values in [0, 1]
            
        Returns:
            Linear RGB values in [0, 1]
        """
        srgb = np.asarray(srgb)
        
        # Inverse sRGB gamma
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        
        return np.clip(linear, 0.0, 1.0)
    
    def rgb_to_display(self, linear_rgb: np.ndarray, as_int: bool = True) -> np.ndarray:
        """
        Convert linear RGB to display-ready values.
        
        Args:
            linear_rgb: Linear RGB in [0, 1]
            as_int: If True, return 8-bit integers [0, 255]
            
        Returns:
            Display-ready RGB values
        """
        srgb = self.linear_rgb_to_srgb(linear_rgb)
        
        if as_int:
            return (srgb * 255).astype(np.uint8)
        return srgb


def test_transformations():
    """Test the color transformations against known vertices."""
    print("Testing Color Transformations")
    print("=" * 50)
    
    transform = ColorTransformations()
    
    print("\nValidating corner vertices (Table S1):")
    print("-" * 50)
    
    for corner_id, vertex in transform.CORNER_VERTICES.items():
        rgb = np.array(vertex['rgb'])
        expected_model = np.array(vertex['model'])
        
        # Test RGB -> Model
        computed_model = transform.rgb_to_model_space(rgb)
        
        # Test Model -> RGB (round trip)
        recovered_rgb = transform.model_space_to_rgb(expected_model)
        
        print(f"\nCorner {corner_id}:")
        print(f"  RGB: {rgb}")
        print(f"  Expected model: {expected_model}")
        print(f"  Computed model: {computed_model}")
        print(f"  Model error: {np.linalg.norm(computed_model - expected_model):.6f}")
        print(f"  Recovered RGB: {recovered_rgb}")
        print(f"  RGB error: {np.linalg.norm(recovered_rgb - rgb):.6f}")
    
    print("\n" + "=" * 50)
    print("Testing stimulus creation:")
    print("-" * 50)
    
    # Test with background color
    ref_rgb = transform.BACKGROUND_RGB.copy()
    ref_model = transform.rgb_to_model_space(ref_rgb)
    print(f"\nBackground RGB: {ref_rgb}")
    print(f"Background in model space: {ref_model}")
    
    # Create trial stimuli
    stimuli, comp_idx = transform.create_trial_stimuli(ref_rgb, delta_magnitude=0.2)
    print(f"\nTrial stimuli (comparison at index {comp_idx}):")
    for i, stim in enumerate(stimuli):
        model = transform.rgb_to_model_space(stim)
        marker = " <-- comparison" if i == comp_idx else ""
        print(f"  Stimulus {i}: RGB={stim}, Model={model}{marker}")
    
    # Test chromatic distance (find a reference stimulus index)
    ref_idx = 1 if comp_idx != 1 else 0
    distance = transform.get_chromatic_distance(stimuli[ref_idx], stimuli[comp_idx])
    print(f"\nChromatic distance (ref to comp): {distance:.4f}")
    
    print("\n" + "=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    test_transformations()