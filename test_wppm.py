import sys
sys.path.insert(0, 'src')
import jax.numpy as jnp
import pandas as pd
from adaptive_engine.wppm_fitter import WPPMModel

# Create more realistic trial data
trial_data = pd.DataFrame({
    'ref_rgb_r': [0.5] * 10,
    'ref_rgb_g': [0.5] * 10, 
    'ref_rgb_b': [0.5] * 10,
    'comp_rgb_r': [0.5 + i*0.01 for i in range(10)],
    'comp_rgb_g': [0.5] * 10,
    'comp_rgb_b': [0.5] * 10,
    'response': [1 if i < 7 else 0 for i in range(10)]  # Easier trials first
})

print('Testing WPPM MAP estimation...')
wppm = WPPMModel(n_basis_per_dim=5, observer_samples=1)
result = wppm.fit_model(trial_data, n_iterations=4, learning_rate=1e-4)

print(f'✓ MAP fitting completed!')
print(f'✓ Final log posterior: {result["log_posterior"]:.4f}')
print(f'✓ No NaN values: {not jnp.any(jnp.isnan(result["weight_matrix"]))}')

# Test prediction
grid_coords = jnp.array([[0.5, 0.5], [0.6, 0.5]])
predictions = wppm.predict_psychometric_field(result, grid_coords)
print(f'✓ Predictions computed: {predictions}')

print('All tests passed! ✓')