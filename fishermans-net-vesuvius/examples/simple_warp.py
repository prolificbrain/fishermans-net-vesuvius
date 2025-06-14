#!/usr/bin/env python3
"""
Simple example demonstrating the Fisherman's Net volume warping algorithm
on synthetic data.

This creates a curved synthetic "scroll" and warps it using fiber guidance.
"""

import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import mlx.core as mx
import matplotlib.pyplot as plt
import tifffile

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fishermans_net.core.warping import FishermansNetWarper, WarpingConfig
from fishermans_net.utils.io import save_volume
from fishermans_net.utils.visualization import visualize_comparison, visualize_metrics


def create_synthetic_scroll(
    shape=(128, 128, 64), 
    curl_factor=0.2, 
    noise_level=0.05
):
    """
    Create a synthetic curved scroll with fibers.
    
    Args:
        shape: Volume shape as (height, width, depth)
        curl_factor: How curved the scroll is
        noise_level: Amount of noise to add
        
    Returns:
        tuple of (volume, fiber_volume, fiber_orientations)
    """
    print(f"Creating synthetic scroll with shape {shape}...")
    
    # Create coordinate grid
    y, x, z = np.meshgrid(
        np.linspace(0, 1, shape[0]),
        np.linspace(0, 1, shape[1]),
        np.linspace(0, 1, shape[2]),
        indexing='ij'
    )
    
    # Create curved surface
    curve = np.sin(x * np.pi * 2) * curl_factor
    
    # Layer patterns (curved sheets)
    layers = np.sin((y + curve) * 10 * np.pi)
    
    # Add some radial structure
    center_x, center_y = 0.5, 0.5
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    radial = np.cos(radius * 15) * 0.3
    
    # Combine for final volume
    volume = layers * (1.0 - z*0.5) + radial
    
    # Add noise
    volume += np.random.normal(0, noise_level, volume.shape)
    
    # Create fiber directions (tangent to curved surface)
    # In this case, fibers run primarily along the x direction
    dx = np.ones_like(volume) * 0.5
    dy = np.gradient(curve, axis=0)[None, :, :] * 2  # Follows curve gradient
    dz = np.ones_like(volume) * 0.2
    
    # Normalize
    magnitude = np.sqrt(dx**2 + dy**2 + dz**2)
    dx = dx / magnitude
    dy = dy / magnitude
    dz = dz / magnitude
    
    # Stack to create orientation field
    orientations = np.stack([dx, dy, dz], axis=-1)
    
    # Create fiber volume (higher intensity where structured)
    fiber_volume = np.abs(np.gradient(layers, axis=0)) + 0.2
    fiber_volume = (fiber_volume - fiber_volume.min()) / (fiber_volume.max() - fiber_volume.min())
    
    # Convert to MLX arrays
    volume_mx = mx.array(volume.astype(np.float32))
    fiber_volume_mx = mx.array(fiber_volume.astype(np.float32))
    orientations_mx = mx.array(orientations.astype(np.float32))
    
    return volume_mx, fiber_volume_mx, orientations_mx


def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    # Create synthetic data
    volume, fiber_volume, fiber_orientations = create_synthetic_scroll()
    
    # Save the synthetic data
    input_path = output_dir / "synthetic_volume.tif"
    fibers_path = output_dir / "synthetic_fibers.tif"
    
    save_volume(input_path, volume)
    save_volume(fibers_path, fiber_volume)
    
    print(f"Saved synthetic data to {output_dir}")
    
    # Configure warping
    config = WarpingConfig(
        elasticity=0.8,
        fiber_strength=2.5,
        smoothing_sigma=1.0,
        num_critical_fibers=15,
        max_iterations=30,
        convergence_threshold=1e-4
    )
    
    # Create warper
    warper = FishermansNetWarper(config)
    
    # Run warping
    print("Starting warping process...")
    start_time = time.time()
    
    result = warper.warp_volume(
        volume=volume,
        fiber_volume=fiber_volume,
        fiber_orientations=fiber_orientations,
        mask=None,
        num_iterations=50,
        checkpoint_interval=10
    )
    
    elapsed_time = time.time() - start_time
    print(f"Warping completed in {elapsed_time:.1f} seconds")
    
    # Save results
    output_path = output_dir / "warped_volume.tif"
    save_volume(output_path, result['warped_volume'])
    
    # Plot results
    print("Creating visualizations...")
    
    # Compare before and after
    fig = visualize_comparison(volume, result['warped_volume'])
    fig.savefig(output_dir / "comparison.png")
    
    # Plot metrics
    fig = visualize_metrics(result['metrics'])
    fig.savefig(output_dir / "metrics.png")
    
    print(f"\nResults saved to {output_dir}")
    print(f"Warped volume: {output_path}")
    print(f"Comparison: {output_dir / 'comparison.png'}")
    print(f"Metrics: {output_dir / 'metrics.png'}")


if __name__ == "__main__":
    main()
