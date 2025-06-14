#!/usr/bin/env python3
"""
Generate synthetic test data for Fisherman's Net algorithm.

This script creates:
1. A synthetic curved volume resembling a scroll
2. Corresponding fiber predictions
3. Orientation fields

Output can be used to test the warping algorithm performance.
"""

import os
import sys
import argparse
import logging
from pathlib import Path

import numpy as np
import tifffile
import matplotlib.pyplot as plt

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fishermans_net.utils.io import save_volume


def generate_curved_scroll(
    shape=(128, 128, 64),
    curvature=0.15,
    layers=5,
    noise=0.05,
    output_dir=None
):
    """
    Generate a synthetic curved scroll volume with fiber structure.
    
    Args:
        shape: (height, width, depth)
        curvature: How curved the scroll is (0-1)
        layers: Number of layers in the scroll
        noise: Amount of noise to add (0-1)
        output_dir: Directory to save outputs
    
    Returns:
        Dictionary with 'volume', 'fibers', and 'orientations'
    """
    print(f"Generating synthetic curved scroll with shape {shape}...")
    
    # Create coordinate grid
    y, x, z = np.meshgrid(
        np.linspace(0, 1, shape[0]),
        np.linspace(0, 1, shape[1]),
        np.linspace(0, 1, shape[2]),
        indexing='ij'
    )
    
    # Create curved surface
    curve = np.sin(x * np.pi) * curvature
    
    # Create layered structure (scroll sheets)
    frequency = layers * 2 * np.pi
    layered = np.sin((y + curve) * frequency)
    
    # Add some radial structure
    center_x, center_y = 0.5, 0.5
    radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    radial = np.cos(radius * 15) * 0.2
    
    # Combine for final volume
    volume = layered * (1.0 - z*0.5) + radial
    
    # Add noise
    if noise > 0:
        volume += np.random.normal(0, noise, volume.shape)
    
    # Normalize to 0-1 range
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Create fiber volume (gradient magnitude of layered structure)
    dy, dx, dz = np.gradient(layered)
    gradient_mag = np.sqrt(dy**2 + dx**2 + dz**2)
    fiber_volume = gradient_mag / gradient_mag.max()
    
    # Add some noise to fiber volume
    if noise > 0:
        fiber_volume += np.random.normal(0, noise*0.5, fiber_volume.shape)
        fiber_volume = np.clip(fiber_volume, 0, 1)
    
    # Create fiber orientations (tangent to layered structure)
    # Handle the 3-component orientation vector field
    orientations = np.zeros((*shape, 3))
    
    # Compute gradients for each slice
    for z_idx in range(shape[2]):
        dy_slice, dx_slice = np.gradient(layered[:, :, z_idx])
        # Make orientation perpendicular to gradient
        orientations[:, :, z_idx, 0] = -dx_slice  # x component
        orientations[:, :, z_idx, 1] = dy_slice   # y component
        orientations[:, :, z_idx, 2] = 0.1        # small z component
    
    # Normalize orientation vectors
    norm = np.sqrt(np.sum(orientations**2, axis=3, keepdims=True))
    orientations = orientations / (norm + 1e-10)
    
    # Add some noise to orientations
    if noise > 0:
        orientations += np.random.normal(0, noise*0.3, orientations.shape)
        # Renormalize
        norm = np.sqrt(np.sum(orientations**2, axis=3, keepdims=True))
        orientations = orientations / (norm + 1e-10)
    
    # Save files if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save volume
        volume_path = output_dir / "test_volume.tif"
        tifffile.imwrite(str(volume_path), (volume * 65535).astype(np.uint16))
        print(f"Saved volume to {volume_path}")
        
        # Save fiber predictions
        fiber_path = output_dir / "test_fibers.tif"
        tifffile.imwrite(str(fiber_path), (fiber_volume * 65535).astype(np.uint16))
        print(f"Saved fiber predictions to {fiber_path}")
        
        # Save orientations
        orient_path = output_dir / "test_orientations.npy"
        np.save(str(orient_path), orientations)
        print(f"Saved orientations to {orient_path}")
        
        # Save visualization
        vis_path = output_dir / "test_visualization.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Volume center slice
        z_mid = shape[2] // 2
        axes[0, 0].imshow(volume[:, :, z_mid], cmap='gray')
        axes[0, 0].set_title("Volume (Center Slice)")
        
        # Fiber center slice
        axes[0, 1].imshow(fiber_volume[:, :, z_mid], cmap='hot')
        axes[0, 1].set_title("Fiber Predictions")
        
        # Volume side view
        y_mid = shape[0] // 2
        axes[1, 0].imshow(volume[y_mid, :, :], cmap='gray')
        axes[1, 0].set_title("Volume (Side View)")
        
        # Orientation visualization
        # Show orientation as RGB color-coded direction
        orient_vis = np.abs(orientations[:, :, z_mid, :3])
        axes[1, 1].imshow(orient_vis)
        axes[1, 1].set_title("Orientation (RGB=XYZ)")
        
        plt.tight_layout()
        fig.savefig(vis_path)
        print(f"Saved visualization to {vis_path}")
    
    return {
        'volume': volume,
        'fibers': fiber_volume,
        'orientations': orientations
    }


def generate_flat_scroll_with_fibers(
    shape=(128, 128, 64),
    fiber_count=10,
    fiber_width=5,
    noise=0.05,
    output_dir=None
):
    """
    Generate a synthetic flat scroll with visible fibers.
    Useful for testing the fiber tracing algorithm.
    
    Args:
        shape: (height, width, depth)
        fiber_count: Number of fibers to generate
        fiber_width: Width of each fiber in pixels
        noise: Amount of noise to add (0-1)
        output_dir: Directory to save outputs
    """
    print(f"Generating flat scroll with {fiber_count} fibers...")
    
    # Create empty volume
    volume = np.zeros(shape)
    fiber_volume = np.zeros(shape)
    
    # Create layers (flat sheets)
    y, x, z = np.meshgrid(
        np.linspace(0, 1, shape[0]),
        np.linspace(0, 1, shape[1]),
        np.linspace(0, 1, shape[2]),
        indexing='ij'
    )
    
    # Create 5 flat layers
    layers = np.sin(y * 10 * np.pi)
    volume += layers * 0.5
    
    # Add fibers
    for i in range(fiber_count):
        # Random fiber parameters
        amplitude = 0.1 + 0.2 * np.random.rand()  # How wavy the fiber is
        frequency = 1 + 3 * np.random.rand()      # Frequency of waves
        phase = 2 * np.pi * np.random.rand()      # Random phase
        y_pos = 0.1 + 0.8 * np.random.rand()      # Y position (0-1)
        
        # Create fiber curve
        fiber_y = shape[0] * y_pos + amplitude * shape[0] * np.sin(frequency * np.pi * x.reshape(-1) + phase)
        fiber_y = fiber_y.reshape(1, shape[1], 1).repeat(shape[2], axis=2)
        
        # Convert to indices
        y_idx = np.arange(shape[0])
        
        # Create fiber mask
        for z_idx in range(shape[2]):
            for x_idx in range(shape[1]):
                # Calculate distance to fiber curve
                dist = np.abs(y_idx - fiber_y[0, x_idx, z_idx])
                # Apply fiber intensity using Gaussian profile
                fiber_intensity = np.exp(-dist**2 / (2 * fiber_width**2))
                # Add to volume and fiber volume
                volume[:, x_idx, z_idx] += fiber_intensity * 0.3
                fiber_volume[:, x_idx, z_idx] = np.maximum(
                    fiber_volume[:, x_idx, z_idx], 
                    fiber_intensity
                )
    
    # Add noise
    if noise > 0:
        volume += np.random.normal(0, noise, volume.shape)
        fiber_volume += np.random.normal(0, noise*0.5, fiber_volume.shape)
        fiber_volume = np.clip(fiber_volume, 0, 1)
    
    # Normalize volume
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Create orientation field (mostly horizontal fibers)
    orientations = np.zeros((*shape, 3))
    orientations[:, :, :, 1] = 1.0  # Primarily y-direction
    
    # Add some variation to orientations
    if noise > 0:
        orientations += np.random.normal(0, noise*0.5, orientations.shape)
        # Renormalize
        norm = np.sqrt(np.sum(orientations**2, axis=3, keepdims=True))
        orientations = orientations / (norm + 1e-10)
    
    # Save files if output directory specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save volume
        volume_path = output_dir / "test_flat_volume.tif"
        tifffile.imwrite(str(volume_path), (volume * 65535).astype(np.uint16))
        
        # Save fiber predictions
        fiber_path = output_dir / "test_flat_fibers.tif"
        tifffile.imwrite(str(fiber_path), (fiber_volume * 65535).astype(np.uint16))
        
        # Save orientations
        orient_path = output_dir / "test_flat_orientations.npy"
        np.save(str(orient_path), orientations)
        
        # Save visualization
        vis_path = output_dir / "test_flat_visualization.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        z_mid = shape[2] // 2
        axes[0, 0].imshow(volume[:, :, z_mid], cmap='gray')
        axes[0, 0].set_title("Volume with Fibers")
        
        axes[0, 1].imshow(fiber_volume[:, :, z_mid], cmap='hot')
        axes[0, 1].set_title("Fiber Predictions")
        
        # Show different slices
        axes[1, 0].imshow(volume[:, :, shape[2]//4], cmap='gray')
        axes[1, 0].set_title("Volume (Different Slice)")
        
        # Show 3D slice
        x_mid = shape[1] // 2
        axes[1, 1].imshow(volume[:, x_mid, :], cmap='gray')
        axes[1, 1].set_title("Side View (YZ plane)")
        
        plt.tight_layout()
        fig.savefig(vis_path)
        
        print(f"Saved flat scroll test data to {output_dir}")
    
    return {
        'volume': volume,
        'fibers': fiber_volume,
        'orientations': orientations
    }


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic test data for Fisherman's Net algorithm")
    parser.add_argument("--output", "-o", default="./test_data", help="Output directory")
    parser.add_argument("--shape", nargs=3, type=int, default=[128, 128, 64], help="Volume shape (height width depth)")
    parser.add_argument("--curvature", type=float, default=0.15, help="Curvature factor (0-1)")
    parser.add_argument("--layers", type=int, default=5, help="Number of layers in scroll")
    parser.add_argument("--noise", type=float, default=0.05, help="Noise factor (0-1)")
    parser.add_argument("--flat", action="store_true", help="Generate flat scroll with visible fibers")
    parser.add_argument("--fibers", type=int, default=10, help="Number of fibers in flat scroll")
    
    args = parser.parse_args()
    
    # Convert shape to tuple
    shape = tuple(args.shape)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate data
    if args.flat:
        generate_flat_scroll_with_fibers(
            shape=shape,
            fiber_count=args.fibers,
            noise=args.noise,
            output_dir=output_dir
        )
    else:
        generate_curved_scroll(
            shape=shape,
            curvature=args.curvature,
            layers=args.layers,
            noise=args.noise,
            output_dir=output_dir
        )
    
    print(f"Test data generation complete. Files saved to {output_dir}")
    print("\nTo test the warping algorithm, run:")
    print(f"python scripts/warp_volume.py --input {output_dir}/test_volume.tif --fibers {output_dir}/test_fibers.tif --orientations {output_dir}/test_orientations.npy --output {output_dir}/warped_volume.tif --save-report --iterations 30")


if __name__ == "__main__":
    main()
