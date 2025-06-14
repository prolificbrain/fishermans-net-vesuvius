#!/usr/bin/env python3
"""
Benchmark script for the Fisherman's Net warping algorithm.

This script evaluates:
- Performance at different scales
- Memory usage
- Effect of algorithm parameters
- Comparison between nearest neighbor and trilinear interpolation
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

import numpy as np
import mlx.core as mx

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from fishermans_net.core.warping import FishermansNetWarper, WarpingConfig
from fishermans_net.utils.io import save_volume
from fishermans_net.mlx_ops.interpolation import trilinear_interpolate, warp_volume
from fishermans_net.mlx_ops.filters import gaussian_blur_3d, separable_gaussian_blur_3d


def create_benchmark_volume(size, fiber_density=0.1):
    """Create synthetic volume with fibers for benchmarking"""
    print(f"Creating synthetic volume of size {size}...")
    
    volume = mx.random.uniform(shape=size)
    
    # Create fiber structures (simple vertical fibers)
    fiber_volume = mx.zeros(size)
    fiber_orientations = mx.zeros((*size, 3))
    
    y, x, z = mx.meshgrid(
        mx.arange(size[0]),
        mx.arange(size[1]),
        mx.arange(size[2]),
        indexing='ij'
    )
    
    # Create fiber patterns
    for i in range(int(size[0] * fiber_density)):
        center_x = np.random.randint(0, size[1])
        radius = 3 + np.random.rand() * 5
        
        # Calculate distance to fiber center
        dist = mx.sqrt((x - center_x)**2)
        
        # Create fiber intensity 
        intensity = mx.exp(-dist**2 / (2 * radius**2))
        fiber_volume = mx.maximum(fiber_volume, intensity)
    
    # Create mainly vertical orientations for fibers
    fiber_orientations = mx.zeros((*size, 3))
    
    # Create random orientation vectors (normalized)
    fiber_orientations = mx.random.normal(shape=(*size, 3))
    
    # Make them primarily point in the y-direction (axis 1) for cleaner visualization
    # In MLX 0.26.1, we need to use a workaround instead of .at[].set()
    # Create a ones array for the y component
    y_component = mx.ones((*size,))
    
    # Reconstruct the orientation array
    x_component = fiber_orientations[..., 0]
    z_component = fiber_orientations[..., 2]
    fiber_orientations = mx.stack([x_component, y_component, z_component], axis=-1)
    
    # Normalize the orientation vectors
    norms = mx.sqrt(mx.sum(fiber_orientations**2, axis=-1, keepdims=True))
    fiber_orientations = fiber_orientations / (norms + 1e-10)
    
    return volume, fiber_volume, fiber_orientations


def benchmark_ops():
    """Benchmark individual operations"""
    print("\n🔍 BENCHMARKING INDIVIDUAL OPERATIONS")
    print("=" * 60)
    
    sizes = [(64, 64, 32), (128, 128, 64), (256, 256, 128)]
    
    print(f"{'Operation':<30} {'Size':<15} {'Time (ms)':<10} {'Memory (MB)':<10}")
    print("-" * 60)
    
    for size in sizes:
        # Create test volume
        volume = mx.random.uniform(shape=size)
        displacements = mx.random.normal(shape=(*size, 3)) * 2.0
        
        # Benchmark interpolation
        coords = mx.random.uniform(shape=(*size, 3)) * mx.array([size[0]-1, size[1]-1, size[2]-1])
        
        # Nearest neighbor interpolation
        start = time.time()
        _ = volume[
            mx.clip(mx.round(coords[..., 0]).astype(mx.int32), 0, size[0]-1),
            mx.clip(mx.round(coords[..., 1]).astype(mx.int32), 0, size[1]-1),
            mx.clip(mx.round(coords[..., 2]).astype(mx.int32), 0, size[2]-1)
        ]
        nn_time = (time.time() - start) * 1000
        
        # Trilinear interpolation
        start = time.time()
        _ = trilinear_interpolate(volume, coords)
        trilinear_time = (time.time() - start) * 1000
        
        # Warping
        start = time.time()
        _ = warp_volume(volume, displacements, mode="nearest")
        warp_nn_time = (time.time() - start) * 1000
        
        start = time.time()
        _ = warp_volume(volume, displacements, mode="bilinear")
        warp_trilinear_time = (time.time() - start) * 1000
        
        # Gaussian blur
        start = time.time()
        _ = gaussian_blur_3d(volume, sigma=1.0, kernel_size=5)
        gauss_time = (time.time() - start) * 1000
        
        start = time.time()
        _ = separable_gaussian_blur_3d(volume, sigma=1.0)
        sep_gauss_time = (time.time() - start) * 1000
        
        # Print results
        vol_size_mb = volume.nbytes / (1024 * 1024)
        print(f"Nearest Neighbor Interp      {str(size):<15} {nn_time:<10.2f} {vol_size_mb:<10.2f}")
        print(f"Trilinear Interpolation      {str(size):<15} {trilinear_time:<10.2f} {vol_size_mb:<10.2f}")
        print(f"Warping (Nearest)            {str(size):<15} {warp_nn_time:<10.2f} {vol_size_mb * 2:<10.2f}")
        print(f"Warping (Trilinear)          {str(size):<15} {warp_trilinear_time:<10.2f} {vol_size_mb * 2:<10.2f}")
        print(f"Gaussian Blur (Full)         {str(size):<15} {gauss_time:<10.2f} {vol_size_mb * 3:<10.2f}")
        print(f"Gaussian Blur (Separable)    {str(size):<15} {sep_gauss_time:<10.2f} {vol_size_mb * 3:<10.2f}")
        print("-" * 60)


def benchmark_full_pipeline():
    """Benchmark full warping pipeline"""
    print("\n BENCHMARKING FULL WARPING PIPELINE")
    print("=" * 60)
    
    sizes = [(64, 64, 32), (128, 128, 64), (256, 256, 128)]
    iterations = [10, 10, 5]  # Fewer iterations for larger volumes
    
    print(f"{'Size':<16} {'Total Time (s)':<16} {'Per Iteration (s)':<16} {'Memory (MB)':<10}")
    print('-' * 60)
    
    for i, size in enumerate(sizes):
        # Create test volume
        volume, fiber_volume, fiber_orientations = create_benchmark_volume(size)
        
        # Configure warping
        config = WarpingConfig(
            elasticity=0.8,
            viscosity=0.2,
            fiber_strength=1.5,
            convergence_threshold=1e-4
        )
        warper = FishermansNetWarper(config)
        
        # Run warping
        start = time.time()
        result = warper.warp_volume(
            volume=volume,
            fiber_volume=fiber_volume,
            fiber_orientations=fiber_orientations,
            num_iterations=iterations[i]
        )
        elapsed = time.time() - start
        
        # Calculate memory usage (approximate)
        memory_mb = (volume.nbytes * 3 + fiber_volume.nbytes + fiber_orientations.nbytes) / (1024 * 1024)
        
        # Print results
        print(f"{str(size):<15} {elapsed:<15.2f} {elapsed/iterations[i]:<15.2f} {memory_mb:<10.2f}")
    
    print("-" * 60)


def benchmark_parameters():
    """Benchmark different parameter configurations"""
    print("\n⚙️ BENCHMARKING PARAMETER CONFIGURATIONS")
    print("=" * 60)
    
    # Fixed medium size
    size = (128, 128, 64)
    iterations = 10
    
    # Create test volume
    volume, fiber_volume, fiber_orientations = create_benchmark_volume(size)
    
    # Parameter combinations to test
    params = [
        {"elasticity": 0.5, "fiber_strength": 1.0, "smoothing_sigma": 0.5},
        {"elasticity": 0.8, "fiber_strength": 2.0, "smoothing_sigma": 1.0},
        {"elasticity": 0.3, "fiber_strength": 3.0, "smoothing_sigma": 1.5},
        {"elasticity": 0.9, "fiber_strength": 1.5, "smoothing_sigma": 2.0},
    ]
    
    print(f"{'Parameters':<45} {'Time (s)':<10} {'Final Flatness':<15} {'Final Strain':<15}")
    print("-" * 85)
    
    for param in params:
        config = WarpingConfig(
            elasticity=param["elasticity"],
            fiber_strength=param["fiber_strength"],
            smoothing_sigma=param["smoothing_sigma"],
            max_iterations=iterations,
            convergence_threshold=1e-4
        )
        
        # Run warping
        warper = FishermansNetWarper(config)
        
        start = time.time()
        result = warper.warp_volume(
            volume=volume,
            fiber_volume=fiber_volume,
            fiber_orientations=fiber_orientations,
            mask=None,
            num_iterations=iterations,
            checkpoint_interval=iterations
        )
        elapsed = time.time() - start
        
        # Get metrics
        final_flatness = result['metrics']['flatness'][-1] if 'flatness' in result['metrics'] else 'N/A' 
        final_strain = result['metrics']['strain'][-1] if 'strain' in result['metrics'] else 'N/A'
        
        # Print results
        param_str = f"E:{param['elasticity']:.1f}, F:{param['fiber_strength']:.1f}, S:{param['smoothing_sigma']:.1f}"
        print(f"{param_str:<45} {elapsed:<10.2f} {final_flatness:<15.4f} {final_strain:<15.4f}")
    
    print("-" * 85)


def main():
    parser = argparse.ArgumentParser(description="Benchmark Fisherman's Net warping algorithm")
    parser.add_argument("--ops", action="store_true", help="Benchmark individual operations")
    parser.add_argument("--pipeline", action="store_true", help="Benchmark full pipeline")
    parser.add_argument("--params", action="store_true", help="Benchmark parameter configurations")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🏆 FISHERMAN'S NET WARPING ALGORITHM BENCHMARK 🏆")
    print("=" * 60)
    
    # Run requested benchmarks
    if args.ops or args.all:
        benchmark_ops()
    
    if args.pipeline or args.all:
        benchmark_full_pipeline()
    
    if args.params or args.all:
        benchmark_parameters()
    
    # If no specific benchmark requested, run full pipeline
    if not (args.ops or args.pipeline or args.params or args.all):
        benchmark_full_pipeline()


if __name__ == "__main__":
    main()
