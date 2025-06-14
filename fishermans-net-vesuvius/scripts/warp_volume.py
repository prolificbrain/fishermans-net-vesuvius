#!/usr/bin/env python3
"""
Main script to warp Vesuvius scroll volumes using Fisherman's Net algorithm.

Usage:
    python warp_volume.py --input volume.tif --fibers fibers.tif --output warped.tif
"""

import argparse
import logging
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import tifffile

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from fishermans_net.core.warping import FishermansNetWarper, WarpingConfig
from fishermans_net.utils.io import load_volume, save_volume, load_fiber_data
from fishermans_net.utils.visualization import create_warping_report


def main():
    parser = argparse.ArgumentParser(
        description="Warp Vesuvius scroll volumes using Fisherman's Net algorithm"
    )
    
    # Required arguments
    parser.add_argument("--input", required=True, help="Input volume (TIF)")
    parser.add_argument("--fibers", required=True, help="Fiber predictions (TIF)")
    parser.add_argument("--output", required=True, help="Output warped volume (TIF)")
    
    # Optional arguments
    parser.add_argument("--orientations", help="Fiber orientations (NPY)")
    parser.add_argument("--mask", help="Valid region mask (TIF)")
    parser.add_argument("--iterations", type=int, default=100, 
                       help="Number of warping iterations")
    parser.add_argument("--checkpoint-interval", type=int, default=20,
                       help="Save checkpoints every N iterations")
    
    # Algorithm parameters
    parser.add_argument("--elasticity", type=float, default=0.8,
                       help="Elasticity coefficient (0-1)")
    parser.add_argument("--fiber-strength", type=float, default=2.0,
                       help="Fiber pulling strength")
    parser.add_argument("--num-fibers", type=int, default=20,
                       help="Number of critical fibers to trace")
    
    # Output options
    parser.add_argument("--save-deformation", action="store_true",
                       help="Save deformation field")
    parser.add_argument("--save-report", action="store_true",
                       help="Generate HTML report")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Load data
    logger.info(f"Loading volume from {args.input}")
    volume = load_volume(args.input)
    logger.info(f"Volume shape: {volume.shape}")
    
    logger.info(f"Loading fiber predictions from {args.fibers}")
    fiber_volume = load_volume(args.fibers)
    
    # Load or generate fiber orientations
    if args.orientations:
        logger.info(f"Loading fiber orientations from {args.orientations}")
        fiber_orientations = np.load(args.orientations)
        fiber_orientations = mx.array(fiber_orientations)
    else:
        logger.info("Generating synthetic fiber orientations")
        fiber_orientations = generate_fiber_orientations(fiber_volume)
    
    # Load mask if provided
    mask = None
    if args.mask:
        logger.info(f"Loading mask from {args.mask}")
        mask = load_volume(args.mask)
        mask = mask > 0.5  # Binarize
    
    # Configure warping
    config = WarpingConfig(
        elasticity=args.elasticity,
        fiber_strength=args.fiber_strength,
        num_critical_fibers=args.num_fibers
    )
    
    # Create warper
    warper = FishermansNetWarper(config)
    
    # Run warping
    logger.info("Starting warping process...")
    start_time = time.time()
    
    result = warper.warp_volume(
        volume=volume,
        fiber_volume=fiber_volume,
        fiber_orientations=fiber_orientations,
        mask=mask,
        num_iterations=args.iterations,
        checkpoint_interval=args.checkpoint_interval
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"Warping completed in {elapsed_time:.1f} seconds")
    
    # Save results
    logger.info(f"Saving warped volume to {args.output}")
    save_volume(args.output, result['warped_volume'])
    
    # Save deformation field if requested
    if args.save_deformation:
        deform_path = Path(args.output).with_suffix('.deformation.npy')
        logger.info(f"Saving deformation field to {deform_path}")
        np.save(deform_path, np.array(result['deformation_field']))
    
    # Generate report if requested
    if args.save_report:
        report_path = Path(args.output).with_suffix('.report.html')
        logger.info(f"Generating report at {report_path}")
        create_warping_report(
            result=result,
            output_path=report_path,
            input_path=args.input,
            elapsed_time=elapsed_time
        )
    
    # Print summary
    print("\n" + "="*50)
    print("WARPING SUMMARY")
    print("="*50)
    print(f"Input shape: {volume.shape}")
    print(f"Output shape: {result['warped_volume'].shape}")
    print(f"Critical fibers found: {len(result['critical_fibers'])}")
    print(f"Final flatness score: {result['metrics']['flatness'][-1]:.4f}")
    print(f"Total strain energy: {result['metrics']['strain'][-1]:.4f}")
    print(f"Processing time: {elapsed_time:.1f} seconds")
    print(f"Output saved to: {args.output}")
    print("="*50)


def generate_fiber_orientations(fiber_volume: mx.array) -> mx.array:
    """
    Generate synthetic fiber orientations from fiber volume.
    
    In production, these would come from your fiber detection model.
    """
    shape = (*fiber_volume.shape, 3)
    
    # Compute gradients to estimate local fiber direction
    dx = mx.diff(fiber_volume, axis=0, prepend=fiber_volume[0:1])
    dy = mx.diff(fiber_volume, axis=1, prepend=fiber_volume[:, 0:1])
    dz = mx.diff(fiber_volume, axis=2, prepend=fiber_volume[:, :, 0:1])
    
    # Stack to create orientation field
    orientations = mx.stack([dx, dy, dz], axis=-1)
    
    # Normalize
    magnitude = mx.sqrt(mx.sum(orientations**2, axis=-1, keepdims=True))
    orientations = orientations / (magnitude + 1e-6)
    
    # Add some noise for realism
    noise = mx.random.normal(shape) * 0.1
    orientations = orientations + noise
    
    # Renormalize
    magnitude = mx.sqrt(mx.sum(orientations**2, axis=-1, keepdims=True))
    orientations = orientations / (magnitude + 1e-6)
    
    return orientations


if __name__ == "__main__":
    main()
