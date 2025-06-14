#!/usr/bin/env python3
"""
Test script for the pure NumPy implementation of Fisherman's Net warping.
This version is more stable and focuses on the core algorithm.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import tifffile
from fishermans_net_numpy import FishermansNetWarperNumPy, WarpingConfig
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to locally downloaded Vesuvius data
LOCAL_DATA_DIR = Path(os.path.abspath(__file__)).parent / "VesuviusDataDownload/Scroll1/segments"


def load_scroll_segment(segment_id="20230518012543", layers=(0, 30)):
    """Load a scroll segment from the locally downloaded data."""
    logger.info(f"Loading scroll segment {segment_id}...")
    
    segment_path = LOCAL_DATA_DIR / str(segment_id) / "layers"
    
    if not segment_path.exists():
        available = [d for d in os.listdir(LOCAL_DATA_DIR) if (LOCAL_DATA_DIR / d).is_dir()]
        raise FileNotFoundError(f"Segment {segment_id} not found. Available: {available}")
    
    # Find all TIFF files
    layer_files = sorted(glob.glob(str(segment_path / "*.tif")))
    
    if not layer_files:
        raise FileNotFoundError(f"No TIFF files found in {segment_path}")
    
    logger.info(f"Found {len(layer_files)} layer files")
    
    # Load specified layers
    z_start, z_end = layers
    z_end = min(z_end, len(layer_files))
    selected_files = layer_files[z_start:z_end]
    
    logger.info(f"Loading layers {z_start} to {z_end-1} (total: {len(selected_files)})...")
    
    # Load data
    volume_slices = []
    for layer_file in selected_files:
        slice_data = tifffile.imread(layer_file)
        volume_slices.append(slice_data)
    
    volume = np.stack(volume_slices, axis=0).astype(np.float32)
    
    logger.info(f"Loaded volume with shape: {volume.shape}")
    logger.info(f"Value range: [{np.min(volume):.1f}, {np.max(volume):.1f}]")
    
    return volume


def create_realistic_fiber_data(volume):
    """
    Create more realistic synthetic fiber data based on scroll structure.
    This simulates what real fiber detection would find.
    """
    logger.info("Creating realistic synthetic fiber data...")
    
    depth, height, width = volume.shape
    
    # Normalize volume
    volume_norm = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-10)
    
    # Find papyrus layers using intensity thresholding
    # Papyrus typically appears as medium-to-high intensity regions
    papyrus_threshold = np.percentile(volume_norm, 70)
    papyrus_mask = volume_norm > papyrus_threshold
    
    # Create fiber volume based on papyrus edges (where fibers are most visible)
    fiber_volume = np.zeros_like(volume_norm)
    
    # Find edges in each direction (where papyrus layers meet air/other materials)
    for axis in range(3):
        edges = np.abs(np.diff(volume_norm, axis=axis))
        
        # Pad to maintain shape
        if axis == 0:
            edges = np.pad(edges, ((0, 1), (0, 0), (0, 0)), mode='edge')
        elif axis == 1:
            edges = np.pad(edges, ((0, 0), (0, 1), (0, 0)), mode='edge')
        else:
            edges = np.pad(edges, ((0, 0), (0, 0), (0, 1)), mode='edge')
        
        # Add edges to fiber volume
        fiber_volume += edges
    
    # Enhance fiber volume where we have papyrus
    fiber_volume = fiber_volume * papyrus_mask
    
    # Normalize fiber volume
    if np.max(fiber_volume) > 0:
        fiber_volume = fiber_volume / np.max(fiber_volume)
    
    # Create fiber orientations
    # In scrolls, fibers tend to follow the papyrus layers (mostly horizontal)
    fiber_orientations = np.zeros((depth, height, width, 3))
    
    # Compute local gradients to estimate fiber directions
    grad_z, grad_y, grad_x = np.gradient(volume_norm)
    
    # Fiber orientations perpendicular to gradients (along the layers)
    # This is a simplification - real fibers would be more complex
    fiber_orientations[..., 0] = -grad_y  # z-component
    fiber_orientations[..., 1] = grad_x   # y-component  
    fiber_orientations[..., 2] = -grad_z  # x-component
    
    # Normalize orientation vectors
    norms = np.linalg.norm(fiber_orientations, axis=-1, keepdims=True)
    fiber_orientations = np.where(norms > 1e-6, 
                                 fiber_orientations / norms, 
                                 fiber_orientations)
    
    # Zero out orientations where there are no fibers
    fiber_mask = fiber_volume > 0.1
    fiber_orientations = fiber_orientations * fiber_mask[..., np.newaxis]
    
    num_fiber_points = np.sum(fiber_volume > 0.1)
    logger.info(f"Created synthetic fiber data with {num_fiber_points} fiber points")
    logger.info(f"Fiber volume range: [{np.min(fiber_volume):.3f}, {np.max(fiber_volume):.3f}]")
    
    return fiber_volume, fiber_orientations


def run_fishermans_net_warping(volume, fiber_volume, fiber_orientations):
    """Run the Fisherman's Net warping algorithm."""
    logger.info("Running Fisherman's Net warping algorithm...")
    
    # Create configuration optimized for scroll warping
    config = WarpingConfig(
        elasticity=0.3,           # Lower elasticity for more deformation
        viscosity=0.1,            # Lower viscosity for faster convergence
        fiber_strength=3.0,       # Higher fiber strength for more pulling force
        smoothing_sigma=1.0,      # Less smoothing to preserve detail
        max_deformation=25.0,     # Allow larger deformations
        num_critical_fibers=80,   # More fibers for better coverage
        step_size=0.3,            # Larger steps for visible changes
        convergence_threshold=0.01,  # Less strict convergence
        min_fiber_strength=0.02   # Lower threshold to find more fibers
    )
    
    # Initialize warper
    warper = FishermansNetWarperNumPy(config)
    
    # Run warping with more iterations to see the effect
    result = warper.warp_volume(
        volume=volume,
        fiber_volume=fiber_volume,
        fiber_orientations=fiber_orientations,
        num_iterations=100  # More iterations for better results
    )
    
    logger.info("Warping complete!")
    return result


def visualize_results(original, warped, deformation_field, output_dir):
    """Create comprehensive visualizations of the warping results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize volumes for visualization
    orig_norm = (original - np.min(original)) / (np.max(original) - np.min(original))
    warp_norm = (warped - np.min(warped)) / (np.max(warped) - np.min(warped))
    
    # Choose middle slices
    mid_z = original.shape[0] // 2
    mid_y = original.shape[1] // 2
    mid_x = original.shape[2] // 2
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Z-slice comparison
    axes[0, 0].imshow(orig_norm[mid_z], cmap='gray')
    axes[0, 0].set_title(f'Original Z-slice {mid_z}')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(warp_norm[mid_z], cmap='gray')
    axes[1, 0].set_title(f'Warped Z-slice {mid_z}')
    axes[1, 0].axis('off')
    
    # Y-slice comparison
    axes[0, 1].imshow(orig_norm[:, mid_y, :], cmap='gray', aspect='auto')
    axes[0, 1].set_title(f'Original Y-slice {mid_y}')
    axes[0, 1].axis('off')
    
    axes[1, 1].imshow(warp_norm[:, mid_y, :], cmap='gray', aspect='auto')
    axes[1, 1].set_title(f'Warped Y-slice {mid_y}')
    axes[1, 1].axis('off')
    
    # X-slice comparison
    axes[0, 2].imshow(orig_norm[:, :, mid_x], cmap='gray', aspect='auto')
    axes[0, 2].set_title(f'Original X-slice {mid_x}')
    axes[0, 2].axis('off')
    
    axes[1, 2].imshow(warp_norm[:, :, mid_x], cmap='gray', aspect='auto')
    axes[1, 2].set_title(f'Warped X-slice {mid_x}')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'volume_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Deformation field visualization
    plt.figure(figsize=(12, 8))
    
    # Show deformation magnitude
    deform_magnitude = np.linalg.norm(deformation_field, axis=-1)
    
    plt.subplot(2, 2, 1)
    plt.imshow(deform_magnitude[mid_z], cmap='hot')
    plt.title(f'Deformation Magnitude Z-slice {mid_z}')
    plt.colorbar()
    
    plt.subplot(2, 2, 2)
    plt.imshow(deform_magnitude[:, mid_y, :], cmap='hot', aspect='auto')
    plt.title(f'Deformation Magnitude Y-slice {mid_y}')
    plt.colorbar()
    
    # Show deformation vectors
    plt.subplot(2, 2, 3)
    y_coords, x_coords = np.mgrid[0:deformation_field.shape[1]:10, 0:deformation_field.shape[2]:10]
    u = deformation_field[mid_z, ::10, ::10, 1]
    v = deformation_field[mid_z, ::10, ::10, 2]
    plt.imshow(orig_norm[mid_z], cmap='gray', alpha=0.7)
    plt.quiver(x_coords, y_coords, v, u, color='red', alpha=0.8, scale=50)
    plt.title('Deformation Vectors (Z-slice)')
    
    plt.subplot(2, 2, 4)
    # Show some statistics
    plt.text(0.1, 0.8, f'Max deformation: {np.max(deform_magnitude):.2f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.7, f'Mean deformation: {np.mean(deform_magnitude):.2f}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.6, f'Volume shape: {original.shape}', transform=plt.gca().transAxes)
    plt.text(0.1, 0.5, f'Non-zero deformations: {np.sum(deform_magnitude > 0.1)}', transform=plt.gca().transAxes)
    plt.axis('off')
    plt.title('Deformation Statistics')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'deformation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Main function to run the complete pipeline."""
    output_dir = Path("./numpy_warping_results")
    output_dir.mkdir(exist_ok=True)
    
    # List available segments
    if not LOCAL_DATA_DIR.exists():
        logger.error(f"Data directory {LOCAL_DATA_DIR} does not exist!")
        logger.error("Please download scroll data first using the download scripts.")
        return
    
    segments = [d for d in os.listdir(LOCAL_DATA_DIR) 
               if (LOCAL_DATA_DIR / d).is_dir() and (LOCAL_DATA_DIR / d / "layers").exists()]
    
    if not segments:
        logger.error("No segments found!")
        return
    
    logger.info(f"Found {len(segments)} segments: {segments}")
    segment_id = segments[0]  # Use first available segment
    
    try:
        # Load scroll data
        volume = load_scroll_segment(segment_id=segment_id, layers=(0, 25))
        
        # Create synthetic fiber data
        fiber_volume, fiber_orientations = create_realistic_fiber_data(volume)
        
        # Run warping
        result = run_fishermans_net_warping(volume, fiber_volume, fiber_orientations)
        
        # Visualize results
        visualize_results(
            original=volume,
            warped=result['warped_volume'],
            deformation_field=result['deformation_field'],
            output_dir=str(output_dir)
        )
        
        # Print summary
        logger.info("=== WARPING SUMMARY ===")
        logger.info(f"Original volume shape: {volume.shape}")
        logger.info(f"Number of critical fibers found: {len(result['critical_fibers'])}")
        logger.info(f"Final flatness score: {result['metrics']['flatness'][-1]:.4f}")
        logger.info(f"Final strain energy: {result['metrics']['strain'][-1]:.4f}")
        logger.info(f"Results saved to: {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during processing: {e}")
        raise


if __name__ == "__main__":
    main()
