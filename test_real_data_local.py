#!/usr/bin/env python3
"""
Pure NumPy/SciPy implementation of Fisherman's Net warping for Vesuvius Challenge.
This version is more stable and focuses on the core algorithm without MLX complications.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob
from PIL import Image
import tifffile
from scipy import ndimage
from scipy.interpolate import RegularGridInterpolator
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path to locally downloaded Vesuvius data
LOCAL_DATA_DIR = Path(os.path.abspath(__file__)).parent / "VesuviusDataDownload/Scroll1/segments"

def load_local_scroll_data(segment_id="20230611145109", layers=None):
    """
    Load a scroll segment from the locally downloaded Vesuvius Challenge data.
    
    Args:
        segment_id: ID of the scroll segment to load (as string or integer)
        layers: Tuple of (start_layer, end_layer) to load, or None for all
        
    Returns:
        volume: 3D array of CT scan data
        segment_info: Dictionary with metadata about the segment
    """
    print(f"Loading local scroll segment {segment_id}...")
    
    # Default to loading a small z-range if none specified
    if layers is None:
        layers = (0, 30)  # Load first 30 layers as default
    
    # Convert segment_id to string if it's not already
    segment_id = str(segment_id)
    
    # Path to the segment layers
    segment_path = LOCAL_DATA_DIR / segment_id / "layers"
    
    if not segment_path.exists():
        print(f"Error: Segment path {segment_path} does not exist!")
        print(f"Available segments: {os.listdir(LOCAL_DATA_DIR) if LOCAL_DATA_DIR.exists() else 'None'}")
        raise FileNotFoundError(f"Segment {segment_id} not found in {LOCAL_DATA_DIR}")
    
    # Find all TIFF files in the layers directory
    layer_files = sorted(glob.glob(str(segment_path / "*.tif")))
    
    if not layer_files:
        print(f"Error: No TIFF files found in {segment_path}")
        raise FileNotFoundError(f"No layer files found for segment {segment_id}")
    
    print(f"Found {len(layer_files)} layer files")
    
    # Calculate the z_range based on layers
    z_start, z_end = layers
    z_end = min(z_end, len(layer_files))  # Make sure we don't exceed available layers
    
    # Get the actual layers within the specified range
    start_layer = max(0, z_start)
    num_layers = z_end - start_layer
    selected_files = layer_files[start_layer:z_end]
    
    print(f"Loading layers {start_layer} to {z_end-1} (total: {num_layers})...")
    
    # Load data slice by slice to avoid memory issues
    volume_slices = []
    for layer_file in selected_files:
        # Load a single z-slice using tifffile for better performance with large TIFFs
        slice_data = tifffile.imread(layer_file)
        volume_slices.append(slice_data)
    
    # Stack the slices to form a volume
    data = np.stack(volume_slices, axis=0)
    
    # Convert to MLX array
    mlx_data = mx.array(data)
    
    # Print some information about the data
    print(f"Loaded volume with shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Value range: [{np.min(data)}, {np.max(data)}]")
    
    # Create a metadata dictionary
    segment_info = {
        "segment_id": segment_id,
        "path": str(segment_path),
        "num_total_layers": len(layer_files),
        "loaded_layers": (start_layer, z_end)
    }
    
    return mlx_data, segment_info

def list_available_segments():
    """
    List all available segments in the local data directory.
    
    Returns:
        List of segment IDs
    """
    if not LOCAL_DATA_DIR.exists():
        print(f"Warning: Local data directory {LOCAL_DATA_DIR} does not exist.")
        return []
    
    # Get all directories in the segments folder
    segments = [d for d in os.listdir(LOCAL_DATA_DIR) 
               if (LOCAL_DATA_DIR / d).is_dir() and (LOCAL_DATA_DIR / d / "layers").exists()]
    
    return segments

def synthesize_fiber_data(volume_data, fiber_density=0.05, fiber_thickness=3):
    """
    Create synthetic fiber data since we don't have real fiber predictions.
    This is a temporary solution for testing the warping pipeline.
    
    Args:
        volume_data: Original CT volume data
        fiber_density: Density of synthetic fibers (0-1)
        fiber_thickness: Thickness of synthetic fibers
        
    Returns:
        fiber_volume: Synthetic fiber prediction volume
        fiber_orientations: Synthetic fiber orientation vectors
    """
    print("Creating synthetic fiber data...")
    
    # Extract shape
    depth, height, width = volume_data.shape
    
    # Create a binary fiber mask based on volume intensity and add some noise
    threshold = np.percentile(np.array(volume_data), 80)  # Use 80th percentile as threshold
    fiber_mask = mx.array(np.array(volume_data) > threshold).astype(mx.float32)
    
    # Add some noise to make it look more like real fiber predictions
    noise = mx.random.uniform(shape=fiber_mask.shape) < fiber_density
    fiber_mask = fiber_mask * noise
    
    # Simplified approach - just use the fiber mask directly with some smoothing
    fiber_volume = fiber_mask

    # Add some simple smoothing by averaging with neighbors
    # This is much simpler and faster than the complex morphological operation
    smoothed = mx.zeros_like(fiber_volume)
    for dz in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                # Create shifted version with padding
                if dz == 0 and dy == 0 and dx == 0:
                    smoothed += fiber_volume
                else:
                    # Simple boundary handling - just skip edge effects
                    z_start = max(0, -dz)
                    z_end = min(depth, depth - dz)
                    y_start = max(0, -dy)
                    y_end = min(height, height - dy)
                    x_start = max(0, -dx)
                    x_end = min(width, width - dx)

                    if z_end > z_start and y_end > y_start and x_end > x_start:
                        smoothed[z_start:z_end, y_start:y_end, x_start:x_end] += \
                            fiber_volume[z_start+dz:z_end+dz, y_start+dy:y_end+dy, x_start+dx:x_end+dx]

    fiber_volume = smoothed / 27.0  # Normalize by number of neighbors
    
    # Normalize fiber volume
    fiber_volume = mx.clip(fiber_volume, 0, 1)
    
    # Create synthetic fiber orientations (predominantly in x-y plane)
    # This is a simplification - real fibers would follow the scroll structure
    fiber_orientations = mx.random.uniform(shape=(depth, height, width, 3)) - 0.5
    
    # Make z component smaller to favor x-y plane orientations
    fiber_orientations = mx.array([
        fiber_orientations[:, :, :, 0],
        fiber_orientations[:, :, :, 1],
        0.1 * fiber_orientations[:, :, :, 2]
    ]).transpose(1, 2, 3, 0)
    
    # Normalize vectors
    norm = mx.sqrt(mx.sum(fiber_orientations**2, axis=-1, keepdims=True))
    fiber_orientations = mx.where(
        norm > 0,
        fiber_orientations / (norm + 1e-10),
        mx.zeros_like(fiber_orientations)
    )
    
    # Zero out orientations where there are no fibers
    fiber_orientations = fiber_orientations * mx.expand_dims(fiber_volume > 0.5, axis=-1)
    
    print(f"Created synthetic fiber data with {mx.sum(fiber_volume > 0.5)} fiber points")
    
    return fiber_volume, fiber_orientations

def run_warping_pipeline(volume, fiber_volume, fiber_orientations):
    """
    Run the Fisherman's Net warping pipeline on the provided data.

    Args:
        volume: Input CT volume
        fiber_volume: Fiber prediction volume
        fiber_orientations: Fiber orientation vectors

    Returns:
        result: Dictionary with warping results
    """
    print("Running Fisherman's Net warping pipeline...")

    # Create a warping configuration with default parameters
    config = WarpingConfig(
        elasticity=0.2,
        viscosity=0.8,
        fiber_strength=1.0,
        smoothing_sigma=1.0,
        max_deformation=10.0,
        num_critical_fibers=10  # Reduce for faster testing
    )

    # Initialize the warper
    warper = FishermansNetWarper(config)

    # Run the warping algorithm - this returns a dictionary
    result = warper.warp_volume(
        volume, fiber_volume, fiber_orientations,
        num_iterations=20  # Reduce iterations for faster testing
    )

    print("Warping complete!")

    return result

def visualize_results(volume, warped_volume, deformation_field, output_dir=None):
    """
    Visualize the original volume, warped volume, and deformation field.
    
    Args:
        volume: Original volume
        warped_volume: Warped volume
        deformation_field: Deformation field
        output_dir: Directory to save visualization images (optional)
    """
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert to NumPy for visualization
    volume_np = np.array(volume)
    warped_np = np.array(warped_volume)
    
    # Normalize for visualization
    volume_norm = (volume_np - np.min(volume_np)) / (np.max(volume_np) - np.min(volume_np))
    warped_norm = (warped_np - np.min(warped_np)) / (np.max(warped_np) - np.min(warped_np))
    
    # Choose middle slice for 2D visualization
    mid_z = volume.shape[0] // 2
    
    # Create figure with two subplots
    plt.figure(figsize=(12, 6))
    
    # Original volume slice
    plt.subplot(1, 2, 1)
    plt.imshow(volume_norm[mid_z], cmap='gray')
    plt.title("Original Volume (Slice {})".format(mid_z))
    plt.axis('off')
    
    # Warped volume slice
    plt.subplot(1, 2, 2)
    plt.imshow(warped_norm[mid_z], cmap='gray')
    plt.title("Warped Volume (Slice {})".format(mid_z))
    plt.axis('off')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "volume_comparison.png"), dpi=300)
    
    plt.tight_layout()
    plt.show()
    
    # Visualize the deformation field
    plt.figure(figsize=(10, 8))
    
    # Get deformation vectors for the middle slice
    u = np.array(deformation_field[mid_z, :, :, 1])
    v = np.array(deformation_field[mid_z, :, :, 2])
    
    # Normalize vectors for visualization
    magnitude = np.sqrt(u**2 + v**2)
    u_norm = u / (np.max(magnitude) + 1e-10)
    v_norm = v / (np.max(magnitude) + 1e-10)
    
    # Create a grid for quiver plot
    y, x = np.mgrid[0:u.shape[0]:5, 0:u.shape[1]:5]
    
    # Plot deformation field
    plt.imshow(np.array(volume[mid_z]), cmap='gray', alpha=0.7)
    plt.quiver(x, y, u_norm[::5, ::5], v_norm[::5, ::5], color='red', alpha=0.8, scale=30)
    plt.title("Deformation Field")
    plt.axis('off')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "deformation_field.png"), dpi=300)
    
    plt.show()

def main():
    # Create output directory for results
    output_dir = Path("./warping_results")
    output_dir.mkdir(exist_ok=True)
    
    # List available segments
    segments = list_available_segments()
    
    if segments:
        print(f"\nFound {len(segments)} locally downloaded segments:")
        for i, segment in enumerate(segments):
            print(f"  {i+1}. {segment}")
        
        # Use the first segment as default
        segment_id = segments[0]
    else:
        print("\nNo local segments found.")
        print("Please make sure to download segments data using the download_data.sh script.")
        print("The data should be in VesuviusDataDownload/Scroll1/segments/")
        return
    
    print(f"\nUsing scroll segment: {segment_id}")
    
    # Load a small portion of a scroll segment (30 z-layers to keep memory usage manageable)
    try:
        volume, segment_info = load_local_scroll_data(segment_id=segment_id, layers=(0, 30))
    except Exception as e:
        print(f"Error loading local data: {e}")
        print("Please make sure the download has completed successfully.")
        return
    
    # Create synthetic fiber data for testing
    # In a real application, you would use actual fiber predictions
    fiber_volume, fiber_orientations = synthesize_fiber_data(volume)
    
    # Run the warping pipeline
    print("\nRunning warping pipeline on real scroll data...")
    result = run_warping_pipeline(volume, fiber_volume, fiber_orientations)
    
    # Use our visualization module to create comprehensive visualizations
    from fishermans_net.mlx_ops.visualization import visualize_warping_results
    
    # Create visualizations for different slices and axes
    print("\nCreating visualizations...")
    
    # Z-axis visualization (mid slice)
    mid_z = volume.shape[0] // 2
    fig_z = visualize_warping_results(
        original=volume,
        warped=result['warped_volume'],
        deformation_field=result['deformation_field'],
        slice_idx=mid_z,
        axis=0,
        output_path=str(output_dir / "warping_z_slice.png")
    )
    
    # Y-axis visualization (mid slice)
    mid_y = volume.shape[1] // 2
    fig_y = visualize_warping_results(
        original=volume,
        warped=result['warped_volume'],
        deformation_field=result['deformation_field'],
        slice_idx=mid_y,
        axis=1,
        output_path=str(output_dir / "warping_y_slice.png")
    )
    
    # X-axis visualization (mid slice)
    mid_x = volume.shape[2] // 2
    fig_x = visualize_warping_results(
        original=volume,
        warped=result['warped_volume'],
        deformation_field=result['deformation_field'],
        slice_idx=mid_x,
        axis=2,
        output_path=str(output_dir / "warping_x_slice.png")
    )
    
    print(f"Results saved to {output_dir}")
    
    # Display the visualizations
    plt.show()

if __name__ == "__main__":
    main()
