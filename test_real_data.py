#!/usr/bin/env python3
"""
Script to test the Fisherman's Net warping pipeline with real Vesuvius scroll data.
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import mlx.core as mx

# Add the Fisherman's Net project to the path
sys.path.append(str(Path(os.path.abspath(__file__)).parent / "fishermans-net-vesuvius"))

# Import Fisherman's Net components
from fishermans_net.core.warping import FishermansNetWarper, WarpingConfig
from fishermans_net.mlx_ops.visualization import visualize_deformation_field, visualize_volume

# Import vesuvius for scroll data access
import vesuvius
from vesuvius import Volume

def load_scroll_data(segment_id=20230210180739, layers=None):
    """
    Load a scroll segment from the Vesuvius Challenge data.
    
    Args:
        segment_id: ID of the scroll segment to load (as integer)
        layers: List of z-layers to load, or None for all
        
    Returns:
        volume: 3D array of CT scan data
        segment_info: Dictionary with metadata about the segment
    """
    print(f"Loading scroll segment {segment_id}...")
    
    # Default to loading a small z-range if none specified
    if layers is None:
        layers = (0, 30)  # Load first 30 layers as default
    
    # Make sure segment_id is an integer
    try:
        if isinstance(segment_id, str):
            segment_id = int(segment_id)
    except ValueError:
        print(f"Warning: Could not convert segment_id '{segment_id}' to integer")
    
    # Try loading real data first
    try:
        # Try loading with default scroll parameters according to the error message guidance
        try:
            # API requires segment_id as integer
            volume_obj = vesuvius.Volume(
                type="segment", 
                scroll_id=1,  # Default scroll_id
                energy=54,    # Default energy
                resolution=7.91, # Default resolution
                segment_id=segment_id, 
                verbose=True
            )
        except Exception as e1:
            print(f"Failed with default params: {e1}")
            # Try direct loading as a fallback
            print("Attempting direct segment loading...")
            volume_obj = vesuvius.Volume(type="segment", segment_id=segment_id, verbose=True)
        
        # Get the shape of the volume
        vol_shape = volume_obj.shape()
        print(f"Volume shape: {vol_shape}")
        
        # Calculate the z_range based on layers
        z_start, z_end = layers
        z_end = min(z_end, vol_shape[0])  # Make sure we don't exceed volume bounds
        
        # Get the actual volume data within the specified layer range
        start_layer = max(0, z_start)
        num_layers = z_end - start_layer
        
        print(f"Loading layers {start_layer} to {z_end-1} (total: {num_layers})...")
        
        # Load data slice by slice to avoid memory issues
        volume_slices = []
        for z in range(start_layer, z_end):
            # Load a single z-slice
            slice_data = volume_obj[z]
            volume_slices.append(slice_data)
        
        # Stack the slices to form a volume
        data = np.stack(volume_slices, axis=0)
        
        # Convert to MLX array
        mlx_data = mx.array(data)
        
        # Print some information about the data
        print(f"Loaded volume with shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Value range: [{np.min(data)}, {np.max(data)}]")
        
        # Return both the MLX array and the original volume object for metadata
        return mlx_data, volume_obj
    
    except Exception as e:
        print(f"Error loading scroll segment: {e}")
        
        # As a fallback, try loading a default scroll
        try:
            print("\nAttempting to load a default scroll (Scroll1)...")
            volume_obj = vesuvius.Volume(type="scroll", scroll_id=1, verbose=True)
            
            # Get a small sample of the scroll data
            print("Loading a small sample of the scroll...")
            # Get shape to know the bounds
            vol_shape = volume_obj.shape()
            print(f"Full scroll shape: {vol_shape}")
            
            # Load a small subset of the scroll
            z_start, z_end = 0, min(30, vol_shape[0])
            y_start, y_end = 0, min(200, vol_shape[1])
            x_start, x_end = 0, min(200, vol_shape[2])
            
            # Load data slice by slice
            volume_slices = []
            for z in range(z_start, z_end):
                # Get a slice with limited y and x range
                slice_data = volume_obj[z, y_start:y_end, x_start:x_end]
                volume_slices.append(slice_data)
            
            # Stack the slices
            data = np.stack(volume_slices, axis=0)
            mlx_data = mx.array(data)
            
            print(f"Loaded scroll sample with shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Value range: [{np.min(data)}, {np.max(data)}]")
            
            return mlx_data, volume_obj
            
        except Exception as e2:
            print(f"Error loading default scroll: {e2}")
            
            # For testing purposes, create a synthetic volume as last resort
            print("\nCreating a synthetic volume for testing...")
            synthetic_volume = mx.random.normal(shape=(30, 200, 200))
            synthetic_volume = mx.abs(synthetic_volume)  # Make all values positive like CT data
            return synthetic_volume, None

def synthesize_fiber_data(volume_data, fiber_density=0.05, fiber_thickness=3):
    """
    Create synthetic fiber data since we don't have real fiber predictions.
    This is a temporary solution for testing the warping pipeline.
    
    Args:
        volume_data: Original CT volume data
        fiber_density: Density of synthetic fibers
        fiber_thickness: Thickness of synthetic fibers
        
    Returns:
        fiber_volume: Synthetic fiber prediction volume
        fiber_orientations: Synthetic fiber orientation vectors
    """
    print("Generating synthetic fiber data for testing...")
    
    # Original volume shape
    shape = volume_data.shape
    
    # Create empty fiber volume
    fiber_volume = mx.zeros(shape)
    
    # Use gradient information to create synthetic fibers in high-gradient regions
    dx = mx.array(np.gradient(np.array(volume_data), axis=0))
    dy = mx.array(np.gradient(np.array(volume_data), axis=1))
    dz = mx.array(np.gradient(np.array(volume_data), axis=2))
    
    # Compute gradient magnitude
    gradient_magnitude = mx.sqrt(dx**2 + dy**2 + dz**2)
    
    # Create fiber predictions in high gradient regions
    threshold = mx.array(np.percentile(np.array(gradient_magnitude), 100 * (1 - fiber_density)))
    fiber_volume = mx.where(gradient_magnitude > threshold, 1.0, 0.0)
    
    # Create fiber orientations - assume they mainly follow the y-axis (vertical direction)
    # but with some variation based on the gradient
    fiber_orientations = mx.zeros((*shape, 3))
    
    # Normalize gradient vectors to create orientation vectors
    norm = mx.sqrt(dx**2 + dy**2 + dz**2) + 1e-10
    
    # Create a base orientation that's primarily in the y direction
    base_orientations = mx.array(np.zeros((*shape, 3)))
    base_orientations = mx.array(np.zeros((*shape, 3)))
    
    # Convert to numpy for easier manipulation
    base_np = np.zeros((*shape, 3))
    base_np[..., 1] = 1.0  # Primary orientation along y-axis
    
    # Add some noise based on gradients
    dx_np = np.array(dx) / (np.array(norm) + 1e-10)
    dy_np = np.array(dy) / (np.array(norm) + 1e-10)
    dz_np = np.array(dz) / (np.array(norm) + 1e-10)
    
    # Blend base orientation with gradient direction
    blend_factor = 0.7  # 70% base, 30% gradient
    orientations_np = np.zeros((*shape, 3))
    orientations_np[..., 0] = (1 - blend_factor) * dx_np
    orientations_np[..., 1] = blend_factor + (1 - blend_factor) * dy_np
    orientations_np[..., 2] = (1 - blend_factor) * dz_np
    
    # Re-normalize
    norms_np = np.sqrt(np.sum(orientations_np**2, axis=-1, keepdims=True)) + 1e-10
    orientations_np = orientations_np / norms_np
    
    # Convert back to MLX
    fiber_orientations = mx.array(orientations_np)
    
    print(f"Created synthetic fiber volume with {mx.sum(fiber_volume > 0).item()} fiber voxels")
    print(f"Fiber volume shape: {fiber_volume.shape}")
    print(f"Fiber orientations shape: {fiber_orientations.shape}")
    
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
    
    # Create warping configuration
    config = WarpingConfig(
        elasticity=0.8,
        viscosity=0.2,
        fiber_strength=1.5,
        smoothing_sigma=1.0,
        max_deformation=50.0,
        convergence_threshold=0.0001,
        step_size=0.1
    )
    
    # Create warper
    warper = FishermansNetWarper(config)
    
    # Run warping
    result = warper.warp_volume(
        volume=volume,
        fiber_volume=fiber_volume,
        fiber_orientations=fiber_orientations,
        num_iterations=30,
        checkpoint_interval=5
    )
    
    print("Warping complete!")
    print(f"Final metrics: {result['metrics']}")
    
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
    print("Visualizing results...")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for volume comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Show middle slice of original volume
    mid_z = volume.shape[0] // 2
    axes[0].imshow(np.array(volume[mid_z]), cmap='gray')
    axes[0].set_title("Original Volume")
    axes[0].axis('off')
    
    # Show middle slice of warped volume
    axes[1].imshow(np.array(warped_volume[mid_z]), cmap='gray')
    axes[1].set_title("Warped Volume")
    axes[1].axis('off')
    
    plt.tight_layout()
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "volume_comparison.png"), dpi=300)
    
    plt.figure(figsize=(8, 8))
    # Visualize deformation field
    deformation_field_np = np.array(deformation_field)
    
    # Extract x and y components for visualization
    u = deformation_field_np[mid_z, ::4, ::4, 0]  # x component
    v = deformation_field_np[mid_z, ::4, ::4, 1]  # y component
    
    # Create meshgrid for quiver plot
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    
    # Normalize vectors for better visualization
    magnitude = np.sqrt(u**2 + v**2) + 1e-10
    u_norm = u / magnitude
    v_norm = v / magnitude
    
    # Plot deformation field
    plt.imshow(np.array(volume[mid_z]), cmap='gray', alpha=0.7)
    plt.quiver(x, y, u_norm, v_norm, color='red', alpha=0.8, scale=30)
    plt.title("Deformation Field")
    plt.axis('off')
    
    if output_dir:
        plt.savefig(os.path.join(output_dir, "deformation_field.png"), dpi=300)
    
    plt.show()

def browse_available_data():
    """Browse available Vesuvius data and provide information about it."""
    print("\nBrowsing available Vesuvius Challenge data:\n")
    
    try:
        # Use list_files() to get information about available scrolls/segments
        from vesuvius import list_files
        scroll_data = list_files()
        
        # Extract segment IDs from the scroll data
        segments = []
        if "segments" in scroll_data:
            for segment_key, segment_info in scroll_data["segments"].items():
                if isinstance(segment_info, dict) and "id" in segment_info:
                    segments.append(segment_info["id"])
        
        # If we found segments, display them
        if segments:
            print(f"Found {len(segments)} scroll segments available:")
            for i, segment in enumerate(segments[:5]):  # Show first 5 for brevity
                print(f"  {i+1}. {segment}")
            
            if len(segments) > 5:
                print(f"  ... and {len(segments) - 5} more segments")
            
            # Return the first segment as default
            return segments[0]
        else:
            print("No segments found in scroll data, using default.")
            return "20230210180739"  # Default segment ID
        
    except Exception as e:
        print(f"Error browsing Vesuvius data: {e}")
        print("Using default segment ID instead.")
        return "20230210180739"  # Default segment ID

def main():
    # Create output directory for results
    output_dir = Path("./warping_results")
    output_dir.mkdir(exist_ok=True)
    
    # Browse available data and choose a segment
    segment_id = browse_available_data()
    
    print(f"\nUsing scroll segment: {segment_id}")
    
    # Load a small portion of a scroll segment (30 z-layers to keep memory usage manageable)
    volume, segment_info = load_scroll_data(segment_id=segment_id, layers=(0, 30))
    
    # Create synthetic fiber data for testing
    # In a real application, you would use actual fiber predictions
    fiber_volume, fiber_orientations = synthesize_fiber_data(volume)
    
    # Run the warping pipeline
    print("\nRunning warping pipeline on real scroll data...")
    result = run_warping_pipeline(volume, fiber_volume, fiber_orientations)
    
    # Use our new visualization module to create comprehensive visualizations
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
