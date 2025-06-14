"""
Visualization utilities for 3D volumes and deformation fields.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
import mlx.core as mx

def visualize_volume(volume: mx.array, 
                     slice_idx: Optional[int] = None, 
                     axis: int = 0, 
                     figsize: Tuple[int, int] = (8, 8),
                     cmap: str = 'gray',
                     title: Optional[str] = None) -> plt.Figure:
    """
    Visualize a slice from a 3D volume.
    
    Args:
        volume: 3D array to visualize
        slice_idx: Index of the slice to visualize (default: middle slice)
        axis: Axis along which to take the slice (0=Z, 1=Y, 2=X)
        figsize: Figure size
        cmap: Colormap
        title: Plot title
        
    Returns:
        Figure object
    """
    # Convert to numpy if needed
    if isinstance(volume, mx.array):
        volume_np = np.array(volume)
    else:
        volume_np = volume
    
    # Determine slice index if not provided
    if slice_idx is None:
        slice_idx = volume_np.shape[axis] // 2
    
    # Extract the slice
    if axis == 0:
        slice_data = volume_np[slice_idx]
    elif axis == 1:
        slice_data = volume_np[:, slice_idx]
    else:  # axis == 2
        slice_data = volume_np[:, :, slice_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(slice_data, cmap=cmap)
    plt.colorbar(im, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Slice {slice_idx} along axis {axis}")
    
    return fig

def visualize_deformation_field(deformation_field: mx.array,
                               volume: Optional[mx.array] = None,
                               slice_idx: Optional[int] = None,
                               axis: int = 0,
                               subsample: int = 4,
                               figsize: Tuple[int, int] = (10, 10),
                               scale: Optional[float] = None,
                               title: Optional[str] = None) -> plt.Figure:
    """
    Visualize a slice of a deformation field as arrows.
    
    Args:
        deformation_field: Deformation field (shape: [*volume.shape, 3])
        volume: Optional underlying volume to display as background
        slice_idx: Index of the slice to visualize (default: middle slice)
        axis: Axis along which to take the slice (0=Z, 1=Y, 2=X)
        subsample: Subsampling factor to avoid too many arrows
        figsize: Figure size
        scale: Scale factor for arrows (default: automatic)
        title: Plot title
        
    Returns:
        Figure object
    """
    # Convert to numpy if needed
    if isinstance(deformation_field, mx.array):
        deformation_field_np = np.array(deformation_field)
    else:
        deformation_field_np = deformation_field
    
    # Determine slice index if not provided
    if slice_idx is None:
        slice_idx = deformation_field_np.shape[axis] // 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Show volume slice as background if provided
    if volume is not None:
        if isinstance(volume, mx.array):
            volume_np = np.array(volume)
        else:
            volume_np = volume
            
        if axis == 0:
            bg_slice = volume_np[slice_idx]
        elif axis == 1:
            bg_slice = volume_np[:, slice_idx]
        else:  # axis == 2
            bg_slice = volume_np[:, :, slice_idx]
            
        ax.imshow(bg_slice, cmap='gray', alpha=0.7)
    
    # Extract deformation vectors for the slice
    if axis == 0:  # Z-slice
        # For Z-slice, we want X and Y components (index 0 and 1)
        u = deformation_field_np[slice_idx, ::subsample, ::subsample, 0]  # X component
        v = deformation_field_np[slice_idx, ::subsample, ::subsample, 1]  # Y component
        y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
        
    elif axis == 1:  # Y-slice
        # For Y-slice, we want X and Z components (index 0 and 2)
        u = deformation_field_np[::subsample, slice_idx, ::subsample, 0]  # X component
        v = deformation_field_np[::subsample, slice_idx, ::subsample, 2]  # Z component
        z, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
        y, x = z, x  # rename for quiver plot
        
    else:  # axis == 2, X-slice
        # For X-slice, we want Y and Z components (index 1 and 2)
        u = deformation_field_np[::subsample, ::subsample, slice_idx, 1]  # Y component
        v = deformation_field_np[::subsample, ::subsample, slice_idx, 2]  # Z component
        z, y = np.mgrid[0:u.shape[0], 0:u.shape[1]]
        y, x = z, y  # rename for quiver plot
    
    # Normalize vectors for better visualization
    magnitude = np.sqrt(u**2 + v**2) + 1e-10
    max_mag = np.max(magnitude)
    
    # Normalize by max magnitude for better visualization
    if max_mag > 0:
        u_norm = u / max_mag
        v_norm = v / max_mag
    else:
        u_norm, v_norm = u, v
    
    # Plot vector field
    q = ax.quiver(x, y, u_norm, v_norm, magnitude, 
                 cmap='viridis', scale=scale or 30, 
                 alpha=0.8, pivot='mid')
    
    plt.colorbar(q, ax=ax, label='Deformation Magnitude')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Deformation Field - Slice {slice_idx} along axis {axis}")
    
    return fig

def visualize_comparison(original: mx.array,
                        warped: mx.array,
                        slice_idx: Optional[int] = None,
                        axis: int = 0,
                        figsize: Tuple[int, int] = (12, 6),
                        title: Optional[str] = None) -> plt.Figure:
    """
    Visualize a side-by-side comparison of original and warped volumes.
    
    Args:
        original: Original volume
        warped: Warped volume
        slice_idx: Index of the slice to visualize (default: middle slice)
        axis: Axis along which to take the slice (0=Z, 1=Y, 2=X)
        figsize: Figure size
        title: Plot title
        
    Returns:
        Figure object
    """
    # Convert to numpy if needed
    if isinstance(original, mx.array):
        original_np = np.array(original)
    else:
        original_np = original
        
    if isinstance(warped, mx.array):
        warped_np = np.array(warped)
    else:
        warped_np = warped
    
    # Determine slice index if not provided
    if slice_idx is None:
        slice_idx = original_np.shape[axis] // 2
    
    # Extract slices
    if axis == 0:
        orig_slice = original_np[slice_idx]
        warped_slice = warped_np[slice_idx]
    elif axis == 1:
        orig_slice = original_np[:, slice_idx]
        warped_slice = warped_np[:, slice_idx]
    else:  # axis == 2
        orig_slice = original_np[:, :, slice_idx]
        warped_slice = warped_np[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot original slice
    im1 = axes[0].imshow(orig_slice, cmap='gray')
    axes[0].set_title("Original")
    plt.colorbar(im1, ax=axes[0])
    
    # Plot warped slice
    im2 = axes[1].imshow(warped_slice, cmap='gray')
    axes[1].set_title("Warped")
    plt.colorbar(im2, ax=axes[1])
    
    if title:
        fig.suptitle(title)
    else:
        fig.suptitle(f"Comparison - Slice {slice_idx} along axis {axis}")
    
    plt.tight_layout()
    return fig

def visualize_warping_results(original: mx.array,
                             warped: mx.array,
                             deformation_field: mx.array,
                             slice_idx: Optional[int] = None,
                             axis: int = 0,
                             output_path: Optional[str] = None):
    """
    Create a comprehensive visualization of warping results.
    
    Args:
        original: Original volume
        warped: Warped volume
        deformation_field: Deformation field
        slice_idx: Index of the slice to visualize (default: middle slice)
        axis: Axis along which to take the slice (0=Z, 1=Y, 2=X)
        output_path: Path to save the figure (optional)
    """
    # Determine slice index if not provided
    if slice_idx is None:
        slice_idx = original.shape[axis] // 2
    
    # Create figure with 1 row and 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert to numpy
    original_np = np.array(original)
    warped_np = np.array(warped)
    deformation_field_np = np.array(deformation_field)
    
    # Extract slices
    if axis == 0:
        orig_slice = original_np[slice_idx]
        warped_slice = warped_np[slice_idx]
        # For deformation field, extract X and Y components
        u = deformation_field_np[slice_idx, ::4, ::4, 0]  # X component
        v = deformation_field_np[slice_idx, ::4, ::4, 1]  # Y component
        y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    elif axis == 1:
        orig_slice = original_np[:, slice_idx]
        warped_slice = warped_np[:, slice_idx]
        # For deformation field, extract X and Z components
        u = deformation_field_np[::4, slice_idx, ::4, 0]  # X component
        v = deformation_field_np[::4, slice_idx, ::4, 2]  # Z component
        z, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
        y, x = z, x  # rename for quiver plot
    else:  # axis == 2
        orig_slice = original_np[:, :, slice_idx]
        warped_slice = warped_np[:, :, slice_idx]
        # For deformation field, extract Y and Z components
        u = deformation_field_np[::4, ::4, slice_idx, 1]  # Y component
        v = deformation_field_np[::4, ::4, slice_idx, 2]  # Z component
        z, y = np.mgrid[0:u.shape[0], 0:u.shape[1]]
        y, x = z, y  # rename for quiver plot
    
    # Plot original slice
    axes[0].imshow(orig_slice, cmap='gray')
    axes[0].set_title("Original Volume")
    axes[0].axis('off')
    
    # Plot warped slice
    axes[1].imshow(warped_slice, cmap='gray')
    axes[1].set_title("Warped Volume")
    axes[1].axis('off')
    
    # Plot deformation field
    # Normalize vectors for better visualization
    magnitude = np.sqrt(u**2 + v**2) + 1e-10
    max_mag = np.max(magnitude)
    
    # Normalize by max magnitude for better visualization
    if max_mag > 0:
        u_norm = u / max_mag
        v_norm = v / max_mag
    else:
        u_norm, v_norm = u, v
    
    # Show warped slice as background
    axes[2].imshow(warped_slice, cmap='gray', alpha=0.7)
    q = axes[2].quiver(x, y, u_norm, v_norm, magnitude, 
                     cmap='viridis', scale=30, 
                     alpha=0.8, pivot='mid')
    axes[2].set_title("Deformation Field")
    axes[2].axis('off')
    
    plt.colorbar(q, ax=axes[2], label='Deformation Magnitude')
    
    fig.suptitle(f"Warping Results - Slice {slice_idx} along axis {axis}")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return fig
