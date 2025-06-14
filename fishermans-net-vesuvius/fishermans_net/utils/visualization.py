"""
Visualization utilities for Fisherman's Net warping results.

Functions for:
- Visualizing volumes
- Plotting warping metrics
- Creating HTML reports
- Visualizing deformation fields
- Visualizing fiber tracing
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import base64
import io
import json
import datetime

import numpy as np
import matplotlib.pyplot as plt
import mlx.core as mx
from matplotlib.figure import Figure
from matplotlib.colors import Normalize


logger = logging.getLogger(__name__)


def visualize_slice(volume: mx.array, 
                   slice_idx: int = None, 
                   axis: int = 2,
                   cmap: str = 'gray',
                   title: str = None) -> Figure:
    """
    Visualize a single slice from a 3D volume.
    
    Args:
        volume: 3D volume (MLX array)
        slice_idx: Index of slice to show (None for middle)
        axis: Axis to slice (0=depth, 1=height, 2=width)
        cmap: Colormap
        title: Optional title
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy for matplotlib
    volume_np = np.array(volume)
    
    # Get slice index (middle by default)
    if slice_idx is None:
        slice_idx = volume_np.shape[axis] // 2
    
    # Get slice
    if axis == 0:
        slice_data = volume_np[slice_idx, :, :]
    elif axis == 1:
        slice_data = volume_np[:, slice_idx, :]
    else:
        slice_data = volume_np[:, :, slice_idx]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(slice_data, cmap=cmap)
    fig.colorbar(im, ax=ax)
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"Slice {slice_idx} along axis {axis}")
    
    return fig


def visualize_comparison(original: mx.array, 
                        warped: mx.array,
                        slice_idx: int = None,
                        axis: int = 2,
                        cmap: str = 'gray') -> Figure:
    """
    Compare original and warped volumes side by side.
    
    Args:
        original: Original volume
        warped: Warped volume
        slice_idx: Index of slice to show
        axis: Axis to slice
        cmap: Colormap
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy for matplotlib
    original_np = np.array(original)
    warped_np = np.array(warped)
    
    # Get slice index (middle by default)
    if slice_idx is None:
        slice_idx = original_np.shape[axis] // 2
    
    # Get slices
    if axis == 0:
        orig_slice = original_np[slice_idx, :, :]
        warp_slice = warped_np[slice_idx, :, :]
    elif axis == 1:
        orig_slice = original_np[:, slice_idx, :]
        warp_slice = warped_np[:, slice_idx, :]
    else:
        orig_slice = original_np[:, :, slice_idx]
        warp_slice = warped_np[:, :, slice_idx]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Ensure same colorscale
    vmin = min(orig_slice.min(), warp_slice.min())
    vmax = max(orig_slice.max(), warp_slice.max())
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot
    axes[0].imshow(orig_slice, cmap=cmap, norm=norm)
    axes[0].set_title("Original")
    
    axes[1].imshow(warp_slice, cmap=cmap, norm=norm)
    axes[1].set_title("Warped")
    
    fig.suptitle(f"Slice {slice_idx} along axis {axis}")
    
    return fig


def visualize_metrics(metrics: Dict[str, List[float]]) -> Figure:
    """
    Visualize warping metrics over iterations.
    
    Args:
        metrics: Dictionary of metrics from warp_volume
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot flatness
    if 'flatness' in metrics:
        axes[0].plot(metrics['flatness'], 'b-', label='Flatness')
        axes[0].set_ylabel('Flatness')
        axes[0].set_title('Warping Metrics')
        axes[0].grid(True)
        axes[0].legend()
    
    # Plot strain energy
    if 'strain' in metrics:
        axes[1].plot(metrics['strain'], 'r-', label='Strain Energy')
        axes[1].set_ylabel('Strain Energy')
        axes[1].set_xlabel('Iteration')
        axes[1].grid(True)
        axes[1].legend()
    
    # Plot convergence if available
    if 'convergence' in metrics:
        ax_conv = axes[1].twinx()
        ax_conv.plot(metrics['convergence'], 'g--', label='Convergence')
        ax_conv.set_ylabel('Convergence')
        ax_conv.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


def visualize_deformation(deformation: mx.array,
                         slice_idx: int = None,
                         axis: int = 2,
                         scale: float = 5.0,
                         skip: int = 10) -> Figure:
    """
    Visualize deformation field as a quiver plot.
    
    Args:
        deformation: Deformation field of shape (H, W, D, 3)
        slice_idx: Index of slice to show
        axis: Axis to slice
        scale: Arrow scale factor
        skip: Show arrows every N pixels for clarity
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy for matplotlib
    deformation_np = np.array(deformation)
    
    # Get slice index (middle by default)
    if slice_idx is None:
        slice_idx = deformation_np.shape[axis] // 2
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    if axis == 0:
        # YZ plane
        Y, Z = np.meshgrid(
            np.arange(0, deformation_np.shape[1], skip),
            np.arange(0, deformation_np.shape[2], skip)
        )
        
        U = deformation_np[slice_idx, Y, Z, 1]  # Y component
        V = deformation_np[slice_idx, Y, Z, 2]  # Z component
        ax.set_title(f"Deformation Field (YZ Plane, X={slice_idx})")
        
    elif axis == 1:
        # XZ plane
        X, Z = np.meshgrid(
            np.arange(0, deformation_np.shape[0], skip),
            np.arange(0, deformation_np.shape[2], skip)
        )
        
        U = deformation_np[X, slice_idx, Z, 0]  # X component
        V = deformation_np[X, slice_idx, Z, 2]  # Z component
        ax.set_title(f"Deformation Field (XZ Plane, Y={slice_idx})")
        
    else:
        # XY plane
        X, Y = np.meshgrid(
            np.arange(0, deformation_np.shape[0], skip),
            np.arange(0, deformation_np.shape[1], skip)
        )
        
        U = deformation_np[X, Y, slice_idx, 0]  # X component
        V = deformation_np[X, Y, slice_idx, 1]  # Y component
        ax.set_title(f"Deformation Field (XY Plane, Z={slice_idx})")
    
    # Plot quiver
    ax.quiver(X, Y, U, V, scale=scale)
    ax.set_aspect('equal')
    
    return fig


def visualize_critical_fibers(volume: mx.array, 
                             fibers: List[mx.array],
                             slice_indices: List[int] = None) -> Figure:
    """
    Visualize critical fibers overlaid on volume.
    
    Args:
        volume: 3D volume
        fibers: List of fiber paths, each as (N, 3) array
        slice_indices: List of slice indices to show (3 slices if None)
        
    Returns:
        Matplotlib figure
    """
    # Convert to numpy
    volume_np = np.array(volume)
    
    # Default slice indices (25%, 50%, 75%)
    if slice_indices is None:
        depth = volume_np.shape[2]
        slice_indices = [depth//4, depth//2, 3*depth//4]
    
    # Create figure
    fig, axes = plt.subplots(1, len(slice_indices), figsize=(16, 6))
    if len(slice_indices) == 1:
        axes = [axes]  # Make into list for iteration
    
    # Plot each slice
    for i, slice_idx in enumerate(slice_indices):
        # Plot volume slice
        axes[i].imshow(volume_np[:, :, slice_idx], cmap='gray')
        axes[i].set_title(f"Slice {slice_idx}")
        
        # Plot fibers
        for fiber in fibers:
            fiber_np = np.array(fiber)
            
            # Find fiber points near this slice
            mask = (fiber_np[:, 2] > slice_idx - 2) & (fiber_np[:, 2] < slice_idx + 2)
            if np.any(mask):
                axes[i].plot(fiber_np[mask, 1], fiber_np[mask, 0], 'r.', markersize=1)
    
    plt.tight_layout()
    return fig


def fig_to_base64(fig: Figure) -> str:
    """
    Convert matplotlib figure to base64 string for HTML embedding.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str


def create_warping_report(result: Dict[str, Any],
                         output_path: Union[str, Path],
                         input_path: Union[str, Path] = None,
                         elapsed_time: float = None) -> None:
    """
    Create an HTML report with warping results visualization.
    
    Args:
        result: Warping result dictionary
        output_path: Path to save HTML report
        input_path: Path to input file (for metadata)
        elapsed_time: Execution time in seconds
    """
    # Convert paths to Path objects
    output_path = Path(output_path)
    if input_path:
        input_path = Path(input_path)
    
    # Create basic metadata
    metadata = {
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "input_file": str(input_path) if input_path else "Unknown",
        "elapsed_time": f"{elapsed_time:.1f} seconds" if elapsed_time else "Unknown",
    }
    
    # Add algorithm parameters if available
    if 'config' in result:
        config = result['config']
        metadata.update({k: str(v) for k, v in config.__dict__.items()})
    
    # Add result metrics
    for key, value in result.get('metrics', {}).items():
        if isinstance(value, list) and len(value) > 0:
            metadata[f"final_{key}"] = f"{value[-1]:.4f}"
    
    # Create images
    images = {}
    
    # Volume comparison
    if 'original_volume' in result and 'warped_volume' in result:
        fig = visualize_comparison(
            result['original_volume'],
            result['warped_volume']
        )
        images['comparison'] = fig_to_base64(fig)
    
    # Metrics plot
    if 'metrics' in result:
        fig = visualize_metrics(result['metrics'])
        images['metrics'] = fig_to_base64(fig)
    
    # Deformation field
    if 'deformation_field' in result:
        fig = visualize_deformation(result['deformation_field'])
        images['deformation'] = fig_to_base64(fig)
    
    # Critical fibers
    if 'critical_fibers' in result and 'original_volume' in result:
        fig = visualize_critical_fibers(
            result['original_volume'], 
            result['critical_fibers']
        )
        images['fibers'] = fig_to_base64(fig)
    
    # Create HTML
    html = []
    html.append("<!DOCTYPE html>")
    html.append("<html>")
    html.append("<head>")
    html.append("<title>Fisherman's Net Warping Report</title>")
    html.append("<style>")
    html.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html.append("h1 { color: #2c3e50; }")
    html.append("h2 { color: #3498db; }")
    html.append(".metadata { background: #f8f9fa; padding: 15px; border-radius: 5px; }")
    html.append(".metadata dt { font-weight: bold; }")
    html.append(".metadata dd { margin-bottom: 10px; }")
    html.append(".figure { margin: 20px 0; text-align: center; }")
    html.append(".figure img { max-width: 100%; box-shadow: 0 0 10px rgba(0,0,0,0.1); }")
    html.append(".figure-caption { font-style: italic; color: #7f8c8d; margin-top: 10px; }")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    
    # Header
    html.append("<h1>Fisherman's Net Warping Report</h1>")
    
    # Metadata
    html.append("<h2>Metadata</h2>")
    html.append("<div class='metadata'>")
    html.append("<dl>")
    for key, value in metadata.items():
        html.append(f"<dt>{key}</dt><dd>{value}</dd>")
    html.append("</dl>")
    html.append("</div>")
    
    # Images
    for name, img_data in images.items():
        title = name.replace('_', ' ').title()
        html.append(f"<h2>{title}</h2>")
        html.append("<div class='figure'>")
        html.append(f"<img src='data:image/png;base64,{img_data}' />")
        html.append(f"<div class='figure-caption'>{title}</div>")
        html.append("</div>")
    
    html.append("</body>")
    html.append("</html>")
    
    # Write HTML file
    with open(output_path, 'w') as f:
        f.write('\n'.join(html))
    
    logger.info(f"Created warping report at {output_path}")
