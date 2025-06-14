"""
Advanced interpolation operations for MLX arrays.

Implementations of:
- Trilinear interpolation
- Tricubic interpolation (TODO)
- Grid sampling
"""

import mlx.core as mx
import numpy as np
from typing import Tuple


def trilinear_interpolate(volume: mx.array, coords: mx.array) -> mx.array:
    """
    Properly implemented trilinear interpolation for 3D MLX arrays.
    
    Args:
        volume: Input volume of shape (H, W, D) or (H, W, D, C)
        coords: Query coordinates of shape (..., 3), in order (y, x, z)
        
    Returns:
        Interpolated values of shape (...)
    """
    # Get volume shape
    if volume.ndim == 3:
        H, W, D = volume.shape
        has_channels = False
    else:
        H, W, D, C = volume.shape
        has_channels = True
    
    # Extract coordinates
    y, x, z = coords[..., 0], coords[..., 1], coords[..., 2]
    
    # Clip coordinates to valid range
    y = mx.clip(y, 0, H - 1.00001)
    x = mx.clip(x, 0, W - 1.00001)
    z = mx.clip(z, 0, D - 1.00001)
    
    # Get integer and fractional parts
    y0 = mx.floor(y).astype(mx.int32)
    x0 = mx.floor(x).astype(mx.int32)
    z0 = mx.floor(z).astype(mx.int32)
    
    y1 = mx.minimum(y0 + 1, H - 1)
    x1 = mx.minimum(x0 + 1, W - 1)
    z1 = mx.minimum(z0 + 1, D - 1)
    
    yd = y - y0.astype(mx.float32)
    xd = x - x0.astype(mx.float32)
    zd = z - z0.astype(mx.float32)
    
    # Get corner values
    if has_channels:
        # For multi-channel volumes
        c000 = volume[y0, x0, z0]
        c001 = volume[y0, x0, z1]
        c010 = volume[y0, x1, z0]
        c011 = volume[y0, x1, z1]
        c100 = volume[y1, x0, z0]
        c101 = volume[y1, x0, z1]
        c110 = volume[y1, x1, z0]
        c111 = volume[y1, x1, z1]
    else:
        # For single channel volumes
        c000 = volume[y0, x0, z0]
        c001 = volume[y0, x0, z1]
        c010 = volume[y0, x1, z0]
        c011 = volume[y0, x1, z1]
        c100 = volume[y1, x0, z0]
        c101 = volume[y1, x0, z1]
        c110 = volume[y1, x1, z0]
        c111 = volume[y1, x1, z1]
    
    # Interpolate along x
    c00 = c000 * (1 - xd)[..., None] + c010 * xd[..., None] if has_channels else c000 * (1 - xd) + c010 * xd
    c01 = c001 * (1 - xd)[..., None] + c011 * xd[..., None] if has_channels else c001 * (1 - xd) + c011 * xd
    c10 = c100 * (1 - xd)[..., None] + c110 * xd[..., None] if has_channels else c100 * (1 - xd) + c110 * xd
    c11 = c101 * (1 - xd)[..., None] + c111 * xd[..., None] if has_channels else c101 * (1 - xd) + c111 * xd
    
    # Interpolate along y
    c0 = c00 * (1 - yd)[..., None] + c10 * yd[..., None] if has_channels else c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd)[..., None] + c11 * yd[..., None] if has_channels else c01 * (1 - yd) + c11 * yd
    
    # Interpolate along z
    interp = c0 * (1 - zd)[..., None] + c1 * zd[..., None] if has_channels else c0 * (1 - zd) + c1 * zd
    
    return interp


def grid_sample_3d(volume: mx.array, grid: mx.array, mode: str = "bilinear") -> mx.array:
    """
    Sample from a 3D volume using a coordinate grid.
    Similar to torch's grid_sample function but for MLX.
    
    Args:
        volume: Input volume of shape (H, W, D) or (H, W, D, C)
        grid: Sampling grid of shape (H', W', D', 3), each value in range [-1, 1]
        mode: Interpolation mode, 'bilinear' or 'nearest'
        
    Returns:
        Resampled volume of shape (H', W', D') or (H', W', D', C)
    """
    # Get shapes
    H, W, D = volume.shape[:3]
    has_channels = volume.ndim > 3
    
    # Convert normalized coordinates [-1, 1] to volume coordinates [0, H-1], [0, W-1], [0, D-1]
    y = ((grid[..., 0] + 1) / 2) * (H - 1)
    x = ((grid[..., 1] + 1) / 2) * (W - 1)
    z = ((grid[..., 2] + 1) / 2) * (D - 1)
    
    # Stack coordinates
    coords = mx.stack([y, x, z], axis=-1)
    
    if mode == "bilinear":
        return trilinear_interpolate(volume, coords)
    elif mode == "nearest":
        # Nearest neighbor interpolation
        y_nn = mx.round(y).astype(mx.int32)
        x_nn = mx.round(x).astype(mx.int32)
        z_nn = mx.round(z).astype(mx.int32)
        
        # Clip to valid range
        y_nn = mx.clip(y_nn, 0, H - 1)
        x_nn = mx.clip(x_nn, 0, W - 1)
        z_nn = mx.clip(z_nn, 0, D - 1)
        
        return volume[y_nn, x_nn, z_nn]
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")


def warp_volume(volume: mx.array, displacement_field: mx.array, mode: str = "bilinear") -> mx.array:
    """
    Warp a volume using a displacement field.
    
    Args:
        volume: Input volume of shape (H, W, D) or (H, W, D, C)
        displacement_field: Displacement field of shape (H, W, D, 3)
        mode: Interpolation mode, 'bilinear' or 'nearest'
        
    Returns:
        Warped volume of the same shape as input
    """
    # Get volume shape
    H, W, D = volume.shape[:3]
    
    # Create base coordinate grid
    y, x, z = mx.meshgrid(
        mx.arange(H), 
        mx.arange(W), 
        mx.arange(D),
        indexing='ij'
    )
    
    # Calculate sampling coordinates
    y_sample = y + displacement_field[..., 0]
    x_sample = x + displacement_field[..., 1]
    z_sample = z + displacement_field[..., 2]
    
    # Stack into coordinates tensor
    coords = mx.stack([y_sample, x_sample, z_sample], axis=-1)
    
    # Perform interpolation
    if mode == "bilinear":
        return trilinear_interpolate(volume, coords)
    elif mode == "nearest":
        # Round coordinates for nearest neighbor
        y_nn = mx.round(y_sample).astype(mx.int32)
        x_nn = mx.round(x_sample).astype(mx.int32)
        z_nn = mx.round(z_sample).astype(mx.int32)
        
        # Clip to valid range
        y_nn = mx.clip(y_nn, 0, H - 1)
        x_nn = mx.clip(x_nn, 0, W - 1)
        z_nn = mx.clip(z_nn, 0, D - 1)
        
        return volume[y_nn, x_nn, z_nn]
    else:
        raise ValueError(f"Unsupported interpolation mode: {mode}")
