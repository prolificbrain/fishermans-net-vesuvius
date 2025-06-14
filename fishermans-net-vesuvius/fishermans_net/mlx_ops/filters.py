"""
Advanced filtering operations for 3D volumes using MLX.

Implementations of:
- 3D Gaussian blur
- Sobel edge detection
- Laplacian filter
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from typing import Tuple, Union, Optional


def create_gaussian_kernel_3d(sigma: float, kernel_size: Optional[int] = None) -> mx.array:
    """
    Create a 3D Gaussian kernel for filtering.
    
    Args:
        sigma: Standard deviation of the Gaussian
        kernel_size: Size of the kernel (will be calculated based on sigma if None)
        
    Returns:
        3D Gaussian kernel as MLX array
    """
    # Calculate kernel size if not provided (make sure it's odd)
    if kernel_size is None:
        kernel_size = max(3, int(2 * np.ceil(3 * sigma) + 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create coordinate grid
    half_size = kernel_size // 2
    x, y, z = mx.meshgrid(
        mx.arange(-half_size, half_size + 1),
        mx.arange(-half_size, half_size + 1),
        mx.arange(-half_size, half_size + 1),
        indexing='ij'
    )
    
    # Calculate Gaussian values
    gauss = mx.exp(-(x**2 + y**2 + z**2) / (2 * sigma**2))
    
    # Normalize
    gauss = gauss / mx.sum(gauss)
    
    # Reshape for convolution (C_out, C_in, H, W, D)
    kernel = gauss.reshape(1, 1, kernel_size, kernel_size, kernel_size)
    
    return kernel


def gaussian_blur_3d(volume: mx.array, sigma: float, kernel_size: Optional[int] = None) -> mx.array:
    """
    Apply 3D Gaussian blur to a volume using MLX convolution.
    
    Args:
        volume: Input volume (3D array)
        sigma: Standard deviation of the Gaussian
        kernel_size: Size of the kernel (calculated from sigma if None)
        
    Returns:
        Blurred volume
    """
    # For simplicity in MLX 0.26.1, we'll implement a simple separable Gaussian blur
    # using numpy and then transfer back to MLX
    
    # Convert to numpy for processing
    np_volume = np.array(volume)
    
    # Calculate kernel size if not provided
    if kernel_size is None:
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Create 1D Gaussian kernel
    half_size = kernel_size // 2
    x = np.arange(-half_size, half_size + 1)
    kernel_1d = np.exp(-(x**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    # Apply separable Gaussian blur using scipy
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(np_volume, sigma=sigma, truncate=3.0)
    
    # Convert back to MLX
    return mx.array(blurred)


def separable_gaussian_blur_3d(volume: mx.array, sigma: float) -> mx.array:
    """
    Apply 3D Gaussian blur using separable 1D convolutions (more efficient).
    
    Args:
        volume: Input volume (3D array)
        sigma: Standard deviation of the Gaussian
        
    Returns:
        Blurred volume
    """
    # For simplicity and efficiency in MLX 0.26.1 environment, we'll use scipy's
    # optimized separable gaussian filter and then convert back to MLX
    
    # Convert to numpy for processing
    np_volume = np.array(volume)
    
    # Apply separable Gaussian blur using scipy (which uses separable 1D convolutions internally)
    from scipy.ndimage import gaussian_filter
    blurred = gaussian_filter(np_volume, sigma=sigma, truncate=3.0)
    
    # Convert back to MLX
    return mx.array(blurred)


def sobel_filter_3d(volume: mx.array) -> Tuple[mx.array, mx.array, mx.array]:
    """
    Apply 3D Sobel filter to detect edges in a volume.
    
    Args:
        volume: Input volume
        
    Returns:
        Tuple of gradients (dx, dy, dz)
    """
    # Convert to numpy for processing
    np_volume = np.array(volume)
    
    # Use scipy's filters for computing gradients
    from scipy.ndimage import sobel
    
    # Apply Sobel filter along each axis
    dx = sobel(np_volume, axis=0)
    dy = sobel(np_volume, axis=1)
    dz = sobel(np_volume, axis=2)
    
    # Convert back to MLX arrays
    return mx.array(dx), mx.array(dy), mx.array(dz)


def laplacian_filter_3d(volume: mx.array) -> mx.array:
    """
    Apply 3D Laplacian filter to detect edges/features.
    
    Args:
        volume: Input volume
        
    Returns:
        Filtered volume
    """
    # Convert to numpy for processing
    np_volume = np.array(volume)
    
    # Use scipy's laplace filter
    from scipy.ndimage import laplace
    lap = laplace(np_volume)
    
    # Convert back to MLX
    return mx.array(lap)


def chunk_process_volume(
    volume: mx.array,
    filter_func: callable,
    chunk_size: Tuple[int, int, int] = (128, 128, 64),
    overlap: int = 16,
    **kwargs
) -> mx.array:
    """
    Process a large volume in chunks with overlapping boundaries.
    
    Args:
        volume: Input volume
        filter_func: Function to apply to each chunk
        chunk_size: Size of each chunk (H, W, D)
        overlap: Overlap between chunks
        **kwargs: Additional arguments for filter_func
        
    Returns:
        Processed volume
    """
    H, W, D = volume.shape
    ch, cw, cd = chunk_size
    
    # Initialize output volume
    output = mx.zeros_like(volume)
    
    # Initialize weight volume for blending
    weights = mx.zeros_like(volume)
    
    # Process chunks
    for z in range(0, D, cd - 2 * overlap):
        for y in range(0, H, ch - 2 * overlap):
            for x in range(0, W, cw - 2 * overlap):
                # Calculate chunk boundaries with overlap
                z_start = max(0, z - overlap)
                y_start = max(0, y - overlap)
                x_start = max(0, x - overlap)
                
                z_end = min(D, z + cd + overlap)
                y_end = min(H, y + ch + overlap)
                x_end = min(W, x + cw + overlap)
                
                # Extract chunk
                chunk = volume[y_start:y_end, x_start:x_end, z_start:z_end]
                
                # Process chunk
                processed = filter_func(chunk, **kwargs)
                
                # Create weight mask for blending
                # Higher weight in center, lower at edges
                weight_y = np.ones(y_end - y_start)
                weight_x = np.ones(x_end - x_start)
                weight_z = np.ones(z_end - z_start)
                
                # Apply tapering at boundaries
                if y_start > 0:
                    weight_y[:overlap] = np.linspace(0, 1, overlap)
                if y_end < H:
                    weight_y[-overlap:] = np.linspace(1, 0, overlap)
                    
                if x_start > 0:
                    weight_x[:overlap] = np.linspace(0, 1, overlap)
                if x_end < W:
                    weight_x[-overlap:] = np.linspace(1, 0, overlap)
                    
                if z_start > 0:
                    weight_z[:overlap] = np.linspace(0, 1, overlap)
                if z_end < D:
                    weight_z[-overlap:] = np.linspace(1, 0, overlap)
                
                # Create 3D weight grid
                wy, wx, wz = np.meshgrid(weight_y, weight_x, weight_z, indexing='ij')
                weight_grid = mx.array(wy * wx * wz)
                
                # Add weighted result to output
                output[y_start:y_end, x_start:x_end, z_start:z_end] += processed * weight_grid
                weights[y_start:y_end, x_start:x_end, z_start:z_end] += weight_grid
    
    # Normalize by weights
    output = output / (weights + 1e-10)
    
    return output
