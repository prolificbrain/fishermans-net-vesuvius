"""
I/O utilities for Vesuvius Challenge data formats.

Handles:
- Multi-page TIF volumes
- NPY arrays
- Vesuvius-specific metadata
"""

import logging
from pathlib import Path
from typing import Union, Optional, Dict, Any

import mlx.core as mx
import numpy as np
import tifffile
import json


logger = logging.getLogger(__name__)


def load_volume(path: Union[str, Path]) -> mx.array:
    """
    Load a 3D volume from various formats.
    
    Args:
        path: Path to volume file (TIF, NPY, etc.)
        
    Returns:
        MLX array of shape (H, W, D)
    """
    path = Path(path)
    
    if path.suffix.lower() in ['.tif', '.tiff']:
        # Load multi-page TIF
        logger.info(f"Loading TIF volume from {path}")
        with tifffile.TiffFile(path) as tif:
            volume = tif.asarray()
            
        # Handle different conventions
        if volume.ndim == 2:
            # Single slice
            volume = volume[..., np.newaxis]
        elif volume.ndim == 4:
            # Multi-channel, take first channel
            logger.warning(f"Multi-channel volume detected, using first channel")
            volume = volume[..., 0]
            
    elif path.suffix == '.npy':
        # Load numpy array
        logger.info(f"Loading NPY volume from {path}")
        volume = np.load(path)
        
    elif path.suffix == '.npz':
        # Load compressed numpy
        logger.info(f"Loading NPZ volume from {path}")
        data = np.load(path)
        # Assume 'volume' key or first key
        if 'volume' in data:
            volume = data['volume']
        else:
            key = list(data.keys())[0]
            logger.info(f"Using key '{key}' from NPZ file")
            volume = data[key]
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")
    
    # Convert to MLX array
    volume = mx.array(volume)
    
    # Ensure float32
    if volume.dtype != mx.float32:
        volume = volume.astype(mx.float32)
    
    logger.info(f"Loaded volume with shape {volume.shape}, dtype {volume.dtype}")
    return volume


def save_volume(path: Union[str, Path], volume: mx.array, 
                metadata: Optional[Dict[str, Any]] = None):
    """
    Save a 3D volume to TIF format.
    
    Args:
        path: Output path
        volume: Volume to save
        metadata: Optional metadata to embed
    """
    path = Path(path)
    
    # Convert to numpy
    volume_np = np.array(volume)
    
    # Normalize to uint16 for TIF (common in Vesuvius)
    if volume_np.dtype == np.float32 or volume_np.dtype == np.float64:
        # Scale to 16-bit range
        vmin, vmax = volume_np.min(), volume_np.max()
        if vmax > vmin:
            volume_np = ((volume_np - vmin) / (vmax - vmin) * 65535).astype(np.uint16)
        else:
            volume_np = np.zeros_like(volume_np, dtype=np.uint16)
    
    # Save as multi-page TIF
    logger.info(f"Saving volume to {path}")
    
    # Prepare metadata
    tif_metadata = {}
    if metadata:
        tif_metadata['description'] = json.dumps(metadata)
    
    # Write TIF
    with tifffile.TiffWriter(path, bigtiff=True) as tif:
        for z in range(volume_np.shape[2]):
            tif.write(volume_np[:, :, z], metadata=tif_metadata)
    
    logger.info(f"Saved volume with shape {volume_np.shape} to {path}")


def load_fiber_data(fiber_path: Union[str, Path], 
                   orientation_path: Optional[Union[str, Path]] = None) -> Dict[str, mx.array]:
    """
    Load fiber volume and orientations.
    
    Args:
        fiber_path: Path to fiber confidence volume
        orientation_path: Optional path to fiber orientations
        
    Returns:
        Dictionary with 'volume' and 'orientations' keys
    """
    result = {}
    
    # Load fiber volume
    result['volume'] = load_volume(fiber_path)
    
    # Load or generate orientations
    if orientation_path:
        if Path(orientation_path).exists():
            logger.info(f"Loading fiber orientations from {orientation_path}")
            orientations = np.load(orientation_path)
            result['orientations'] = mx.array(orientations)
        else:
            logger.warning(f"Orientation file not found: {orientation_path}")
            result['orientations'] = generate_synthetic_orientations(result['volume'])
    else:
        logger.info("Generating synthetic fiber orientations")
        result['orientations'] = generate_synthetic_orientations(result['volume'])
    
    return result


def generate_synthetic_orientations(fiber_volume: mx.array) -> mx.array:
    """
    Generate synthetic fiber orientations from volume gradients.
    """
    # Compute gradients
    dx = mx.diff(fiber_volume, axis=0, prepend=fiber_volume[0:1])
    dy = mx.diff(fiber_volume, axis=1, prepend=fiber_volume[:, 0:1])
    dz = mx.diff(fiber_volume, axis=2, prepend=fiber_volume[:, :, 0:1])
    
    # Stack to create orientation field
    orientations = mx.stack([dx, dy, dz], axis=-1)
    
    # Normalize
    magnitude = mx.sqrt(mx.sum(orientations**2, axis=-1, keepdims=True))
    orientations = orientations / (magnitude + 1e-6)
    
    return orientations


def load_vesuvius_segment(segment_id: str, base_path: Union[str, Path]) -> Dict[str, mx.array]:
    """
    Load a complete Vesuvius segment with all associated data.
    
    Args:
        segment_id: Segment identifier (e.g., '20230205180739')
        base_path: Base directory containing segment data
        
    Returns:
        Dictionary with 'volume', 'mask', 'fibers', etc.
    """
    base_path = Path(base_path)
    segment_path = base_path / segment_id
    
    if not segment_path.exists():
        raise ValueError(f"Segment path not found: {segment_path}")
    
    result = {}
    
    # Load main volume
    volume_path = segment_path / f"{segment_id}_volume.tif"
    if volume_path.exists():
        result['volume'] = load_volume(volume_path)
    
    # Load mask
    mask_path = segment_path / f"{segment_id}_mask.tif"
    if mask_path.exists():
        result['mask'] = load_volume(mask_path) > 0.5
    
    # Load fiber predictions if available
    fiber_path = segment_path / f"{segment_id}_fibers.tif"
    if fiber_path.exists():
        fiber_data = load_fiber_data(fiber_path)
        result.update(fiber_data)
    
    # Load any metadata
    meta_path = segment_path / f"{segment_id}_meta.json"
    if meta_path.exists():
        with open(meta_path, 'r') as f:
            result['metadata'] = json.load(f)
    
    logger.info(f"Loaded segment {segment_id} with keys: {list(result.keys())}")
    return result


def save_warping_result(result: Dict[str, Any], 
                       output_dir: Union[str, Path],
                       prefix: str = "warped"):
    """
    Save complete warping results including checkpoints.
    
    Args:
        result: Warping result dictionary
        output_dir: Output directory
        prefix: Filename prefix
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save warped volume
    volume_path = output_dir / f"{prefix}_volume.tif"
    save_volume(volume_path, result['warped_volume'])
    
    # Save deformation field
    deform_path = output_dir / f"{prefix}_deformation.npy"
    np.save(deform_path, np.array(result['deformation_field']))
    
    # Save metrics
    metrics_path = output_dir / f"{prefix}_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(result['metrics'], f, indent=2)
    
    # Save checkpoints if present
    if 'checkpoints' in result:
        checkpoint_dir = output_dir / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        for checkpoint in result['checkpoints']:
            iter_num = checkpoint['iteration']
            cp_path = checkpoint_dir / f"checkpoint_{iter_num:04d}.npz"
            np.savez_compressed(
                cp_path,
                deformation=np.array(checkpoint['deformation_field']),
                metrics=checkpoint['metrics']
            )
    
    logger.info(f"Saved warping results to {output_dir}")


class VesuviusDataLoader:
    """
    Convenience class for loading Vesuvius data in batches.
    """
    
    def __init__(self, data_dir: Union[str, Path]):
        self.data_dir = Path(data_dir)
        self.segments = list(self.data_dir.glob("*/"))
        logger.info(f"Found {len(self.segments)} segments in {data_dir}")
    
    def load_segment(self, idx: int) -> Dict[str, mx.array]:
        """Load a segment by index."""
        if idx >= len(self.segments):
            raise IndexError(f"Segment index {idx} out of range")
        
        segment_path = self.segments[idx]
        return load_vesuvius_segment(segment_path.name, self.data_dir)
    
    def iterate_segments(self):
        """Iterate over all segments."""
        for segment_path in self.segments:
            yield load_vesuvius_segment(segment_path.name, self.data_dir)
