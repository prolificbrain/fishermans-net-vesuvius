#!/usr/bin/env python3
"""
Script to download a sample dataset from the Vesuvius Challenge for testing.
This approach downloads data directly from the Vesuvius Challenge publicly available URLs
instead of relying on the vesuvius package's built-in data access mechanisms.
"""
import os
import sys
import requests
import numpy as np
import zarr
import shutil
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Directory to store the downloaded data
DATA_DIR = Path("./vesuvius_data")

def ensure_dir_exists(dir_path):
    """Ensure directory exists."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_file(url, target_path):
    """
    Download file from URL with progress bar.
    """
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Failed to download from {url}, status code: {response.status_code}")
        return False
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(target_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    
    print(f"Downloaded to {target_path}")
    return True

def download_npy_data():
    """
    Download sample fragment data in .npy format, which is simpler to work with.
    These are from the Vesuvius Challenge public dataset.
    """
    # Fragment sample URL
    base_url = "https://github.com/educelab/vesuvius-build-data/raw/main/"
    
    # Create directory for fragment data
    fragment_dir = DATA_DIR / "fragment_data"
    ensure_dir_exists(fragment_dir)
    
    # Fragment data files (these are small numpy arrays of actual scroll data)
    fragment_files = [
        "sample_aves_1.npy",  # Sample fragment volume
        "sample_aves_1_mask.npy",  # Corresponding ink mask
        "sample_aves_1_surface.npy",  # Surface height map
    ]
    
    # Download each file
    for filename in fragment_files:
        url = base_url + filename
        target_path = fragment_dir / filename
        
        if not target_path.exists():
            success = download_file(url, target_path)
            if not success:
                print(f"Failed to download {filename}")
        else:
            print(f"{filename} already exists at {target_path}")
    
    return fragment_dir

def download_zarr_sample(scroll_id=1, segment="20230827161847"):
    """
    Try to download a small subset of a zarr-formatted scroll dataset.
    This is more challenging due to the zarr format, but would be most compatible with the vesuvius library.
    """
    # Unfortunately, direct access to zarr files is more complex
    # This is a placeholder for future implementation if needed
    print("Note: Direct zarr download not implemented yet due to complexity.")
    print("Using NPY samples instead which will work for testing.")
    return None

def load_fragment_data(fragment_dir):
    """
    Load the fragment data and display some basic information.
    """
    # Load the fragment volume
    volume_path = fragment_dir / "sample_aves_1.npy"
    mask_path = fragment_dir / "sample_aves_1_mask.npy"
    surface_path = fragment_dir / "sample_aves_1_surface.npy"
    
    # Check if files exist
    if not all(p.exists() for p in [volume_path, mask_path, surface_path]):
        print("Error: Some fragment files are missing.")
        return None, None, None
    
    # Load the data
    volume = np.load(volume_path)
    mask = np.load(mask_path)
    surface = np.load(surface_path)
    
    # Display info
    print("\nFragment Data Information:")
    print(f"Volume shape: {volume.shape}")
    print(f"Volume data type: {volume.dtype}")
    print(f"Volume value range: [{volume.min()}, {volume.max()}]")
    
    print(f"Mask shape: {mask.shape}")
    print(f"Surface shape: {surface.shape}")
    
    # Save a visualization
    visualize_sample(volume, mask, surface, DATA_DIR / "fragment_visualization.png")
    
    return volume, mask, surface

def visualize_sample(volume, mask, surface, output_path):
    """
    Create a visualization of the sample data.
    """
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Show a slice from the middle of the volume
    mid_slice = volume.shape[0] // 2
    axes[0].imshow(volume[mid_slice], cmap='gray')
    axes[0].set_title(f"Volume Slice {mid_slice}")
    axes[0].axis('off')
    
    # Show the mask if it's 2D, or a slice if it's 3D
    if len(mask.shape) == 2:
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Ink Mask")
    else:
        axes[1].imshow(mask[mid_slice], cmap='gray')
        axes[1].set_title(f"Ink Mask Slice {mid_slice}")
    axes[1].axis('off')
    
    # Show the surface
    axes[2].imshow(surface, cmap='viridis')
    axes[2].set_title("Surface Height Map")
    axes[2].axis('off')
    
    # Add a colorbar to the surface plot
    plt.colorbar(axes[2].imshow(surface, cmap='viridis'), ax=axes[2], label='Height')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Visualization saved to {output_path}")

def convert_to_zarr(volume, output_path):
    """
    Convert numpy array to zarr format for compatibility with vesuvius library.
    """
    # Create zarr array
    z = zarr.create(shape=volume.shape, 
                   chunks=(10, 128, 128), 
                   dtype=volume.dtype,
                   store=output_path)
    
    # Copy data to zarr array
    z[:] = volume
    
    print(f"Created zarr file at {output_path}")
    return output_path

def prepare_data_for_fisherman_net(volume):
    """
    Prepare the data for use with the Fisherman's Net algorithm by converting to MLX format.
    """
    # Since we're dealing with actual data files, we'll import MLX here for the conversion
    import mlx.core as mx
    
    # Convert to MLX array
    volume_mlx = mx.array(volume)
    
    print(f"Converted volume to MLX array with shape {volume_mlx.shape}")
    return volume_mlx

def main():
    # Create base data directory
    ensure_dir_exists(DATA_DIR)
    
    # Download sample fragment data (NPY format)
    fragment_dir = download_npy_data()
    
    # Try to download zarr sample (more challenging)
    zarr_dir = download_zarr_sample()
    
    # Load fragment data
    volume, mask, surface = load_fragment_data(fragment_dir)
    
    # If successful, try converting to zarr for vesuvius library compatibility
    if volume is not None:
        zarr_path = DATA_DIR / "volume.zarr"
        if not os.path.exists(zarr_path):
            convert_to_zarr(volume, zarr_path)
        else:
            print(f"Zarr file already exists at {zarr_path}")
    
    print("\nData preparation complete!")
    print("You can now use this data to test the Fisherman's Net algorithm.")
    print(f"Data directory: {DATA_DIR}")

if __name__ == "__main__":
    main()
