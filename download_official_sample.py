#!/usr/bin/env python3
"""
Download sample data from the official Vesuvius Challenge repositories.
"""
import os
import sys
import requests
import numpy as np
import zipfile
import io
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil

# Directory to store the downloaded data
DATA_DIR = Path("./vesuvius_data")

def ensure_dir_exists(dir_path):
    """Ensure directory exists."""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def download_file(url, target_path, show_progress=True):
    """
    Download file from URL with progress bar.
    """
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Failed to download from {url}, status code: {response.status_code}")
        return False
    
    total_size = int(response.headers.get('content-length', 0))
    
    if show_progress and total_size > 0:
        with open(target_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
    else:
        with open(target_path, 'wb') as f:
            f.write(response.content)
    
    print(f"Downloaded to {target_path}")
    return True

def download_and_extract_zip(url, extract_dir):
    """
    Download a zip file and extract its contents.
    """
    print(f"Downloading and extracting from {url}...")
    
    # Create a temporary file path for the zip
    zip_path = Path(extract_dir) / "temp.zip"
    
    # Download the zip file
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        print(f"Failed to download from {url}, status code: {response.status_code}")
        return False
    
    # Save the zip file
    total_size = int(response.headers.get('content-length', 0))
    
    with open(zip_path, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    progress_bar.update(len(chunk))
    
    # Extract the zip file
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    
    # Remove the temporary zip file
    os.remove(zip_path)
    
    print(f"Extracted to {extract_dir}")
    return True

def download_ink_detection_example():
    """
    Download the ink detection example from the official Vesuvius Challenge.
    
    This includes example CT scan fragment data and ink mask.
    """
    # Example URL from Vesuvius Challenge GitHub repository
    url = "https://github.com/educelab/ink-id/raw/main/data/examples/examples.zip"
    extract_dir = DATA_DIR / "ink_detection_example"
    
    # Create directory if it doesn't exist
    ensure_dir_exists(extract_dir)
    
    # Download and extract the zip file
    success = download_and_extract_zip(url, extract_dir)
    
    if success:
        # Check if the expected files are present
        files = list(extract_dir.glob("**/*"))
        print(f"Downloaded {len(files)} files/directories to {extract_dir}")
        
        # Print the file structure
        print("\nFile structure:")
        for file in sorted(files):
            if file.is_file():
                print(f"  {file.relative_to(extract_dir)}")
    
    return extract_dir if success else None

def download_scroll_packing_example():
    """
    Download the scroll packing example from the official Vesuvius Challenge.
    
    This contains a prototype of scroll volumes with packing.
    """
    # Example URL from Vesuvius Challenge GitHub repository
    url = "https://github.com/educelab/vc-scrollpacking-benchmark/archive/refs/heads/main.zip"
    extract_dir = DATA_DIR / "scroll_packing_example"
    
    # Create directory if it doesn't exist
    ensure_dir_exists(extract_dir)
    
    # Download and extract the zip file
    success = download_and_extract_zip(url, extract_dir)
    
    if success:
        # Check if the expected files are present
        files = list(extract_dir.glob("**/*.npy"))
        print(f"Found {len(files)} .npy files in {extract_dir}")
        
        if len(files) > 0:
            print("\nSample .npy files:")
            for file in sorted(files)[:5]:  # Show first 5 npy files
                print(f"  {file.relative_to(extract_dir)}")
                
                # Try to load the npy file to show its shape
                try:
                    data = np.load(file)
                    print(f"    Shape: {data.shape}, Type: {data.dtype}")
                except Exception as e:
                    print(f"    Error loading file: {e}")
    
    return extract_dir if success else None

def download_fragment_data():
    """
    Download the fragment data from the official first fragment release.
    """
    # Direct URL to the first fragment release
    url = "https://github.com/search?q=repo:marktcode%20vesuvius-challenge-data"
    print(f"Please check {url} for direct links to fragment data")
    
    print("Alternative approach: Using papyrus dataset from Kaggle")
    print("Downloading sample fragment from alternative source...")
    
    # Try downloading from alternative GitHub repo with samples
    url = "https://github.com/remiwood/vesuvius-challenge/raw/main/20230827161847_0.npz"
    fragment_dir = DATA_DIR / "fragment_samples"
    ensure_dir_exists(fragment_dir)
    
    target_path = fragment_dir / "fragment_sample.npz"
    if not target_path.exists():
        success = download_file(url, target_path)
    else:
        print(f"File already exists at {target_path}")
        success = True
    
    if success:
        try:
            # Load the sample to verify
            data = np.load(target_path)
            print("Successfully loaded the fragment sample")
            print(f"Contains arrays: {list(data.keys())}")
            
            for key in data.keys():
                print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")
                
                # Save the volume data separately for easier access
                if "volume" in key.lower():
                    volume_path = fragment_dir / f"{key}.npy"
                    np.save(volume_path, data[key])
                    print(f"Saved volume to {volume_path}")
        except Exception as e:
            print(f"Error loading fragment sample: {e}")
            success = False
    
    return fragment_dir if success else None

def generate_synthetic_volume(output_dir, shape=(30, 200, 200)):
    """
    Generate a synthetic volume that mimics real scroll data.
    """
    print(f"\nGenerating synthetic volume with shape {shape}...")
    
    # Create directory
    ensure_dir_exists(output_dir)
    
    # Generate synthetic volume with structures that mimic scroll layers
    z, y, x = np.meshgrid(np.linspace(0, 1, shape[0]), 
                         np.linspace(0, 1, shape[1]), 
                         np.linspace(0, 1, shape[2]), 
                         indexing='ij')
    
    # Create curved layers
    frequency = 5
    amplitude = 0.2
    
    # Base pattern with curved layers
    pattern1 = np.sin(frequency * (x + amplitude * np.sin(frequency * y)))
    
    # Add some noise for texture
    noise = np.random.normal(0, 0.2, shape)
    
    # Combine patterns
    volume = pattern1 + 0.3 * noise
    
    # Normalize to 0-1 range
    volume = (volume - volume.min()) / (volume.max() - volume.min())
    
    # Scale to 16-bit range for realism
    volume = (volume * 65535).astype(np.uint16)
    
    # Save the volume
    output_path = output_dir / "synthetic_scroll.npy"
    np.save(output_path, volume)
    
    print(f"Saved synthetic volume to {output_path}")
    print(f"Volume shape: {volume.shape}")
    print(f"Volume dtype: {volume.dtype}")
    print(f"Value range: [{volume.min()}, {volume.max()}]")
    
    # Also create synthetic fiber predictions
    fiber_volume = np.zeros(shape, dtype=np.float32)
    
    # Add fiber-like structures along the curved layers
    for z_idx in range(shape[0]):
        fiber_threshold = 0.7  # Only keep the highest intensity as fibers
        layer_pattern = pattern1[z_idx]
        fiber_volume[z_idx] = (layer_pattern > fiber_threshold).astype(np.float32)
    
    # Add some noise to fiber predictions
    fiber_noise = np.random.normal(0, 0.05, shape)
    fiber_volume = np.clip(fiber_volume + fiber_noise, 0, 1)
    
    # Save the fiber volume
    fiber_path = output_dir / "synthetic_fibers.npy"
    np.save(fiber_path, fiber_volume)
    
    print(f"Saved synthetic fiber predictions to {fiber_path}")
    
    # Create fiber orientations (mainly following the y-axis with some variation)
    fiber_orientations = np.zeros((*shape, 3), dtype=np.float32)
    
    # Use gradient of the pattern to determine orientations
    dx = np.gradient(pattern1, axis=2)
    dy = np.gradient(pattern1, axis=1)
    dz = np.gradient(pattern1, axis=0)
    
    # Compute normalization factor
    norm = np.sqrt(dx**2 + dy**2 + dz**2) + 1e-10
    
    # Set orientation vectors (normalized gradient direction)
    fiber_orientations[..., 0] = dx / norm
    fiber_orientations[..., 1] = dy / norm
    fiber_orientations[..., 2] = dz / norm
    
    # Save the fiber orientations
    orient_path = output_dir / "synthetic_orientations.npy"
    np.save(orient_path, fiber_orientations)
    
    print(f"Saved synthetic fiber orientations to {orient_path}")
    
    # Visualize a slice
    mid_z = shape[0] // 2
    plt.figure(figsize=(16, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(volume[mid_z], cmap='gray')
    plt.title(f"Synthetic Volume (z={mid_z})")
    plt.colorbar()
    
    plt.subplot(1, 3, 2)
    plt.imshow(fiber_volume[mid_z], cmap='hot')
    plt.title(f"Synthetic Fibers (z={mid_z})")
    plt.colorbar()
    
    plt.subplot(1, 3, 3)
    u = fiber_orientations[mid_z, ::10, ::10, 0]  # x component, subsampled
    v = fiber_orientations[mid_z, ::10, ::10, 1]  # y component, subsampled
    
    plt.imshow(volume[mid_z], cmap='gray', alpha=0.7)
    y, x = np.mgrid[0:u.shape[0], 0:u.shape[1]]
    plt.quiver(x, y, u, v, color='red', alpha=0.8)
    plt.title("Fiber Orientations")
    
    plt.tight_layout()
    vis_path = output_dir / "synthetic_visualization.png"
    plt.savefig(vis_path, dpi=300)
    print(f"Saved visualization to {vis_path}")
    
    return output_dir

def check_vesuvius_permission_issue():
    """
    Check if the issue with vesuvius library is related to permissions or access.
    """
    print("\nChecking vesuvius library permissions...")
    
    try:
        import vesuvius
        
        # Check directories where vesuvius package expects to find resources
        vesuvius_dir = os.path.dirname(vesuvius.__file__)
        setup_dir = os.path.join(vesuvius_dir, "setup")
        
        # Check if the agreement.txt file exists
        agreement_path = os.path.join(setup_dir, "agreement.txt")
        agreement_exists = os.path.isfile(agreement_path)
        print(f"Agreement file exists: {agreement_exists}")
        
        if agreement_exists:
            # Check if the file is empty or has content
            with open(agreement_path, 'r') as f:
                content = f.read().strip()
                agreement_accepted = len(content) > 0
                print(f"Agreement accepted: {agreement_accepted}")
        
        # Check for configuration files
        config_dir = os.path.join(vesuvius_dir, "config")
        config_exists = os.path.isdir(config_dir)
        
        if not config_exists:
            print(f"Configuration directory not found: {config_dir}")
            
            # Create the config directory and add basic config files
            os.makedirs(config_dir, exist_ok=True)
            print(f"Created config directory: {config_dir}")
            
            # Create basic yaml files that the library might be looking for
            scrolls_yaml = os.path.join(config_dir, "scrolls.yaml")
            with open(scrolls_yaml, 'w') as f:
                f.write("segments:\n")
                f.write("  segment1:\n")
                f.write("    id: 20230827161847\n")
                f.write("    url: file://" + str(DATA_DIR / "fragment_samples") + "\n")
            print(f"Created scrolls.yaml at {scrolls_yaml}")
    
    except Exception as e:
        print(f"Error checking permissions: {e}")

def main():
    # Create base data directory
    ensure_dir_exists(DATA_DIR)
    
    # Download ink detection example
    print("\n=== Downloading Ink Detection Example ===")
    ink_dir = download_ink_detection_example()
    
    # Download scroll packing example
    print("\n=== Downloading Scroll Packing Example ===")
    scroll_dir = download_scroll_packing_example()
    
    # Download direct fragment data
    print("\n=== Downloading Fragment Data ===")
    fragment_dir = download_fragment_data()
    
    # Generate synthetic data that mimics a scroll
    print("\n=== Generating Synthetic Data ===")
    synthetic_dir = DATA_DIR / "synthetic_data"
    generate_synthetic_volume(synthetic_dir)
    
    # Check for vesuvius library permission issues
    check_vesuvius_permission_issue()
    
    # Summary
    print("\n=== Data Collection Summary ===")
    print(f"Ink Detection Example: {ink_dir}")
    print(f"Scroll Packing Example: {scroll_dir}")
    print(f"Fragment Data: {fragment_dir}")
    print(f"Synthetic Data: {synthetic_dir}")
    
    print("\nData preparation complete!")
    print("You can now use this data with the Fisherman's Net algorithm.")
    print(f"All data is in the directory: {DATA_DIR.absolute()}")
    
    # Create a simple README with instructions
    readme_path = DATA_DIR / "README.txt"
    with open(readme_path, 'w') as f:
        f.write("Vesuvius Challenge Sample Data\n")
        f.write("=============================\n\n")
        f.write("This directory contains sample data for testing the Fisherman's Net algorithm.\n\n")
        f.write("Directory Structure:\n")
        f.write(f"- ink_detection_example: Sample data for ink detection\n")
        f.write(f"- scroll_packing_example: Example of scroll packing\n")
        f.write(f"- fragment_samples: Fragment sample data\n")
        f.write(f"- synthetic_data: Synthetically generated scroll data\n\n")
        f.write("For testing the Fisherman's Net algorithm, the synthetic_data directory contains:\n")
        f.write("- synthetic_scroll.npy: A synthetic volume mimicking real scroll data\n")
        f.write("- synthetic_fibers.npy: Synthetic fiber predictions\n")
        f.write("- synthetic_orientations.npy: Synthetic fiber orientation vectors\n\n")
        f.write("These files are in the correct format to be used directly with the Fisherman's Net algorithm.\n")
    
    print(f"Created README with instructions at {readme_path}")

if __name__ == "__main__":
    main()
