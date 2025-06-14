#!/usr/bin/env python3
"""
Download and prepare sample data from the Vesuvius Challenge.

This script:
1. Downloads sample scroll fragments from the Vesuvius Challenge
2. Extracts and organizes the data
3. Converts to the format expected by Fisherman's Net
"""

import os
import sys
import argparse
import logging
import urllib.request
import tarfile
import zipfile
import shutil
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


# Vesuvius Challenge sample data URLs
SAMPLE_URLS = {
    "small_fragment": "https://scrollprize.org/data/Vesuvius_Challenge_Scroll_1_Sample_Data.zip",
    "scroll1_segment": "https://scrollprize.org/data/full_scrolls/Scroll1.volpkg.zip",
    "synthetic_scroll": "https://scrollprize.org/data/examples/synthetic_scroll_data.zip"
}


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    """Download a file from URL with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path,
                                  reporthook=t.update_to)


def extract_archive(archive_path, extract_dir):
    """Extract zip or tar archive"""
    print(f"Extracting {archive_path} to {extract_dir}...")
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif archive_path.suffix in ['.tar', '.gz', '.tgz']:
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_dir)
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")
    
    print(f"Extraction complete.")


def convert_vesuvius_to_tif(data_dir, output_dir):
    """Convert Vesuvius Challenge data format to TIF files"""
    from fishermans_net.utils.io import save_volume
    import numpy as np
    import tifffile
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting data from {data_dir} to TIF format in {output_dir}...")
    
    # Find fragment directories
    fragment_dirs = [d for d in data_dir.glob('*') if d.is_dir()]
    
    for fragment_dir in fragment_dirs:
        print(f"Processing fragment: {fragment_dir.name}")
        
        # Check for volume data
        volume_dir = fragment_dir / 'volume'
        if volume_dir.exists():
            # Find all TIFF files
            tiff_files = sorted(list(volume_dir.glob('*.tif')))
            
            if tiff_files:
                print(f"Found {len(tiff_files)} TIFF slices in {volume_dir}")
                
                # Read all slices into a 3D volume
                slices = []
                for tiff_file in tqdm(tiff_files, desc="Reading slices"):
                    slice_data = tifffile.imread(tiff_file)
                    slices.append(slice_data)
                
                # Stack slices into volume
                volume = np.stack(slices, axis=2)
                print(f"Created volume with shape {volume.shape}")
                
                # Save as multi-page TIF
                output_path = output_dir / f"{fragment_dir.name}_volume.tif"
                save_volume(output_path, volume)
                print(f"Saved volume to {output_path}")
        
        # Check for mask data
        mask_dir = fragment_dir / 'mask'
        if mask_dir.exists():
            mask_files = sorted(list(mask_dir.glob('*.tif')))
            if mask_files:
                print(f"Found {len(mask_files)} mask slices")
                
                # Read mask slices
                mask_slices = []
                for mask_file in tqdm(mask_files, desc="Reading masks"):
                    mask_data = tifffile.imread(mask_file)
                    mask_slices.append(mask_data)
                
                # Stack slices into volume
                mask_volume = np.stack(mask_slices, axis=2)
                
                # Save as multi-page TIF
                mask_path = output_dir / f"{fragment_dir.name}_mask.tif"
                save_volume(mask_path, mask_volume)
                print(f"Saved mask to {mask_path}")
    
    print(f"Conversion complete. Data saved to {output_dir}")


def download_and_prepare_data(sample_type, output_dir):
    """Download and prepare Vesuvius sample data"""
    if sample_type not in SAMPLE_URLS:
        raise ValueError(f"Unknown sample type: {sample_type}. Available types: {list(SAMPLE_URLS.keys())}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download URL
    url = SAMPLE_URLS[sample_type]
    archive_path = output_dir / url.split('/')[-1]
    
    print(f"Downloading {url} to {archive_path}...")
    download_url(url, archive_path)
    
    # Extract archive
    extract_dir = output_dir / sample_type
    extract_archive(archive_path, extract_dir)
    
    # Convert to TIF format
    processed_dir = output_dir / f"{sample_type}_processed"
    convert_vesuvius_to_tif(extract_dir, processed_dir)
    
    print(f"\nAll done! Processed data available at: {processed_dir}")
    print("\nTo run the Fisherman's Net algorithm, use:")
    print(f"python scripts/warp_volume.py --input {processed_dir}/fragment_volume.tif --fibers {processed_dir}/fragment_volume.tif --output {output_dir}/warped_result.tif --save-report")
    print("\nNote: For real data, you need fiber predictions. The command above uses the volume itself as placeholder.")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare Vesuvius Challenge sample data")
    parser.add_argument("--type", choices=list(SAMPLE_URLS.keys()), default="small_fragment",
                       help="Type of sample data to download")
    parser.add_argument("--output", "-o", default="./vesuvius_data", 
                       help="Output directory for downloaded data")
    parser.add_argument("--skip-download", action="store_true",
                       help="Skip download and just process existing data")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Download and prepare data
    if args.skip_download:
        # Only process existing data
        extract_dir = Path(args.output) / args.type
        processed_dir = Path(args.output) / f"{args.type}_processed"
        convert_vesuvius_to_tif(extract_dir, processed_dir)
    else:
        download_and_prepare_data(args.type, args.output)


if __name__ == "__main__":
    main()
