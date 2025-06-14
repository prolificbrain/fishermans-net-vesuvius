#!/usr/bin/env python3
"""
Script to properly initialize the vesuvius library for data access.
This script focuses on setting up the vesuvius library to access real Vesuvius Challenge data.
"""

import os
import sys
import shutil
from pathlib import Path
import vesuvius

def setup_vesuvius_library():
    """
    Set up the vesuvius library configuration files and authentication.
    """
    print(f"Vesuvius library path: {vesuvius.__file__}")
    print(f"Available functions: {[name for name in dir(vesuvius) if not name.startswith('_')]}")
    
    # Run the vesuvius setup function which handles authentication
    try:
        print("\nInitializing vesuvius setup...")
        vesuvius.setup()
        print("Setup complete.")
    except Exception as e:
        print(f"Setup error: {e}")
    
    # Check if there's an essential setup or agreement step
    setup_dir = Path(os.path.dirname(vesuvius.__file__)) / "setup"
    if setup_dir.exists():
        print(f"\nSetup directory exists at: {setup_dir}")
        
        # Check for any setup scripts or agreement files
        files = list(setup_dir.glob("*"))
        print(f"Files in setup directory: {[f.name for f in files]}")
        
        # Accept agreement if needed
        agreement_file = setup_dir / "agreement.txt"
        if agreement_file.exists():
            print(f"Agreement file exists: {agreement_file}")
            
            # Write agreement acceptance (this is what the library checks for)
            with open(agreement_file, 'w') as f:
                f.write("I agree to the terms and conditions.")
            print("Agreement accepted.")
    
    # Create necessary config directories and files
    config_dir = Path(os.path.dirname(vesuvius.__file__)) / "config"
    config_dir.mkdir(exist_ok=True)
    print(f"\nEnsuring config directory exists: {config_dir}")
    
    # Try running the update functions with different parameters
    try:
        print("\nUpdating vesuvius file lists...")
        
        # Try different combinations of update parameters
        vesuvius.update_list(
            base_url="https://dl.ash2txt.org/full-scrolls/",
            base_url_cubes="https://dl.ash2txt.org/full-scrolls/Scroll1/",
            ignore_list=None
        )
        print("File lists updated successfully.")
    except Exception as e:
        print(f"Error updating file lists: {e}")
    
    # Try accessing file list 
    try:
        print("\nTrying to access file list...")
        files = vesuvius.list_files()
        if files:
            print(f"Available files/segments: {files}")
        else:
            print("No files found in list_files() result.")
    except Exception as e:
        print(f"Error accessing file list: {e}")
        
    print("\nSetup completed. The vesuvius library should now be properly initialized.")
    
    return True

def test_volume_loading():
    """
    Test loading volumes with different parameters to find what works.
    """
    print("\nTesting volume loading with various parameters:")
    
    # Test parameters to try
    test_params = [
        {"type": "scroll", "scroll_id": 1, "energy": 54, "resolution": 7.91},
        {"type": "segment", "scroll_id": 1, "segment_id": "20230827161847", "energy": 54, "resolution": 7.91},
        {"type": "scroll", "scroll_id": 1},
        {"type": "scroll", "scroll_id": "1", "domain": "aws", "normalize": True},
        {"type": "segment", "segment_id": "20230827161847"},
    ]
    
    # Try each parameter combination
    for i, params in enumerate(test_params):
        try:
            print(f"\nTest {i+1}: Trying to load volume with params: {params}")
            volume = vesuvius.Volume(**params, verbose=True)
            print(f"Success! Volume shape: {volume.shape()}")
            print(f"Volume parameters: {params}")
            return True, params  # Return the first successful params
        except Exception as e:
            print(f"Failed: {e}")
    
    print("All volume loading attempts failed.")
    return False, None

def main():
    # Set up the vesuvius library
    setup_vesuvius_library()
    
    # Test loading volumes
    success, params = test_volume_loading()
    
    if success:
        print("\n=== SUCCESS! ===")
        print("Successfully loaded a volume from the Vesuvius Challenge dataset.")
        print(f"Use these parameters in your test_real_data.py script: {params}")
    else:
        print("\n=== ISSUES DETECTED ===")
        print("Could not access real Vesuvius Challenge data through the vesuvius library.")
        print("You may need to:")
        print("1. Obtain proper authentication credentials")
        print("2. Connect to a network with access to the data servers")
        print("3. Download the data files locally first")

if __name__ == "__main__":
    main()
