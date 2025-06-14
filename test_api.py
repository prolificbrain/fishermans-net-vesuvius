#!/usr/bin/env python3
"""
Test script for investigating the vesuvius library API and accessing real data.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import vesuvius

# Try to access prebaked examples or direct URLs that might be available
def explore_available_data():
    """
    Explore what data is available via the vesuvius API.
    """
    print("\n=== Vesuvius API Exploration ===")
    print(f"Vesuvius module located at: {vesuvius.__file__}")
    print(f"Available functions/classes: {[name for name in dir(vesuvius) if not name.startswith('_')]}")
    
    try:
        print("\nTrying to list available files...")
        files = vesuvius.list_files()
        print(f"Files: {files}")
    except Exception as e:
        print(f"Error listing files: {e}")
    
    try:
        print("\nTrying to list available cubes...")
        cubes = vesuvius.cubes
        print(f"Cubes: {cubes}")
    except Exception as e:
        print(f"Error listing cubes: {e}")
    
    # Try loading a scroll directly with different variants
    try:
        print("\nTrying to load canonical Scroll1...")
        volume = vesuvius.Volume(type="scroll", scroll_id=1, verbose=True)
        print(f"Volume shape: {volume.shape()}")
        print("Successfully loaded canonical Scroll1!")
        
        # Try accessing a slice
        print("Getting a sample slice...")
        slice_data = volume[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(slice_data, cmap='gray')
        plt.title("Scroll1 - First Slice")
        plt.colorbar()
        plt.savefig("scroll1_first_slice.png")
        print(f"Saved first slice to scroll1_first_slice.png")
        
        return True, volume
        
    except Exception as e:
        print(f"Error loading canonical Scroll1: {e}")
    
    try:
        print("\nTrying to load canonical Scroll1 with specified energy and resolution...")
        volume = vesuvius.Volume(
            type="scroll", 
            scroll_id=1,
            energy=54,
            resolution=7.91,
            verbose=True
        )
        print(f"Volume shape: {volume.shape()}")
        print("Successfully loaded canonical Scroll1 with params!")
        
        # Try accessing a slice
        print("Getting a sample slice...")
        slice_data = volume[0]
        plt.figure(figsize=(10, 10))
        plt.imshow(slice_data, cmap='gray')
        plt.title("Scroll1 - First Slice")
        plt.colorbar()
        plt.savefig("scroll1_first_slice.png")
        print(f"Saved first slice to scroll1_first_slice.png")
        
        return True, volume
        
    except Exception as e:
        print(f"Error loading canonical Scroll1 with params: {e}")
        
    try:
        # Try with a local path
        print("\nTrying with local domain...")
        volume = vesuvius.Volume(
            type="scroll",
            scroll_id=1,
            domain="local",
            path="/tmp",  # Just a placeholder; unlikely to work
            verbose=True
        )
        print(f"Volume shape: {volume.shape()}")
        return True, volume
    except Exception as e:
        print(f"Error loading with local domain: {e}")
        
    return False, None

# Check if there's a connection issue by downloading a test file from the Vesuvius Challenge server
def test_connection():
    """
    Test connection to the Vesuvius Challenge server.
    """
    import requests
    
    print("\n=== Testing Connection to Vesuvius Challenge Server ===")
    
    try:
        # Try a simple GET request to the main domain
        response = requests.get("https://dl.ash2txt.org/", timeout=10)
        print(f"Status code: {response.status_code}")
        
        if response.status_code == 200:
            print("Successfully connected to dl.ash2txt.org")
            return True
        else:
            print(f"Connection issue: Status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Connection error: {e}")
        return False

# Check the required authentication mechanism
def check_authentication():
    """
    Check if there's any authentication setup required.
    """
    print("\n=== Checking Authentication Requirements ===")
    
    # Check if there are any environment variables used by vesuvius
    vesuvius_env_vars = [var for var in os.environ if 'VESUVIUS' in var.upper()]
    if vesuvius_env_vars:
        print(f"Found vesuvius-related environment variables: {vesuvius_env_vars}")
    else:
        print("No vesuvius-related environment variables found")
    
    # Look for any config files in the vesuvius package
    vesuvius_dir = os.path.dirname(vesuvius.__file__)
    print(f"Vesuvius package directory: {vesuvius_dir}")
    
    config_files = []
    for root, dirs, files in os.walk(vesuvius_dir):
        for file in files:
            if file.endswith(('.yaml', '.yml', '.json', '.conf', '.cfg')):
                config_files.append(os.path.join(root, file))
    
    print(f"Found {len(config_files)} config files in vesuvius package")
    for config_file in config_files:
        print(f"  {config_file}")
        
        # If it's small, let's see what's in it
        try:
            if os.path.getsize(config_file) < 10000:  # Only read if less than 10KB
                with open(config_file, 'r') as f:
                    content = f.read()
                print(f"Content of {os.path.basename(config_file)}:")
                print("---")
                print(content[:500] + "..." if len(content) > 500 else content)
                print("---")
        except Exception as e:
            print(f"Error reading file: {e}")

def investigate_volume_class():
    """
    Investigate the Volume class implementation to understand how to use it correctly.
    """
    print("\n=== Investigating Volume Class ===")
    
    # Look at the Volume implementation
    import inspect
    from vesuvius import Volume
    
    # Get the source code of the Volume.__init__ method
    try:
        init_code = inspect.getsource(Volume.__init__)
        print("Volume.__init__ source code:")
        print("---")
        print(init_code)
        print("---")
    except Exception as e:
        print(f"Couldn't get Volume.__init__ source: {e}")
    
    # Get the Volume module
    try:
        volume_module = inspect.getmodule(Volume)
        print(f"Volume class is defined in module: {volume_module.__name__}")
        
        # If the Volume is imported from somewhere else, find the actual implementation
        if volume_module.__name__ != 'vesuvius.volume':
            print(f"Looking for actual implementation...")
            for name, obj in inspect.getmembers(vesuvius):
                if name == 'volume' and inspect.ismodule(obj):
                    print(f"Found volume module at {obj.__name__}")
                    volume_module = obj
                    break
    except Exception as e:
        print(f"Error getting Volume module: {e}")

def main():
    """Main function to run the API exploration."""
    
    # Test connection to the server
    connection_ok = test_connection()
    if not connection_ok:
        print("\nWARNING: Connection to Vesuvius Challenge server failed. Network issues might prevent data access.")
    
    # Check authentication requirements
    check_authentication()
    
    # Investigate the Volume class implementation
    investigate_volume_class()
    
    # Try to access data
    success, volume = explore_available_data()
    
    if success:
        print("\nSUCCESS! Was able to access at least some data via the vesuvius API.")
        print("Use this approach in your test_real_data.py script.")
    else:
        print("\nFAILED to access any data via the vesuvius API.")
        print("You may need:")
        print("1. Network connectivity to dl.ash2txt.org")
        print("2. Authentication credentials or token")
        print("3. Local data files if online access is not possible")

if __name__ == "__main__":
    main()
