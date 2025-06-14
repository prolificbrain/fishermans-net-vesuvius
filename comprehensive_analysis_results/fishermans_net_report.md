# Fisherman's Net Volume Warping for Vesuvius Challenge

## Algorithm Overview

The Fisherman's Net algorithm treats ancient scroll deformation as a physics problem where:
- Fiber predictions act as 'threads' we can pull
- Physics simulation ensures natural deformation
- Progressive unwrapping preserves papyrus integrity

## Data Analysis

- **Volume Shape**: (25, 827, 2611)
- **Total Voxels**: 53,982,425
- **Value Range**: [0.0, 65535.0]
- **Mean Intensity**: 18158.0

## Results Summary

### Conservative Configuration

- **Critical Fibers Found**: 15
- **Final Flatness Score**: 0.000222
- **Final Strain Energy**: 11.415663
- **Max Deformation**: 0.27 voxels
- **Mean Deformation**: 0.00 voxels
- **Deformed Voxels**: 1,135

### Aggressive Configuration

- **Critical Fibers Found**: 58
- **Final Flatness Score**: 0.000222
- **Final Strain Energy**: 10853.544922
- **Max Deformation**: 11.42 voxels
- **Mean Deformation**: 0.00 voxels
- **Deformed Voxels**: 40,748

### Balanced Configuration

- **Critical Fibers Found**: 30
- **Final Flatness Score**: 0.000222
- **Final Strain Energy**: 637.024597
- **Max Deformation**: 1.72 voxels
- **Mean Deformation**: 0.00 voxels
- **Deformed Voxels**: 13,732

## Key Innovation

The Fisherman's Net approach is unique because it:
1. Uses fiber predictions as physical constraints
2. Applies progressive deformation rather than rigid unwrapping
3. Preserves local papyrus structure while correcting global distortion
4. Scales efficiently to large volumes using NumPy/SciPy

## Next Steps

- Integrate real fiber predictions from segmentation models
- Optimize parameters for different scroll types
- Apply to full scroll volumes
- Validate results with ground truth data
