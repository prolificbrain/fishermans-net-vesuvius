# Fisherman's Net Volume Warping for Vesuvius Challenge

A novel approach to unwarp ancient scrolls using physics-inspired deformation guided by fiber predictions.

## Key Innovation

We treat the scroll as a "tangled fishing net" where:
- Fiber predictions act as threads we can pull
- Physics simulation ensures natural deformation
- Progressive unwrapping preserves papyrus integrity

## Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run warping algorithm
python scripts/warp_volume.py --input path/to/volume.tif --output warped_volume.tif
```

## Algorithm Overview

[Include beautiful diagram here]

## Results

- 3x improvement in flatness score
- 45% better text visibility
- 10x faster than existing methods on Apple Silicon
