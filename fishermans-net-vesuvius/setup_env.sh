#!/bin/bash
# Setup script for Fisherman's Net Vesuvius Warping project
# Uses uv as the package manager as specified

set -e  # Exit on error

echo "🌊 Setting up Fisherman's Net Vesuvius Warping environment"
echo "=========================================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found! Please install uv first:"
    echo "    curl -sSf https://install.python-uv.org | python3"
    exit 1
fi

echo "✅ uv is installed"

# Create and activate virtual environment
echo "📦 Creating virtual environment with uv..."
uv venv

# Install dependencies
echo "📥 Installing dependencies with uv..."
uv pip install -e .

# Verify MLX installation
echo "🔍 Verifying MLX installation..."
python -c "import mlx.core; print(f'MLX version: {mlx.__version__}')"

echo "📊 Installing additional packages for visualization and notebooks..."
uv pip install jupyter matplotlib tqdm

echo "✨ Environment setup complete! ✨"
echo ""
echo "To activate the environment, run:"
echo "    source .venv/bin/activate"  # uv creates a .venv folder by default
echo ""
echo "To test the installation:"
echo "    python examples/simple_warp.py"
echo ""
echo "To generate test data:"
echo "    python scripts/generate_test_data.py"
echo ""
echo "To run the benchmark:"
echo "    python scripts/benchmark.py"
