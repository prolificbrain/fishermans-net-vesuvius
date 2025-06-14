from setuptools import setup, find_packages

setup(
    name="fishermans-net-vesuvius",
    version="0.1.0",
    author="Vesuvius Challenge Team",
    description="Physics-inspired volume warping for Vesuvius Challenge",
    packages=find_packages(),
    install_requires=[
        "mlx>=0.5.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "tifffile>=2023.7.10",
        "matplotlib>=3.7.0",
        "h5py>=3.8.0",
        "scikit-image>=0.20.0",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
