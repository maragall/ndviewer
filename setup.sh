#!/bin/bash

# Exit on any error
set -e

# Get the conda base directory
CONDA_BASE=$(conda info --base)

# Source conda initialization
source "$CONDA_BASE/etc/profile.d/conda.sh"

echo "Creating conda environment 'ndv' with Python 3.12..."
conda create -n ndv python=3.12 -y

echo "Activating conda environment 'ndv'..."
conda activate ndv

echo "Installing Python packages..."
pip install --no-cache-dir vispy PyQt6 tifffile zarr pandas numpy numcodecs==0.15.1 dask napari

echo "Setup complete! Environment 'ndv' is ready to use."
echo "To activate the environment in the future, run: conda activate ndv"