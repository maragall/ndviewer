#!/bin/bash
set -e

echo "Installing system packages..."
sudo apt update
sudo apt install -y mesa-utils libgl1-mesa-glx libgl1-mesa-dri python3-pip

echo "Installing Python packages with system pip..."
pip3 install --user vispy PyQt6 tifffile zarr pandas numpy numcodecs dask napari

echo "Setup complete! Use system Python:"
echo "  python3 napari_launcher.py"
