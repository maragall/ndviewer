#!/usr/bin/env python3
"""
Downsampler module for creating fast preview mosaics.
Provides a lightweight navigator view for large acquisitions.
"""

import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Callable
import numpy as np
import tifffile as tf
from PIL import Image
import threading

# Pattern for acquisitions: {region}_{fov}_{z}_Fluorescence_{wavelength}_nm_Ex.tiff
FPATTERN = re.compile(
    r"([^_]+)_(\d+)_(\d+)_Fluorescence_(\d+)_nm_Ex\.tiff?", re.IGNORECASE
)

class DownsampledNavigator:
    """Fast downsampled mosaic generator for navigation."""
    
    def __init__(self, acquisition_dir: Path, tile_size: int = 50, 
                 cache_enabled: bool = True, progress_callback: Optional[Callable] = None):
        """
        Initialize the navigator.
        
        Parameters:
        -----------
        acquisition_dir : Path
            Root directory of the acquisition
        tile_size : int
            Size of each tile in pixels (smaller = faster)
        cache_enabled : bool
            Whether to use caching
        progress_callback : Callable
            Function to call with progress updates (percent, message)
        """
        self.acquisition_dir = Path(acquisition_dir)
        self.tile_size = tile_size
        self.cache_enabled = cache_enabled
        self.progress_callback = progress_callback
        
        # Data storage
        self.coordinates = {}  # {fov: (x_mm, y_mm)}
        self.regions = {}  # {fov: region}
        self.channels = []
        self.file_map = {}  # {(channel, region, fov): filepath}
        self.fov_grid = {}  # {(row, col): (region, fov)}
        self.grid_dims = (0, 0)
        self.grid_bounds = None  # (x_min, x_max, y_min, y_max) in mm
        
        # Cache directory
        if self.cache_enabled:
            self.cache_dir = self.acquisition_dir / "cache"
            self.cache_dir.mkdir(exist_ok=True)
        else:
            self.cache_dir = None
            
    def create_mosaic(self, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Create a downsampled mosaic for the specified timepoint.
        
        Returns:
        --------
        mosaic : np.ndarray
            The mosaic image as a numpy array
        metadata : dict
            Metadata including grid info and coordinate mappings
        """
        # Find timepoint directory
        timepoint_dir = self.acquisition_dir / str(timepoint)
        if not timepoint_dir.exists():
            raise ValueError(f"Timepoint directory {timepoint_dir} not found")
            
        # Load coordinates
        self._report_progress(5, "Loading coordinates...")
        self._load_coordinates(timepoint_dir)
        
        # Scan files
        self._report_progress(10, "Scanning files...")
        self._scan_files(timepoint_dir)
        
        # Build grid
        self._report_progress(15, "Building grid structure...")
        self._build_grid()
        
        # Create mosaic
        self._report_progress(20, "Creating mosaic...")
        mosaic_array = self._create_mosaic_array()
        
        # Build metadata
        metadata = {
            'grid_dims': self.grid_dims,
            'tile_size': self.tile_size,
            'fov_grid': self.fov_grid,
            'coordinates': self.coordinates,
            'regions': self.regions,
            'channels': self.channels,
            'grid_bounds': self.grid_bounds,
            'pixel_to_mm_scale': self._calculate_pixel_to_mm_scale()
        }
        
        self._report_progress(100, "Complete!")
        return mosaic_array, metadata
    
    def _report_progress(self, percent: int, message: str):
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(percent, message)
    
    def _load_coordinates(self, timepoint_dir: Path):
        """Load FOV coordinates and regions from CSV."""
        coord_file = timepoint_dir / "coordinates.csv"
        if not coord_file.exists():
            raise FileNotFoundError(f"coordinates.csv not found in {timepoint_dir}")
            
        self.coordinates.clear()
        self.regions.clear()
        
        with open(coord_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fov = int(row['fov'])
                x_mm = float(row['x (mm)'])
                y_mm = float(row['y (mm)'])
                region = row.get('region', 'default')
                
                self.coordinates[fov] = (x_mm, y_mm)
                self.regions[fov] = region
    
    def _scan_files(self, timepoint_dir: Path):
        """Scan directory for TIFF files."""
        self.file_map.clear()
        self.channels = set()
        
        tiff_files = list(timepoint_dir.glob("*.tif")) + list(timepoint_dir.glob("*.tiff"))
        
        for filepath in tiff_files:
            match = FPATTERN.match(filepath.name)
            if not match:
                continue
                
            region, fov, z, wavelength = match.groups()
            fov = int(fov)
            channel = f"{wavelength}nm"
            
            # Only include FOVs that have coordinates
            if fov not in self.coordinates:
                continue
                
            self.channels.add(channel)
            key = (channel, region, fov)
            
            if key not in self.file_map:
                self.file_map[key] = []
            self.file_map[key].append(filepath)
        
        self.channels = sorted(list(self.channels))
    
    def _build_grid(self):
        """Build grid structure from coordinates."""
        if not self.coordinates:
            return
            
        # Get coordinate bounds
        x_coords = [c[0] for c in self.coordinates.values()]
        y_coords = [c[1] for c in self.coordinates.values()]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        self.grid_bounds = (x_min, x_max, y_min, y_max)
        
        # Get unique positions
        x_positions = sorted(set(x_coords))
        y_positions = sorted(set(y_coords))
        
        n_cols = len(x_positions)
        n_rows = len(y_positions)
        self.grid_dims = (n_rows, n_cols)
        
        # Create position mappings
        tolerance = 0.001  # 1 micron
        x_to_col = {x: i for i, x in enumerate(x_positions)}
        y_to_row = {y: i for i, y in enumerate(y_positions)}
        
        # Build FOV grid
        self.fov_grid.clear()
        
        for fov, (x_mm, y_mm) in self.coordinates.items():
            # Find closest position
            col = None
            row = None
            
            for x_pos, idx in x_to_col.items():
                if abs(x_pos - x_mm) < tolerance:
                    col = idx
                    break
                    
            for y_pos, idx in y_to_row.items():
                if abs(y_pos - y_mm) < tolerance:
                    row = idx
                    break
                    
            if col is not None and row is not None:
                region = self.regions.get(fov, 'default')
                self.fov_grid[(row, col)] = (region, fov)
    
    def _calculate_pixel_to_mm_scale(self) -> Tuple[float, float]:
        """Calculate mm per pixel for the mosaic."""
        if not self.grid_bounds or not self.grid_dims:
            return (1.0, 1.0)
            
        x_min, x_max, y_min, y_max = self.grid_bounds
        n_rows, n_cols = self.grid_dims
        
        # Calculate mm per tile
        mm_per_tile_x = (x_max - x_min) / (n_cols - 1) if n_cols > 1 else 1.0
        mm_per_tile_y = (y_max - y_min) / (n_rows - 1) if n_rows > 1 else 1.0
        
        # Convert to mm per pixel
        mm_per_pixel_x = mm_per_tile_x / self.tile_size
        mm_per_pixel_y = mm_per_tile_y / self.tile_size
        
        return (mm_per_pixel_x, mm_per_pixel_y)
    
    def _create_mosaic_array(self) -> np.ndarray:
        """Create the mosaic array."""
        n_rows, n_cols = self.grid_dims
        mosaic = np.zeros((n_rows * self.tile_size, n_cols * self.tile_size), dtype=np.uint8)
        
        total_tiles = len(self.fov_grid)
        processed = 0
        
        for (row, col), (region, fov) in self.fov_grid.items():
            processed += 1
            if processed % 10 == 0:
                progress = 20 + int(80 * processed / total_tiles)
                self._report_progress(progress, f"Processing tile {processed}/{total_tiles}")
            
            # Get tile image
            tile_img = self._get_tile_image(region, fov)
            if tile_img is None:
                continue
                
            # Place in mosaic
            y_start = row * self.tile_size
            x_start = col * self.tile_size
            mosaic[y_start:y_start + self.tile_size, 
                   x_start:x_start + self.tile_size] = tile_img
        
        return mosaic
    
    def _get_tile_image(self, region: str, fov: int) -> Optional[np.ndarray]:
        """Get or generate a tile image."""
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"nav_{region}_{fov}_{self.tile_size}.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path)
                except:
                    pass  # Fall through to regenerate
        
        # Generate tile
        tile_img = self._generate_tile(region, fov)
        
        # Save to cache
        if tile_img is not None and self.cache_dir:
            cache_path = self.cache_dir / f"nav_{region}_{fov}_{self.tile_size}.npy"
            np.save(cache_path, tile_img)
            
        return tile_img
    
    def _generate_tile(self, region: str, fov: int) -> Optional[np.ndarray]:
        """Generate a tile using the brightest channel approach."""
        best_mean = -1
        best_image = None
        
        # Quick scan for brightest channel
        for channel in self.channels:
            key = (channel, region, fov)
            if key not in self.file_map:
                continue
                
            # Get middle z file
            files = self.file_map[key]
            if not files:
                continue
                
            # Sort by z and get middle
            z_files = []
            for f in files:
                match = FPATTERN.match(f.name)
                if match:
                    z = int(match.group(3))
                    z_files.append((z, f))
            
            if not z_files:
                continue
                
            z_files.sort()
            mid_file = z_files[len(z_files) // 2][1]
            
            try:
                # Quick sample for mean
                img = tf.imread(mid_file, aszarr=True)
                mean_val = np.mean(img[::20, ::20])
                
                if mean_val > best_mean:
                    best_mean = mean_val
                    best_image = tf.imread(mid_file)
            except:
                continue
        
        if best_image is None:
            return None
            
        # Convert to 8-bit
        if best_image.dtype == np.uint16:
            img_8bit = (best_image >> 8).astype(np.uint8)
        else:
            img_min, img_max = best_image.min(), best_image.max()
            if img_max > img_min:
                img_8bit = ((best_image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_8bit = np.zeros_like(best_image, dtype=np.uint8)
        
        # Create thumbnail
        pil_img = Image.fromarray(img_8bit)
        pil_img.thumbnail((self.tile_size, self.tile_size), Image.Resampling.NEAREST)
        
        return np.array(pil_img)
    
    def pixel_to_mm(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert mosaic pixel coordinates to mm coordinates."""
        if not self.grid_bounds:
            return (0.0, 0.0)
            
        x_min, _, y_min, _ = self.grid_bounds
        mm_per_pixel_x, mm_per_pixel_y = self._calculate_pixel_to_mm_scale()
        
        x_mm = x_min + pixel_x * mm_per_pixel_x
        y_mm = y_min + pixel_y * mm_per_pixel_y
        
        return (x_mm, y_mm)
    
    def mm_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Convert mm coordinates to mosaic pixel coordinates."""
        if not self.grid_bounds:
            return (0, 0)
            
        x_min, _, y_min, _ = self.grid_bounds
        mm_per_pixel_x, mm_per_pixel_y = self._calculate_pixel_to_mm_scale()
        
        pixel_x = int((x_mm - x_min) / mm_per_pixel_x)
        pixel_y = int((y_mm - y_min) / mm_per_pixel_y)
        
        return (pixel_x, pixel_y)