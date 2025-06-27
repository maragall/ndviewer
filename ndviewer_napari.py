# =====================================================
# UPDATED DOWNSAMPLER.PY
# =====================================================

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

# Pattern for acquisitions: {region}_{fov}_{z_layer}_{imaging_modality}_{channel_info}_{suffix}.tiff
# Examples: C5_0_0_Fluorescence_405_nm_Ex.tiff, D6_2_3_Brightfield_BF_Ex.tiff
FPATTERN = re.compile(
    r"(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<modality>[^_]+)_(?P<channel>[^_]+)_.*\.tiff?", re.IGNORECASE
)

class DownsampledNavigator:
    """Fast downsampled mosaic generator for navigation."""
    
    def __init__(self, acquisition_dir: Path, tile_size: int = 75,
                 cache_enabled: bool = True, progress_callback: Optional[Callable] = None):
        """
        Initialize the navigator.
        
        Parameters:
        -----------
        acquisition_dir : Path
            Root directory of the acquisition
        tile_size : int
            Size of each tile in pixels (75 for good balance of speed/quality)
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
        self.regions = {}      # {fov: region_name} - Added back for compatibility
        self.channels = []
        self.file_map = {}     # {(channel, fov): [filepath, ...]}
        self.fov_grid = {}     # {(row, col): fov}
        self.grid_dims = (0, 0)
        self.grid_bounds = None  # (x_min, x_max, y_min, y_max) in mm
        
        # Cache directory
        if self.cache_enabled:
            self.cache_dir = self.acquisition_dir / "cache"
            self.cache_dir.mkdir(exist_ok=True)
        else:
            self.cache_dir = None
    
    def create_mosaic_for_region(self, region_name: str, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Create a downsampled mosaic for a specific region.
        
        Parameters:
        -----------
        region_name : str
            Name of the region to create a mosaic for
        timepoint : int
            Timepoint to process
            
        Returns:
        --------
        mosaic : np.ndarray
            The mosaic image as a numpy array
        metadata : dict
            Metadata including grid info and coordinate mappings for this region
        """
        # Find timepoint directory - match grid viewer approach
        timepoint_dirs = []
        for item in self.acquisition_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                timepoint_dirs.append(item)
        
        if not timepoint_dirs:
            raise ValueError(f"No timepoint directories found in {self.acquisition_dir}")
        
        # Use first timepoint if specific one not found
        if timepoint == 0 or not (self.acquisition_dir / str(timepoint)).exists():
            timepoint_dir = sorted(timepoint_dirs, key=lambda x: int(x.name))[0]
        else:
            timepoint_dir = self.acquisition_dir / str(timepoint)
        
        print(f"[NAVIGATOR] Creating region-specific mosaic for region '{region_name}' from {timepoint_dir}")
        
        # Load coordinates
        self._report_progress(5, f"Loading coordinates for region {region_name}...")
        self._load_coordinates(timepoint_dir)
        
        # Scan files
        self._report_progress(10, f"Scanning files for region {region_name}...")
        self._scan_files(timepoint_dir)
        
        # Filter data to only include the specified region
        region_fovs = [fov for fov, region in self.regions.items() if region == region_name]
        
        if not region_fovs:
            raise ValueError(f"No FOVs found for region '{region_name}'. Available regions: {set(self.regions.values())}")
        
        print(f"[NAVIGATOR] Found {len(region_fovs)} FOVs for region '{region_name}': {region_fovs}")
        
        # Build grid for this specific region
        self._report_progress(15, f"Building grid for region {region_name}...")
        self._build_region_grid(region_fovs)
        
        # Create mosaic
        self._report_progress(20, f"Creating mosaic for region {region_name}...")
        mosaic_array = self._create_mosaic_array()
        
        # Create region-specific mappings (CRITICAL FIX)
        region_mappings = self._create_region_specific_mappings(region_fovs)
        
        print(f"[NAVIGATOR] Region {region_name} specific grid has {len(region_mappings['fov_grid'])} FOV positions")
        print(f"[NAVIGATOR] Region {region_name} FOVs in grid: {list(region_mappings['fov_grid'].values())}")
        
        # Build metadata for napari integration
        metadata = {
            'region_name': region_name,
            'grid_dims': self.grid_dims,
            'tile_size': self.tile_size,
            'fov_grid': region_mappings['fov_grid'],  # Use region-specific grid
            'coordinates': {fov: self.coordinates[fov] for fov in region_fovs if fov in self.coordinates},
            'regions': {fov: self.regions[fov] for fov in region_fovs if fov in self.regions},
            'channels': self.channels,
            'grid_bounds': self.grid_bounds,
            'pixel_to_mm_scale': self._calculate_pixel_to_mm_scale(),
            # Add region-specific mapping for napari integration
            'fov_to_grid_pos': region_mappings['fov_to_grid_pos'],  # Use region-specific mapping
            'region_fov_mapping': self._create_region_fov_mapping()
        }
        
        self._report_progress(100, f"Complete for region {region_name}!")
        return mosaic_array, metadata
    
    def _build_region_grid(self, region_fovs: List[int]):
        """Build grid structure for a specific region's FOVs."""
        print(f"[NAVIGATOR] Building grid structure for FOVs: {region_fovs}")
        
        if not region_fovs:
            return
        
        # Get coordinates for this region only
        region_coordinates = {fov: self.coordinates[fov] for fov in region_fovs if fov in self.coordinates}
        
        if not region_coordinates:
            print(f"[NAVIGATOR WARNING] No coordinates found for region FOVs")
            return
        
        # Get coordinate bounds and unique positions for this region
        x_positions = sorted(set(c[0] for c in region_coordinates.values()))
        y_positions = sorted(set(c[1] for c in region_coordinates.values()))
        
        n_cols = len(x_positions)
        n_rows = len(y_positions)
        
        print(f"[NAVIGATOR] Region grid dimensions: {n_rows} rows x {n_cols} cols")
        
        # Store bounds for this region
        self.grid_bounds = (min(x_positions), max(x_positions), 
                           min(y_positions), max(y_positions))
        
        # Create position to index mappings with tolerance
        x_to_col = {x: i for i, x in enumerate(x_positions)}
        y_to_row = {y: i for i, y in enumerate(y_positions)}
        
        tolerance = 0.001  # 1 micron tolerance
        
        # Build FOV grid for this region only
        self.fov_grid.clear()
        
        for fov in region_fovs:
            if fov not in self.coordinates:
                continue
                
            x_mm, y_mm = self.coordinates[fov]
            
            # Find closest x position
            col = None
            for x_pos, idx in x_to_col.items():
                if abs(x_pos - x_mm) < tolerance:
                    col = idx
                    break
                    
            # Find closest y position
            row = None
            for y_pos, idx in y_to_row.items():
                if abs(y_pos - y_mm) < tolerance:
                    row = idx
                    break
                    
            if col is not None and row is not None:
                self.fov_grid[(row, col)] = fov
            else:
                print(f"[NAVIGATOR WARNING] Could not place FOV {fov} at ({x_mm}, {y_mm})")
        
        self.grid_dims = (n_rows, n_cols)
        print(f"[NAVIGATOR] Region grid built with {len(self.fov_grid)} tiles")

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
        # Find timepoint directory - match grid viewer approach
        timepoint_dirs = []
        for item in self.acquisition_dir.iterdir():
            if item.is_dir() and item.name.isdigit():
                timepoint_dirs.append(item)
        
        if not timepoint_dirs:
            raise ValueError(f"No timepoint directories found in {self.acquisition_dir}")
        
        # Use first timepoint if specific one not found
        if timepoint == 0 or not (self.acquisition_dir / str(timepoint)).exists():
            timepoint_dir = sorted(timepoint_dirs, key=lambda x: int(x.name))[0]
        else:
            timepoint_dir = self.acquisition_dir / str(timepoint)
        
        print(f"[NAVIGATOR] Using timepoint directory: {timepoint_dir}")
        
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
        
        # Build metadata for napari integration
        metadata = {
            'grid_dims': self.grid_dims,
            'tile_size': self.tile_size,
            'fov_grid': self.fov_grid,
            'coordinates': self.coordinates,
            'regions': self.regions,
            'channels': self.channels,
            'grid_bounds': self.grid_bounds,
            'pixel_to_mm_scale': self._calculate_pixel_to_mm_scale(),
            # Add mapping for napari integration
            'fov_to_grid_pos': self._create_fov_to_grid_mapping(),
            'region_fov_mapping': self._create_region_fov_mapping()
        }
        
        self._report_progress(100, "Complete!")
        return mosaic_array, metadata
    
    def _create_fov_to_grid_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Create mapping from FOV number to grid position."""
        fov_to_pos = {}
        for (row, col), fov in self.fov_grid.items():
            fov_to_pos[fov] = (row, col)
        return fov_to_pos
    
    def _create_region_fov_mapping(self) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """Create mapping from (region_name, fov_num) to grid position."""
        region_fov_to_pos = {}
        for (row, col), fov in self.fov_grid.items():
            region_name = self.regions.get(fov, 'unknown')
            region_fov_to_pos[(region_name, fov)] = (row, col)
        return region_fov_to_pos
    
    def _create_region_specific_mappings(self, region_fovs: List[int]) -> Dict:
        """Create mappings that only include FOVs from the specified region."""
        # Filter the fov_grid to only include FOVs from this region
        region_specific_fov_grid = {}
        for (row, col), fov in self.fov_grid.items():
            if fov in region_fovs:
                region_specific_fov_grid[(row, col)] = fov
        
        # Create region-specific fov_to_grid mapping
        region_fov_to_pos = {}
        for (row, col), fov in region_specific_fov_grid.items():
            region_fov_to_pos[fov] = (row, col)
        
        return {
            'fov_grid': region_specific_fov_grid,
            'fov_to_grid_pos': region_fov_to_pos
        }
    
    def _report_progress(self, percent: int, message: str):
        """Report progress if callback is available."""
        if self.progress_callback:
            self.progress_callback(percent, message)
        print(f"[NAVIGATOR] {percent}% - {message}")
    
    def _load_coordinates(self, timepoint_dir: Path):
        """Load FOV coordinates from CSV - matches grid viewer exactly."""
        coord_file = timepoint_dir / "coordinates.csv"
        if not coord_file.exists():
            raise FileNotFoundError(f"coordinates.csv not found in {timepoint_dir}")
        
        self.coordinates.clear()
        self.regions.clear()
        
        try:
            with open(coord_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fov = int(row['fov'])
                    x_mm = float(row['x (mm)'])
                    y_mm = float(row['y (mm)'])
                    # Try to get region from CSV, default to 'default' if not present
                    region = row.get('region', 'default')
                    
                    self.coordinates[fov] = (x_mm, y_mm)
                    self.regions[fov] = region
            
            print(f"[NAVIGATOR] Loaded {len(self.coordinates)} FOV coordinates")
            print(f"[NAVIGATOR] Regions found: {set(self.regions.values())}")
            
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to load coordinates: {e}")
            raise
    
    def _scan_files(self, timepoint_dir: Path):
        """Scan directory for TIFF files - matches grid viewer logic exactly."""
        print(f"[NAVIGATOR] Scanning {timepoint_dir} for TIFF files")
        
        self.file_map.clear()
        self.channels = set()
        
        # Find all TIFF files
        tiff_files = list(timepoint_dir.glob("*.tif")) + list(timepoint_dir.glob("*.tiff"))
        print(f"[NAVIGATOR] Found {len(tiff_files)} TIFF files")
        
        for filepath in tiff_files:
            match = FPATTERN.search(filepath.name)
            if not match:
                print(f"[NAVIGATOR WARNING] File doesn't match pattern: {filepath.name}")
                continue
                
            region = match.group("region")
            fov = int(match.group("fov"))
            z_layer = int(match.group("z"))
            modality = match.group("modality")
            channel_info = match.group("channel")
            
            # Create a comprehensive channel identifier - exactly like grid viewer
            if modality.lower() == "fluorescence":
                # For fluorescence, use the wavelength/channel info
                channel = f"{channel_info}"
            else:
                # For other modalities (brightfield, etc.), use modality + channel
                channel = f"{modality}_{channel_info}"
            
            # Only include FOVs that have coordinates
            if fov not in self.coordinates:
                print(f"[NAVIGATOR WARNING] FOV {fov} not found in coordinates, skipping")
                continue
            
            # Update region mapping from filename if not already set
            if fov not in self.regions or self.regions[fov] == 'default':
                self.regions[fov] = region
                
            self.channels.add(channel)
            key = (channel, fov)  # Simplified like grid viewer
            
            if key not in self.file_map:
                self.file_map[key] = []
            self.file_map[key].append(filepath)
        
        self.channels = sorted(list(self.channels))
        print(f"[NAVIGATOR] Found channels: {self.channels}")
        print(f"[NAVIGATOR] Mapped {len(self.file_map)} channel-FOV combinations")
    
    def _build_grid(self):
        """Build grid structure from coordinates - matches grid viewer exactly."""
        print("[NAVIGATOR] Building grid structure")
        
        if not self.coordinates:
            return
        
        # Get coordinate bounds and unique positions - exactly like grid viewer
        x_positions = sorted(set(c[0] for c in self.coordinates.values()))
        y_positions = sorted(set(c[1] for c in self.coordinates.values()))
        
        n_cols = len(x_positions)
        n_rows = len(y_positions)
        
        print(f"[NAVIGATOR] Grid dimensions: {n_rows} rows x {n_cols} cols")
        
        # Store bounds
        self.grid_bounds = (min(x_positions), max(x_positions), 
                           min(y_positions), max(y_positions))
        
        # Create position to index mappings with tolerance - exactly like grid viewer
        x_to_col = {}
        y_to_row = {}
        
        tolerance = 0.001  # 1 micron tolerance
        
        for i, x in enumerate(x_positions):
            x_to_col[x] = i
            
        for i, y in enumerate(y_positions):
            y_to_row[y] = i
        
        # Build FOV grid - exactly like grid viewer
        self.fov_grid.clear()
        
        for fov, (x_mm, y_mm) in self.coordinates.items():
            # Find closest x position
            col = None
            for x_pos, idx in x_to_col.items():
                if abs(x_pos - x_mm) < tolerance:
                    col = idx
                    break
                    
            # Find closest y position
            row = None
            for y_pos, idx in y_to_row.items():
                if abs(y_pos - y_mm) < tolerance:
                    row = idx
                    break
                    
            if col is not None and row is not None:
                self.fov_grid[(row, col)] = fov  # Simplified like grid viewer
            else:
                print(f"[NAVIGATOR WARNING] Could not place FOV {fov} at ({x_mm}, {y_mm})")
        
        self.grid_dims = (n_rows, n_cols)
        print(f"[NAVIGATOR] Grid built with {len(self.fov_grid)} tiles")
    
    def _calculate_pixel_to_mm_scale(self) -> Tuple[float, float]:
        """Calculate mm per pixel for the mosaic."""
        if not self.grid_bounds or not self.grid_dims:
            return (1.0, 1.0)
            
        x_min, x_max, y_min, y_max = self.grid_bounds
        n_rows, n_cols = self.grid_dims
        
        # Calculate mm per tile - matches grid viewer logic
        if n_cols > 1:
            mm_per_tile_x = (x_max - x_min) / (n_cols - 1)
        else:
            mm_per_tile_x = 1.0  # Default value for single column
            
        if n_rows > 1:
            mm_per_tile_y = (y_max - y_min) / (n_rows - 1)
        else:
            mm_per_tile_y = 1.0  # Default value for single row
        
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
        
        for (row, col), fov in self.fov_grid.items():
            processed += 1
            if processed % 5 == 0:  # Report more frequently
                progress = 20 + int(80 * processed / total_tiles)
                self._report_progress(progress, f"Processing tile {processed}/{total_tiles}")
            
            # Get tile image
            tile_img = self._get_tile_image(fov)
            if tile_img is None:
                print(f"[NAVIGATOR WARNING] No tile image for FOV {fov}")
                continue
            
            # Place in mosaic
            y_start = row * self.tile_size
            x_start = col * self.tile_size
            
            # Handle size mismatch
            if tile_img.shape != (self.tile_size, self.tile_size):
                pil_img = Image.fromarray(tile_img)
                pil_img = pil_img.resize((self.tile_size, self.tile_size), Image.Resampling.LANCZOS)
                tile_img = np.array(pil_img)
            
            mosaic[y_start:y_start + self.tile_size,
                   x_start:x_start + self.tile_size] = tile_img
        
        print(f"[NAVIGATOR] Mosaic created: {mosaic.shape}, non-zero pixels: {np.count_nonzero(mosaic)}")
        return mosaic
    
    def _get_tile_image(self, fov: int) -> Optional[np.ndarray]:
        """Get or generate a tile image."""
        # Check cache first
        if self.cache_dir:
            cache_path = self.cache_dir / f"nav_fov_{fov}_{self.tile_size}.npy"
            if cache_path.exists():
                try:
                    cached = np.load(cache_path)
                    print(f"[NAVIGATOR] Loaded cached tile for FOV {fov}")
                    return cached
                except:
                    print(f"[NAVIGATOR WARNING] Failed to load cache for FOV {fov}")
                    pass  # Fall through to regenerate
        
        # Generate tile using the same approach as grid viewer
        tile_img = self._generate_fov_mip(fov)
        
        # Save to cache
        if tile_img is not None and self.cache_dir:
            cache_path = self.cache_dir / f"nav_fov_{fov}_{self.tile_size}.npy"
            try:
                np.save(cache_path, tile_img)
                print(f"[NAVIGATOR] Cached tile for FOV {fov}")
            except Exception as e:
                print(f"[NAVIGATOR WARNING] Failed to cache tile for FOV {fov}: {e}")
        
        return tile_img
    
    def _get_middle_z_file(self, files: List[Path]) -> Path:
        """Select middle z-layer from list of files - exactly like grid viewer."""
        if len(files) == 1:
            return files[0]
        
        # Sort files by z-index
        z_files = []
        for f in files:
            match = FPATTERN.search(f.name)
            if match:
                z = int(match.group("z"))
                z_files.append((z, f))
        
        if not z_files:
            return files[0]  # Fallback
            
        z_files.sort(key=lambda x: x[0])
        mid_idx = len(z_files) // 2
        
        selected_file = z_files[mid_idx][1]
        
        return selected_file
    
    def _generate_fov_mip(self, fov: int) -> Optional[np.ndarray]:
        """Generate fast composite image - matches grid viewer approach exactly."""
        best_channel = None
        best_mean = -1
        best_image = None
        
        print(f"[NAVIGATOR] Generating MIP for FOV {fov}, channels: {self.channels}")
        
        # Quick scan: find channel with highest mean intensity - exactly like grid viewer
        for channel in self.channels:
            key = (channel, fov)
            if key not in self.file_map:
                print(f"[NAVIGATOR] No files for channel {channel}, FOV {fov}")
                continue
                
            # Get middle z file
            file_path = self._get_middle_z_file(self.file_map[key])
            print(f"[NAVIGATOR] Testing channel {channel}, file: {file_path.name}")
            
            try:
                # Read image for mean calculation (small sample) - exactly like grid viewer
                img_array = tf.imread(file_path)
                
                # Quick downsample for mean calculation (every 10th pixel)
                downsampled = img_array[::10, ::10]
                mean_intensity = np.mean(downsampled)
                
                print(f"[NAVIGATOR] Channel {channel}: mean intensity = {mean_intensity}")
                
                if mean_intensity > best_mean:
                    best_mean = mean_intensity
                    best_channel = channel
                    # Store the best image
                    best_image = img_array
                
            except Exception as e:
                print(f"[NAVIGATOR ERROR] Failed to load {channel} for FOV {fov}: {e}")
                continue
        
        if best_image is None:
            print(f"[NAVIGATOR WARNING] No channel data for FOV {fov}")
            return None
        
        print(f"[NAVIGATOR] Best channel for FOV {fov}: {best_channel} (mean: {best_mean})")
        
        try:
            # Improved conversion to 8-bit to avoid "stripped" appearance
            if best_image.dtype == np.uint16:
                # Use percentile-based contrast enhancement instead of simple bit shifting
                # This preserves more detail and avoids "stripped" appearance
                p1, p99 = np.percentile(best_image, [1, 99])
                if p99 > p1:
                    # Normalize using percentiles to enhance contrast
                    normalized = np.clip((best_image - p1) / (p99 - p1), 0, 1)
                    # Apply mild gamma correction for visibility
                    gamma_corrected = np.power(normalized, 0.8)
                    img_8bit = (gamma_corrected * 255).astype(np.uint8)
                else:
                    # Fallback to simpler conversion
                    img_8bit = (best_image >> 8).astype(np.uint8)
            elif best_image.dtype != np.uint8:
                # Quick normalize for other types with percentile-based enhancement
                p1, p99 = np.percentile(best_image, [1, 99])
                if p99 > p1:
                    normalized = np.clip((best_image - p1) / (p99 - p1), 0, 1)
                    # Apply gamma correction for brighter appearance
                    gamma_corrected = np.power(normalized, 0.8)
                    img_8bit = (gamma_corrected * 255).astype(np.uint8)
                else:
                    img_8bit = np.zeros_like(best_image, dtype=np.uint8)
            else:
                # Already 8-bit, apply mild contrast enhancement
                p1, p99 = np.percentile(best_image, [1, 99])
                if p99 > p1:
                    normalized = np.clip((best_image.astype(np.float32) - p1) / (p99 - p1), 0, 1)
                    img_8bit = (normalized * 255).astype(np.uint8)
                else:
                    img_8bit = best_image
            
            # Create PIL image and thumbnail with better resampling
            pil_img = Image.fromarray(img_8bit)
            
            # Use LANCZOS for better quality downsampling to avoid artifacts
            pil_img.thumbnail((self.tile_size, self.tile_size), 
                            Image.Resampling.LANCZOS)
            
            result = np.array(pil_img)
            
            # Ensure the result is the right size (pad if necessary)
            if result.shape != (self.tile_size, self.tile_size):
                padded = np.zeros((self.tile_size, self.tile_size), dtype=np.uint8)
                h, w = result.shape
                padded[:h, :w] = result
                result = padded
            
            print(f"[NAVIGATOR] Generated tile for FOV {fov}: shape {result.shape}, range {result.min()}-{result.max()}")
            return result
            
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to generate composite for FOV {fov}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def pixel_to_mm(self, pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert mosaic pixel coordinates to mm coordinates."""
        if not self.grid_bounds:
            return (0.0, 0.0)
            
        x_min, x_max, y_min, y_max = self.grid_bounds
        mm_per_pixel_x, mm_per_pixel_y = self._calculate_pixel_to_mm_scale()
        
        x_mm = x_min + pixel_x * mm_per_pixel_x
        y_mm = y_min + pixel_y * mm_per_pixel_y
        
        return (x_mm, y_mm)
    
    def mm_to_pixel(self, x_mm: float, y_mm: float) -> Tuple[int, int]:
        """Convert mm coordinates to mosaic pixel coordinates."""
        if not self.grid_bounds:
            return (0, 0)
            
        x_min, x_max, y_min, y_max = self.grid_bounds
        mm_per_pixel_x, mm_per_pixel_y = self._calculate_pixel_to_mm_scale()
        
        pixel_x = int((x_mm - x_min) / mm_per_pixel_x)
        pixel_y = int((y_mm - y_min) / mm_per_pixel_y)
        
        return (pixel_x, pixel_y)

# Convenience function for easy use
def create_navigation_mosaic(acquisition_dir: Path, timepoint: int = 0, 
                           tile_size: int = 75, cache_enabled: bool = True,
                           progress_callback: Optional[Callable] = None) -> Tuple[np.ndarray, Dict]:
    """
    Create a navigation mosaic from an acquisition directory.
    
    Parameters:
    -----------
    acquisition_dir : Path
        Path to the acquisition directory
    timepoint : int
        Timepoint to process (default: 0, uses first available)
    tile_size : int
        Size of each tile in pixels (default: 75)
    cache_enabled : bool
        Whether to use caching (default: True)
    progress_callback : Callable
        Optional progress callback function
    
    Returns:
    --------
    mosaic : np.ndarray
        The navigation mosaic as a numpy array
    metadata : dict
        Metadata including coordinate mappings and grid info
    """
    navigator = DownsampledNavigator(
        acquisition_dir=acquisition_dir,
        tile_size=tile_size,
        cache_enabled=cache_enabled,
        progress_callback=progress_callback
    )
    
    return navigator.create_mosaic(timepoint=timepoint)
