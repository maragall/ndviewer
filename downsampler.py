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
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import queue
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Pattern for acquisitions: {region}_{fov}_{z_layer}_{imaging_modality}_{channel_info}_{suffix}.tiff
# Examples: C5_0_0_Fluorescence_405_nm_Ex.tiff, D6_2_3_Brightfield_BF_Ex.tiff
FPATTERN = re.compile(
    r"(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<modality>[^_]+)_(?P<channel>[^_]+)_.*\.tiff?", re.IGNORECASE
)

class DownsampledNavigator:
    """Fast downsampled mosaic generator for navigation."""
    
    def __init__(self, acquisition_dir: Path, tile_size: int = 75,
                 cache_enabled: bool = True, progress_callback: Optional[Callable] = None,
                 n_workers: Optional[int] = None):
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
        n_workers : int
            Number of worker threads/processes (default: CPU count)
        """
        self.acquisition_dir = Path(acquisition_dir)
        self.tile_size = tile_size
        self.cache_enabled = cache_enabled
        self.progress_callback = progress_callback
        self.n_workers = n_workers or multiprocessing.cpu_count()
        
        # Data storage
        self.coordinates = {}  # {fov: (x_mm, y_mm)}
        self.regions = {}      # {fov: region_name} - Added back for compatibility
        self.channels = []
        self.file_map = {}     # {(region, channel, fov): [filepath, ...]}
        self.fov_grid = {}     # {(row, col): fov}
        self.grid_dims = (0, 0)
        self.grid_bounds = None  # (x_min, x_max, y_min, y_max) in mm
        
        # Performance optimizations
        self._intensity_cache = {}  # Cache channel intensities
        self._global_contrast_limits = {}  # Cache global contrast limits per region
        
        # Cache directory
        if self.cache_enabled:
            self.cache_dir = self.acquisition_dir / "cache"
            self.cache_dir.mkdir(exist_ok=True)
            # CRITICAL FIX: Clear old contaminated cache files that don't have region info
            self._clear_old_cache_files()
        else:
            self.cache_dir = None
    
    def _clear_old_cache_files(self):
        """Clear old cache files that don't have region information in their names."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
            
        try:
            # Find old cache files with pattern nav_fov_*_{tile_size}.npy (no region)
            old_cache_pattern = f"nav_fov_*_{self.tile_size}.npy"
            old_cache_files = list(self.cache_dir.glob(old_cache_pattern))
            
            if old_cache_files:
                print(f"[NAVIGATOR] Clearing {len(old_cache_files)} old contaminated cache files...")
                for cache_file in old_cache_files:
                    try:
                        cache_file.unlink()
                    except Exception as e:
                        print(f"[NAVIGATOR WARNING] Failed to delete old cache file {cache_file.name}: {e}")
                print(f"[NAVIGATOR] Cleared old cache files to prevent cross-region contamination")
                
                # Also clear global contrast limits cache to ensure consistency
                self._global_contrast_limits.clear()
                print(f"[NAVIGATOR] Cleared global contrast limits cache for consistency")
            
        except Exception as e:
            print(f"[NAVIGATOR WARNING] Failed to clear old cache files: {e}")
    
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
        
        # CRITICAL FIX: Filter coordinates to only include this region before scanning files
        original_coordinates = self.coordinates.copy()
        original_regions = self.regions.copy()
        
        # Filter to only this region's FOVs
        region_fovs = [fov for fov, region in self.regions.items() if region == region_name]
        if not region_fovs:
            raise ValueError(f"No FOVs found for region '{region_name}'. Available regions: {set(self.regions.values())}")
        
        # Filter coordinates and regions to only include this region
        self.coordinates = {fov: self.coordinates[fov] for fov in region_fovs if fov in self.coordinates}
        self.regions = {fov: self.regions[fov] for fov in region_fovs if fov in self.regions}
        
        print(f"[NAVIGATOR] Filtered to {len(region_fovs)} FOVs for region '{region_name}': {region_fovs}")
        
        # Scan files - now only files for this region's FOVs will be included
        self._report_progress(10, f"Scanning files for region {region_name}...")
        self._scan_files(timepoint_dir)
        
        # Build grid for this specific region
        self._report_progress(15, f"Building grid for region {region_name}...")
        self._build_region_grid(region_fovs)
        
        # Create mosaic - now file_map and channels only contain this region's data
        # OPTIMIZATION: The prescanning will now calculate global contrast limits automatically
        self._report_progress(20, f"Creating mosaic for region {region_name}...")
        mosaic_array = self._create_mosaic_array(region_name)
        
        # Create region-specific mappings
        region_mappings = self._create_region_specific_mappings(region_fovs)
        
        print(f"[NAVIGATOR] Region {region_name} specific grid has {len(region_mappings['fov_grid'])} FOV positions")
        print(f"[NAVIGATOR] Region {region_name} FOVs in grid: {list(region_mappings['fov_grid'].values())}")
        
        # Build metadata for napari integration
        metadata = {
            'region_name': region_name,
            'grid_dims': self.grid_dims,
            'tile_size': self.tile_size,
            'fov_grid': region_mappings['fov_grid'],  # Use region-specific grid
            'coordinates': self.coordinates.copy(),  # Already filtered to this region
            'regions': self.regions.copy(),  # Already filtered to this region
            'channels': self.channels.copy(),  # Now only contains this region's channels
            'grid_bounds': self.grid_bounds,
            'pixel_to_mm_scale': self._calculate_pixel_to_mm_scale(),
            # Add region-specific mapping for napari integration
            'fov_to_grid_pos': region_mappings['fov_to_grid_pos'],  # Use region-specific mapping
            'region_fov_mapping': self._create_region_fov_mapping()
        }
        
        # CRITICAL: Restore original data for other operations
        self.coordinates = original_coordinates
        self.regions = original_regions
        
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
        # OPTIMIZATION: The prescanning will now calculate global contrast limits automatically
        self._report_progress(20, "Creating mosaic...")
        mosaic_array = self._create_mosaic_array("full_mosaic")
        
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
        """Optimized file scanning with parallel processing."""
        print(f"[NAVIGATOR] Scanning {timepoint_dir} for TIFF files")
        
        self.file_map.clear()
        self.channels = set()
        
        # Find all TIFF files
        tiff_files = list(timepoint_dir.glob("*.tif")) + list(timepoint_dir.glob("*.tiff"))
        print(f"[NAVIGATOR] Found {len(tiff_files)} TIFF files")
        
        # Process files in parallel
        def process_file(filepath):
            match = FPATTERN.search(filepath.name)
            if not match:
                return None
                
            region = match.group("region")
            fov = int(match.group("fov"))
            z_layer = int(match.group("z"))
            modality = match.group("modality")
            channel_info = match.group("channel")
            
            # Create a comprehensive channel identifier
            if modality.lower() == "fluorescence":
                channel = f"{channel_info}"
            else:
                channel = f"{modality}_{channel_info}"
            
            # CRITICAL FIX: Only include FOVs that have coordinates AND are in the current filtered set
            # This prevents cross-region contamination when create_mosaic_for_region filters coordinates
            if fov not in self.coordinates:
                return None
            
            return (region, fov, channel, filepath)
        
        # Use thread pool for I/O bound file scanning
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(process_file, tiff_files))
        
        # Process results
        for result in results:
            if result is None:
                continue
                
            region, fov, channel, filepath = result
            
            # CRITICAL FIX: Only update region mapping if the FOV is in our current coordinate set
            # This prevents overriding region assignments during filtered scanning
            if fov in self.coordinates:
                # Update region mapping from filename if not already set or if it's default
                if fov not in self.regions or self.regions[fov] == 'default':
                    self.regions[fov] = region
                    
                self.channels.add(channel)
                
                # CRITICAL FIX: Use region-aware key to prevent FOV collisions
                # This ensures that FOV 1 from Region A doesn't conflict with FOV 1 from Region B
                key = (region, channel, fov)  # Include region in the key!
                
                if key not in self.file_map:
                    self.file_map[key] = []
                self.file_map[key].append(filepath)
        
        self.channels = sorted(list(self.channels))
        print(f"[NAVIGATOR] Found channels: {self.channels}")
        print(f"[NAVIGATOR] Mapped {len(self.file_map)} region-channel-FOV combinations")
        
        # Debug: Print some example keys to verify region separation
        example_keys = list(self.file_map.keys())[:5]
        print(f"[NAVIGATOR] Example file_map keys: {example_keys}")
    
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
    
    def _create_mosaic_array(self, region_name: str) -> np.ndarray:
        """Create mosaic array using parallel tile processing."""
        n_rows, n_cols = self.grid_dims
        mosaic = np.zeros((n_rows * self.tile_size, n_cols * self.tile_size), dtype=np.uint8)
        
        # Get FOV to grid position mapping
        fov_to_pos = {}
        for (row, col), fov in self.fov_grid.items():
            fov_to_pos[fov] = (row, col)
        
        total_tiles = len(fov_to_pos)
        processed = 0
        
        # Pre-scan for best channels (parallel)
        print(f"[NAVIGATOR] Pre-scanning channels for {total_tiles} tiles...")
        self._prescan_channels_parallel(list(fov_to_pos.keys()), region_name)
        
        # Process tiles in parallel batches
        batch_size = min(32, total_tiles)  # Process in batches to manage memory
        
        def process_tile_batch(batch_fovs):
            """Process a batch of tiles."""
            results = []
            for fov in batch_fovs:
                if fov in fov_to_pos:
                    row, col = fov_to_pos[fov]
                    tile_img = self._get_tile_image_optimized(fov)
                    if tile_img is not None:
                        results.append((row, col, tile_img))
            return results
        
        # Split FOVs into batches
        fov_list = list(fov_to_pos.keys())
        batches = [fov_list[i:i + batch_size] for i in range(0, len(fov_list), batch_size)]
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_batch = {executor.submit(process_tile_batch, batch): i 
                             for i, batch in enumerate(batches)}
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    
                    # Place tiles in mosaic
                    for row, col, tile_img in batch_results:
                        y_start = row * self.tile_size
                        x_start = col * self.tile_size
                        
                        # Ensure correct size
                        if tile_img.shape != (self.tile_size, self.tile_size):
                            tile_img = self._resize_tile_fast(tile_img, self.tile_size)
                        
                        mosaic[y_start:y_start + self.tile_size,
                               x_start:x_start + self.tile_size] = tile_img
                        
                        processed += 1
                    
                    # Update progress
                    progress = 20 + int(80 * processed / total_tiles)
                    self._report_progress(progress, f"Processed batch {batch_idx + 1}/{len(batches)}")
                    
                except Exception as e:
                    print(f"[NAVIGATOR ERROR] Batch {batch_idx} failed: {e}")
        
        print(f"[NAVIGATOR] Mosaic created: {mosaic.shape}, tiles processed: {processed}")
        return mosaic
    
    def _get_tile_image(self, fov: int) -> Optional[np.ndarray]:
        """Get or generate a tile image."""
        # Check cache first
        if self.cache_dir:
            # CRITICAL FIX: Include region in cache key to prevent cross-region contamination
            fov_region = self.regions.get(fov, 'unknown')
            cache_path = self.cache_dir / f"nav_{fov_region}_fov_{fov}_{self.tile_size}.npy"
            if cache_path.exists():
                try:
                    cached = np.load(cache_path)
                    print(f"[NAVIGATOR] Loaded cached tile for region {fov_region}, FOV {fov}")
                    return cached
                except:
                    print(f"[NAVIGATOR WARNING] Failed to load cache for region {fov_region}, FOV {fov}")
                    pass  # Fall through to regenerate
        
        # Generate tile using the same approach as grid viewer
        tile_img = self._generate_fov_mip(fov)
        
        # Save to cache
        if tile_img is not None and self.cache_dir:
            # CRITICAL FIX: Include region in cache key to prevent cross-region contamination
            fov_region = self.regions.get(fov, 'unknown')
            cache_path = self.cache_dir / f"nav_{fov_region}_fov_{fov}_{self.tile_size}.npy"
            try:
                np.save(cache_path, tile_img)
                print(f"[NAVIGATOR] Cached tile for region {fov_region}, FOV {fov}")
            except Exception as e:
                print(f"[NAVIGATOR WARNING] Failed to cache tile for region {fov_region}, FOV {fov}: {e}")
        
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
        
        # CRITICAL FIX: Get the region for this FOV to ensure we only use files from the correct region
        fov_region = self.regions.get(fov, 'unknown')
        if fov_region == 'unknown':
            print(f"[NAVIGATOR WARNING] Unknown region for FOV {fov}")
            return None
        
        print(f"[NAVIGATOR] FOV {fov} belongs to region '{fov_region}'")
        
        # Quick scan: find channel with highest mean intensity - but only for this region
        for channel in self.channels:
            # CRITICAL FIX: Use region-aware key to prevent cross-contamination
            key = (fov_region, channel, fov)  # Include region in the key!
            
            if key not in self.file_map:
                print(f"[NAVIGATOR] No files for region {fov_region}, channel {channel}, FOV {fov}")
                continue
                
            # Get middle z file
            file_path = self._get_middle_z_file(self.file_map[key])
            print(f"[NAVIGATOR] Testing region {fov_region}, channel {channel}, file: {file_path.name}")
            
            try:
                # Read image for mean calculation (small sample) - exactly like grid viewer
                img_array = tf.imread(file_path)
                
                # Quick downsample for mean calculation (every 10th pixel)
                downsampled = img_array[::10, ::10]
                mean_intensity = np.mean(downsampled)
                
                print(f"[NAVIGATOR] Region {fov_region}, Channel {channel}: mean intensity = {mean_intensity}")
                
                if mean_intensity > best_mean:
                    best_mean = mean_intensity
                    best_channel = channel
                    # Store the best image
                    best_image = img_array
                
            except Exception as e:
                print(f"[NAVIGATOR ERROR] Failed to load {channel} for FOV {fov} in region {fov_region}: {e}")
                continue
        
        if best_image is None:
            print(f"[NAVIGATOR WARNING] No channel data for FOV {fov} in region {fov_region}")
            return None
        
        print(f"[NAVIGATOR] Best channel for FOV {fov} in region {fov_region}: {best_channel} (mean: {best_mean})")
        
        try:
            # HOMOGENEOUS CONTRAST FIX: Use global contrast limits instead of per-tile limits
            global_p1, global_p99 = self._global_contrast_limits.get(fov_region, (None, None))
            
            # If no region-specific limits, try full mosaic limits
            if global_p1 is None and global_p99 is None:
                global_p1, global_p99 = self._global_contrast_limits.get("full_mosaic", (None, None))
            
            # Improved conversion to 8-bit with GLOBAL contrast normalization - NO GAMMA CORRECTION
            if best_image.dtype == np.uint16:
                if global_p1 is not None and global_p99 is not None and global_p99 > global_p1:
                    # Use GLOBAL percentiles for homogeneous contrast across all tiles
                    print(f"[NAVIGATOR] Applying global contrast limits to FOV {fov}: p1={global_p1:.1f}, p99={global_p99:.1f}")
                    normalized = np.clip((best_image - global_p1) / (global_p99 - global_p1), 0, 1)
                    # NO gamma correction - preserve original intensity relationships
                    img_8bit = (normalized * 255).astype(np.uint8)
                else:
                    # Fallback to simple bit shifting if no global limits available
                    print(f"[NAVIGATOR WARNING] No global contrast limits for region {fov_region}, using fallback")
                    img_8bit = (best_image >> 8).astype(np.uint8)
            elif best_image.dtype != np.uint8:
                if global_p1 is not None and global_p99 is not None and global_p99 > global_p1:
                    # Use GLOBAL percentiles for homogeneous contrast
                    normalized = np.clip((best_image - global_p1) / (global_p99 - global_p1), 0, 1)
                    # NO gamma correction - preserve original intensity relationships
                    img_8bit = (normalized * 255).astype(np.uint8)
                else:
                    # Fallback for other types
                    img_8bit = np.zeros_like(best_image, dtype=np.uint8)
            else:
                # Already 8-bit, apply global contrast if available
                if global_p1 is not None and global_p99 is not None and global_p99 > global_p1:
                    normalized = np.clip((best_image.astype(np.float32) - global_p1) / (global_p99 - global_p1), 0, 1)
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
            
            print(f"[NAVIGATOR] Generated tile for FOV {fov} in region {fov_region}: shape {result.shape}, range {result.min()}-{result.max()}")
            return result
            
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to generate composite for FOV {fov} in region {fov_region}: {e}")
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
    
    def _prescan_channels_parallel(self, fovs: List[int], region_name: str = None):
        """Pre-scan channels to find best intensity for each FOV AND collect contrast samples in one pass."""
        print(f"[NAVIGATOR] Pre-scanning {len(fovs)} FOVs for best channels and contrast samples...")
        
        def scan_fov_channels(fov):
            """Scan all channels for a single FOV and collect contrast samples."""
            best_channel = None
            best_mean = -1
            best_samples = None  # Store samples from the best channel
            
            # CRITICAL FIX: Get the region for this FOV
            fov_region = self.regions.get(fov, 'unknown')
            if fov_region == 'unknown':
                return (fov, None, -1, None)
            
            for channel in self.channels:
                # CRITICAL FIX: Use region-aware key
                key = (fov_region, channel, fov)
                if key not in self.file_map:
                    continue
                
                # Get middle z file
                file_path = self._get_middle_z_file(self.file_map[key])
                
                try:
                    # Quick intensity check using memory mapping
                    with tf.TiffFile(file_path) as tif:
                        img_array = tif.asarray(out='memmap')
                        
                        # Sample every 20th pixel for speed (REUSE existing downsampling!)
                        downsampled = img_array[::20, ::20]
                        mean_intensity = np.mean(downsampled)
                        
                        if mean_intensity > best_mean:
                            best_mean = mean_intensity
                            best_channel = channel
                            # OPTIMIZATION: Collect contrast samples from the SAME downsampled data!
                            # Flatten and take a random subset for global contrast calculation
                            flat_samples = downsampled.flatten()
                            # Take every 4th pixel from the already-downsampled data (~250 samples per FOV)
                            best_samples = flat_samples[::4]  # Much faster than random sampling!
                        
                        # Close memmap
                        del img_array
                        
                except Exception:
                    continue
            
            return (fov, best_channel, best_mean, best_samples)
        
        # Process FOVs in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(scan_fov_channels, fovs))
        
        # Store results in intensity cache AND collect contrast samples
        region_samples = []
        for fov, channel, intensity, samples in results:
            if channel:
                self._intensity_cache[fov] = {
                    'channel': channel,
                    'intensity': intensity
                }
                
                # OPTIMIZATION: Collect contrast samples during the same pass!
                if samples is not None and region_name:
                    # Only include samples from FOVs that belong to the target region
                    fov_region = self.regions.get(fov, 'unknown')
                    if region_name == "full_mosaic" or fov_region == region_name:
                        region_samples.extend(samples.tolist())
        
        # OPTIMIZATION: Calculate global contrast limits immediately from collected samples
        if region_name and region_samples:
            samples_array = np.array(region_samples)
            p1, p99 = np.percentile(samples_array, [1, 99])
            self._global_contrast_limits[region_name] = (p1, p99)
            print(f"[NAVIGATOR] Calculated global contrast limits for '{region_name}': p1={p1:.1f}, p99={p99:.1f} (from {len(region_samples)} samples)")
        elif region_name:
            print(f"[NAVIGATOR WARNING] No contrast samples collected for region '{region_name}', using defaults")
            self._global_contrast_limits[region_name] = (0.0, 255.0)
    
    def _get_tile_image_optimized(self, fov: int) -> Optional[np.ndarray]:
        """Optimized tile image generation."""
        # Check cache first
        if self.cache_dir:
            # CRITICAL FIX: Include region in cache key to prevent cross-region contamination
            fov_region = self.regions.get(fov, 'unknown')
            cache_path = self.cache_dir / f"nav_{fov_region}_fov_{fov}_{self.tile_size}.npy"
            if cache_path.exists():
                try:
                    return np.load(cache_path)
                except:
                    pass  # Fall through to regenerate
        
        # Use pre-scanned best channel if available
        if fov in self._intensity_cache:
            best_channel = self._intensity_cache[fov]['channel']
            
            # CRITICAL FIX: Get the region for this FOV and use region-aware key
            fov_region = self.regions.get(fov, 'unknown')
            if fov_region == 'unknown':
                # Fall back to original method
                return self._get_tile_image(fov)
            
            key = (fov_region, best_channel, fov)  # Use region-aware key
            
            if key in self.file_map:
                file_path = self._get_middle_z_file(self.file_map[key])
                
                try:
                    # Use memory mapping for efficient reading
                    with tf.TiffFile(file_path) as tif:
                        img_array = tif.asarray(out='memmap')
                        
                        # Get global contrast limits for this region
                        global_p1, global_p99 = self._global_contrast_limits.get(fov_region, (None, None))
                        
                        # If no region-specific limits, try full mosaic limits
                        if global_p1 is None and global_p99 is None:
                            global_p1, global_p99 = self._global_contrast_limits.get("full_mosaic", (None, None))
                        
                        # Optimized 8-bit conversion with global contrast limits
                        img_8bit = self._convert_to_8bit_fast(img_array, global_p1, global_p99)
                        
                        # Fast downsampling
                        tile = self._downsample_fast(img_8bit, self.tile_size)
                        
                        # Close memmap
                        del img_array
                        
                        # Cache the result
                        if self.cache_dir:
                            try:
                                # CRITICAL FIX: Include region in cache key
                                cache_path = self.cache_dir / f"nav_{fov_region}_fov_{fov}_{self.tile_size}.npy"
                                np.save(cache_path, tile)
                            except:
                                pass
                        
                        return tile
                        
                except Exception as e:
                    print(f"[NAVIGATOR ERROR] Failed to generate optimized tile for FOV {fov}: {e}")
        
        # Fall back to original method
        return self._get_tile_image(fov)
    
    def _convert_to_8bit_fast(self, img: np.ndarray, global_p1: Optional[float] = None, global_p99: Optional[float] = None) -> np.ndarray:
        """Fast 8-bit conversion with global contrast limits for homogeneous appearance."""
        if img.dtype == np.uint8:
            return img
        
        # Use global contrast limits if provided, otherwise fall back to per-image calculation
        if global_p1 is not None and global_p99 is not None and global_p99 > global_p1:
            p1, p99 = global_p1, global_p99
            print(f"[NAVIGATOR] Using global contrast limits for fast conversion: p1={p1:.1f}, p99={p99:.1f}")
        else:
            # Fallback to per-image percentile calculation
            sample = img[::10, ::10]
            p1, p99 = np.percentile(sample, [1, 99])
            print(f"[NAVIGATOR WARNING] No global limits provided, using per-image contrast")
        
        if p99 > p1:
            # Vectorized normalization
            scale = 255.0 / (p99 - p1)
            
            # Apply scaling efficiently
            normalized = np.clip((img.astype(np.float32) - p1) * scale, 0, 255)
            result = normalized.astype(np.uint8)
            
            return result
        else:
            # Fallback for flat images
            return np.zeros(img.shape, dtype=np.uint8)
    
    def _downsample_fast(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """Fast image downsampling using area averaging."""
        h, w = img.shape
        
        # Calculate downsampling factors
        factor_h = h / target_size
        factor_w = w / target_size
        
        # If already small enough, just resize
        if factor_h <= 1 and factor_w <= 1:
            pil_img = Image.fromarray(img)
            pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            return np.array(pil_img)
        
        # For large downsampling factors, use area averaging for speed
        if factor_h > 4 and factor_w > 4:
            # Fast area averaging using stride tricks
            kernel_h = int(factor_h)
            kernel_w = int(factor_w)
            
            # Crop to multiple of kernel size
            crop_h = (h // kernel_h) * kernel_h
            crop_w = (w // kernel_w) * kernel_w
            img_cropped = img[:crop_h, :crop_w]
            
            # Reshape and average
            reshaped = img_cropped.reshape(crop_h // kernel_h, kernel_h,
                                         crop_w // kernel_w, kernel_w)
            downsampled = reshaped.mean(axis=(1, 3)).astype(np.uint8)
            
            # Final resize to exact target
            pil_img = Image.fromarray(downsampled)
            pil_img = pil_img.resize((target_size, target_size), Image.Resampling.LANCZOS)
            return np.array(pil_img)
        else:
            # For smaller factors, use PIL directly
            pil_img = Image.fromarray(img)
            pil_img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            result = np.array(pil_img)
            
            # Pad if necessary
            if result.shape != (target_size, target_size):
                padded = np.zeros((target_size, target_size), dtype=np.uint8)
                h, w = result.shape
                padded[:h, :w] = result
                result = padded
            
            return result
    
    def _resize_tile_fast(self, tile: np.ndarray, target_size: int) -> np.ndarray:
        """Fast tile resizing."""
        if tile.shape == (target_size, target_size):
            return tile
        
        pil_img = Image.fromarray(tile)
        pil_img = pil_img.resize((target_size, target_size), Image.Resampling.NEAREST)
        return np.array(pil_img)

    def create_channel_mosaics_for_region(self, region_name: str, timepoint: int = 0) -> Tuple[Dict[str, np.ndarray], Dict]:
        """
        Create downsampled mosaics for ALL channels in a specific region.
        
        Parameters:
        -----------
        region_name : str
            Name of the region to create mosaics for
        timepoint : int
            Timepoint to process
            
        Returns:
        --------
        channel_mosaics : Dict[str, np.ndarray]
            Dictionary mapping channel names to mosaic arrays
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
        
        print(f"[NAVIGATOR] Creating multi-channel mosaics for region '{region_name}' from {timepoint_dir}")
        
        # Load coordinates
        self._report_progress(5, f"Loading coordinates for region {region_name}...")
        self._load_coordinates(timepoint_dir)
        
        # CRITICAL: Filter coordinates to only include this region before scanning files
        original_coordinates = self.coordinates.copy()
        original_regions = self.regions.copy()
        
        # Filter to only this region's FOVs
        region_fovs = [fov for fov, region in self.regions.items() if region == region_name]
        if not region_fovs:
            raise ValueError(f"No FOVs found for region '{region_name}'. Available regions: {set(self.regions.values())}")
        
        # Filter coordinates and regions to only include this region
        self.coordinates = {fov: self.coordinates[fov] for fov in region_fovs if fov in self.coordinates}
        self.regions = {fov: self.regions[fov] for fov in region_fovs if fov in self.regions}
        
        print(f"[NAVIGATOR] Filtered to {len(region_fovs)} FOVs for region '{region_name}': {region_fovs}")
        
        # Scan files - now only files for this region's FOVs will be included
        self._report_progress(10, f"Scanning files for region {region_name}...")
        self._scan_files(timepoint_dir)
        
        # Build grid for this specific region
        self._report_progress(15, f"Building grid for region {region_name}...")
        self._build_region_grid(region_fovs)
        
        # Create mosaics for ALL channels
        self._report_progress(20, f"Creating multi-channel mosaics for region {region_name}...")
        channel_mosaics = self._create_channel_mosaic_arrays(region_name)
        
        # Create region-specific mappings
        region_mappings = self._create_region_specific_mappings(region_fovs)
        
        print(f"[NAVIGATOR] Created {len(channel_mosaics)} channel mosaics for region {region_name}")
        print(f"[NAVIGATOR] Channels: {list(channel_mosaics.keys())}")
        print(f"[NAVIGATOR] Region {region_name} specific grid has {len(region_mappings['fov_grid'])} FOV positions")
        
        # Build metadata for napari integration
        metadata = {
            'region_name': region_name,
            'grid_dims': self.grid_dims,
            'tile_size': self.tile_size,
            'fov_grid': region_mappings['fov_grid'],  # Use region-specific grid
            'coordinates': self.coordinates.copy(),  # Already filtered to this region
            'regions': self.regions.copy(),  # Already filtered to this region
            'channels': self.channels.copy(),  # Now only contains this region's channels
            'grid_bounds': self.grid_bounds,
            'pixel_to_mm_scale': self._calculate_pixel_to_mm_scale(),
            # Add region-specific mapping for napari integration
            'fov_to_grid_pos': region_mappings['fov_to_grid_pos'],  # Use region-specific mapping
            'region_fov_mapping': self._create_region_fov_mapping()
        }
        
        # CRITICAL: Restore original data for other operations
        self.coordinates = original_coordinates
        self.regions = original_regions
        
        self._report_progress(100, f"Complete for region {region_name}!")
        return channel_mosaics, metadata

    def _create_channel_mosaic_arrays(self, region_name: str) -> Dict[str, np.ndarray]:
        """
        Create mosaic arrays for ALL channels using memory-conscious parallel processing.
        
        LOGIC:
        1. Initialize empty mosaics for each channel
        2. Process FOVs in parallel batches 
        3. For each FOV, generate tiles for ALL channels sequentially (memory-conscious)
        4. Place tiles into appropriate channel mosaics
        
        Returns:
        --------
        Dict[str, np.ndarray] : Channel name -> mosaic array mapping
        """
        print(f"[NAVIGATOR] Creating channel mosaic arrays for region {region_name}")
        print(f"[NAVIGATOR] Channels to process: {self.channels}")
        
        # Initialize mosaics for each channel
        n_rows, n_cols = self.grid_dims
        mosaic_shape = (n_rows * self.tile_size, n_cols * self.tile_size)
        
        channel_mosaics = {}
        for channel in self.channels:
            channel_mosaics[channel] = np.zeros(mosaic_shape, dtype=np.uint8)
            print(f"[NAVIGATOR] Initialized mosaic for channel {channel}: shape {mosaic_shape}")
        
        # Get FOV to grid position mapping
        fov_to_pos = {}
        for (row, col), fov in self.fov_grid.items():
            fov_to_pos[fov] = (row, col)
        
        total_tiles = len(fov_to_pos)
        processed_fovs = 0
        
        print(f"[NAVIGATOR] Processing {total_tiles} FOVs for {len(self.channels)} channels")
        
        # Pre-scan for global contrast limits per channel
        print(f"[NAVIGATOR] Pre-scanning for global contrast limits...")
        self._prescan_all_channels_parallel(list(fov_to_pos.keys()), region_name)
        
        # Process FOVs in parallel batches
        batch_size = min(16, total_tiles)  # Smaller batches for memory management
        
        def process_fov_batch(batch_fovs):
            """
            Process a batch of FOVs - generate tiles for ALL channels per FOV.
            
            CRITICAL LOGIC:
            - For each FOV in batch, generate tiles for ALL channels
            - Process channels SEQUENTIALLY per FOV to manage memory
            - Return results as {fov: {channel: tile}}
            """
            batch_results = {}
            
            for fov in batch_fovs:
                if fov in fov_to_pos:
                    print(f"[NAVIGATOR] Processing FOV {fov} for all channels...")
                    
                    # Generate tiles for ALL channels for this FOV
                    fov_channel_tiles = self._generate_fov_all_channels(fov)
                    
                    if fov_channel_tiles:  # Only store if we got some tiles
                        batch_results[fov] = fov_channel_tiles
                        print(f"[NAVIGATOR] FOV {fov}: generated {len(fov_channel_tiles)} channel tiles")
                    else:
                        print(f"[NAVIGATOR] WARNING: No tiles generated for FOV {fov}")
            
            return batch_results
        
        # Split FOVs into batches
        fov_list = list(fov_to_pos.keys())
        batches = [fov_list[i:i + batch_size] for i in range(0, len(fov_list), batch_size)]
        
        print(f"[NAVIGATOR] Processing {len(batches)} batches of FOVs...")
        
        # Process batches in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            future_to_batch = {executor.submit(process_fov_batch, batch): i 
                             for i, batch in enumerate(batches)}
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    batch_results = future.result()
                    
                    # Place tiles into appropriate channel mosaics
                    for fov, fov_tiles in batch_results.items():
                        if fov in fov_to_pos:
                            row, col = fov_to_pos[fov]
                            
                            # Place each channel's tile in its corresponding mosaic
                            for channel, tile in fov_tiles.items():
                                if tile is not None and channel in channel_mosaics:
                                    y_start = row * self.tile_size
                                    x_start = col * self.tile_size
                                    
                                    # Ensure correct size
                                    if tile.shape != (self.tile_size, self.tile_size):
                                        tile = self._resize_tile_fast(tile, self.tile_size)
                                    
                                    # Place tile in channel mosaic
                                    channel_mosaics[channel][y_start:y_start + self.tile_size,
                                                           x_start:x_start + self.tile_size] = tile
                            
                            processed_fovs += 1
                    
                    # Update progress
                    progress = 20 + int(75 * processed_fovs / total_tiles)
                    self._report_progress(progress, f"Processed batch {batch_idx + 1}/{len(batches)} ({processed_fovs}/{total_tiles} FOVs)")
                    
                except Exception as e:
                    print(f"[NAVIGATOR ERROR] Batch {batch_idx} failed: {e}")
                    import traceback
                    traceback.print_exc()
        
        # Report final statistics
        for channel, mosaic in channel_mosaics.items():
            non_zero_pixels = np.count_nonzero(mosaic)
            print(f"[NAVIGATOR] Channel {channel}: {non_zero_pixels:,} non-zero pixels out of {mosaic.size:,} total")
        
        print(f"[NAVIGATOR] Multi-channel mosaics created: {len(channel_mosaics)} channels, processed {processed_fovs} FOVs")
        return channel_mosaics

    def _generate_fov_all_channels(self, fov: int) -> Dict[str, np.ndarray]:
        """
        Generate tiles for ALL channels of a single FOV.
        
        CRITICAL LOGIC:
        1. Process channels SEQUENTIALLY to manage memory
        2. Use channel-specific caching 
        3. Apply global contrast limits per channel
        4. Clean up intermediate data between channels
        
        Returns:
        --------
        Dict[str, np.ndarray] : Channel name -> tile array mapping
        """
        fov_region = self.regions.get(fov, 'unknown')
        if fov_region == 'unknown':
            print(f"[NAVIGATOR WARNING] Unknown region for FOV {fov}")
            return {}
        
        print(f"[NAVIGATOR] Generating all channel tiles for FOV {fov} in region '{fov_region}'")
        
        channel_tiles = {}
        
        # Process each channel sequentially for this FOV
        for channel in self.channels:
            try:
                # Check cache first - channel-specific cache key
                tile = self._get_cached_tile_for_channel(fov, channel, fov_region)
                
                if tile is None:
                    # Generate tile for this specific channel
                    tile = self._generate_tile_for_channel(fov, channel, fov_region)
                    
                    # Cache the result if successful
                    if tile is not None:
                        self._cache_tile_for_channel(fov, channel, fov_region, tile)
                
                if tile is not None:
                    channel_tiles[channel] = tile
                    print(f"[NAVIGATOR] FOV {fov}, channel {channel}: tile generated successfully")
                else:
                    print(f"[NAVIGATOR] WARNING: Failed to generate tile for FOV {fov}, channel {channel}")
                    
            except Exception as e:
                print(f"[NAVIGATOR ERROR] Failed to process FOV {fov}, channel {channel}: {e}")
                continue
        
        print(f"[NAVIGATOR] FOV {fov}: generated {len(channel_tiles)}/{len(self.channels)} channel tiles")
        return channel_tiles

    def _get_cached_tile_for_channel(self, fov: int, channel: str, fov_region: str) -> Optional[np.ndarray]:
        """Get cached tile for specific FOV and channel."""
        if not self.cache_dir:
            return None
            
        # CRITICAL: Channel-specific cache key to prevent cross-contamination
        cache_path = self.cache_dir / f"nav_{fov_region}_fov_{fov}_ch_{channel}_{self.tile_size}.npy"
        
        if cache_path.exists():
            try:
                cached_tile = np.load(cache_path)
                print(f"[NAVIGATOR] Loaded cached tile for region {fov_region}, FOV {fov}, channel {channel}")
                return cached_tile
            except Exception as e:
                print(f"[NAVIGATOR WARNING] Failed to load cache for region {fov_region}, FOV {fov}, channel {channel}: {e}")
                # Remove corrupted cache file
                try:
                    cache_path.unlink()
                except:
                    pass
        
        return None

    def _cache_tile_for_channel(self, fov: int, channel: str, fov_region: str, tile: np.ndarray):
        """Cache tile for specific FOV and channel."""
        if not self.cache_dir:
            return
            
        # CRITICAL: Channel-specific cache key
        cache_path = self.cache_dir / f"nav_{fov_region}_fov_{fov}_ch_{channel}_{self.tile_size}.npy"
        
        try:
            np.save(cache_path, tile)
            print(f"[NAVIGATOR] Cached tile for region {fov_region}, FOV {fov}, channel {channel}")
        except Exception as e:
            print(f"[NAVIGATOR WARNING] Failed to cache tile for region {fov_region}, FOV {fov}, channel {channel}: {e}")

    def _generate_tile_for_channel(self, fov: int, channel: str, fov_region: str) -> Optional[np.ndarray]:
        """
        Generate tile for a specific FOV and channel.
        
        CRITICAL LOGIC:
        1. Use region-aware key to get correct files
        2. Select middle z-layer from files
        3. Apply global contrast limits for this channel
        4. Convert and downsample efficiently
        """
        print(f"[NAVIGATOR] Generating tile for region {fov_region}, FOV {fov}, channel {channel}")
        
        # CRITICAL: Use region-aware key to prevent cross-contamination
        key = (fov_region, channel, fov)
        
        if key not in self.file_map:
            print(f"[NAVIGATOR] No files for region {fov_region}, channel {channel}, FOV {fov}")
            return None
        
        # Get middle z file
        file_path = self._get_middle_z_file(self.file_map[key])
        print(f"[NAVIGATOR] Using file: {file_path.name}")
        
        try:
            # Read image using memory mapping for efficiency
            with tf.TiffFile(file_path) as tif:
                img_array = tif.asarray(out='memmap')
                
                # Get global contrast limits for this channel and region
                global_p1, global_p99 = self._global_contrast_limits.get(f"{fov_region}_{channel}", (None, None))
                
                # Fallback to region-wide limits if channel-specific not available
                if global_p1 is None and global_p99 is None:
                    global_p1, global_p99 = self._global_contrast_limits.get(fov_region, (None, None))
                
                # Convert to 8-bit with global contrast limits
                img_8bit = self._convert_to_8bit_fast(img_array, global_p1, global_p99)
                
                # Fast downsampling to tile size
                tile = self._downsample_fast(img_8bit, self.tile_size)
                
                # Clean up memory mapping
                del img_array
                
                print(f"[NAVIGATOR] Generated tile for region {fov_region}, FOV {fov}, channel {channel}: shape {tile.shape}, range {tile.min()}-{tile.max()}")
                return tile
                
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to generate tile for region {fov_region}, FOV {fov}, channel {channel}: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _prescan_all_channels_parallel(self, fovs: List[int], region_name: str):
        """
        Pre-scan ALL channels for ALL FOVs to collect global contrast samples.
        
        CRITICAL LOGIC:
        1. Scan every channel for every FOV (not just brightest)
        2. Collect contrast samples per channel per region
        3. Calculate global contrast limits per channel
        4. Store as region_channel keys for lookup
        """
        print(f"[NAVIGATOR] Pre-scanning ALL channels for {len(fovs)} FOVs in region {region_name}...")
        
        # Storage for contrast samples per channel
        channel_samples = {channel: [] for channel in self.channels}
        
        def scan_fov_all_channels(fov):
            """Scan ALL channels for a single FOV and collect contrast samples."""
            fov_region = self.regions.get(fov, 'unknown')
            if fov_region == 'unknown':
                return {}
            
            fov_samples = {}
            
            for channel in self.channels:
                # Use region-aware key
                key = (fov_region, channel, fov)
                if key not in self.file_map:
                    continue
                
                # Get middle z file
                file_path = self._get_middle_z_file(self.file_map[key])
                
                try:
                    # Quick sampling using memory mapping
                    with tf.TiffFile(file_path) as tif:
                        img_array = tif.asarray(out='memmap')
                        
                        # Sample every 20th pixel for speed
                        downsampled = img_array[::20, ::20]
                        
                        # Collect samples for contrast calculation
                        flat_samples = downsampled.flatten()
                        # Take every 4th pixel from downsampled data
                        channel_samples_for_fov = flat_samples[::4]
                        
                        fov_samples[channel] = channel_samples_for_fov.tolist()
                        
                        # Clean up memmap
                        del img_array
                        
                except Exception as e:
                    print(f"[NAVIGATOR WARNING] Failed to sample FOV {fov}, channel {channel}: {e}")
                    continue
            
            return fov_samples
        
        # Process FOVs in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(scan_fov_all_channels, fovs))
        
        # Collect all samples by channel
        for fov_samples in results:
            for channel, samples in fov_samples.items():
                if samples:
                    channel_samples[channel].extend(samples)
        
        # Calculate global contrast limits per channel
        for channel in self.channels:
            if channel_samples[channel]:
                samples_array = np.array(channel_samples[channel])
                p1, p99 = np.percentile(samples_array, [1, 99])
                
                # Store with region_channel key for specificity
                contrast_key = f"{region_name}_{channel}"
                self._global_contrast_limits[contrast_key] = (p1, p99)
                
                print(f"[NAVIGATOR] Global contrast for {region_name}_{channel}: p1={p1:.1f}, p99={p99:.1f} (from {len(channel_samples[channel])} samples)")
            else:
                print(f"[NAVIGATOR WARNING] No samples collected for {region_name}_{channel}, using defaults")
                contrast_key = f"{region_name}_{channel}"
                self._global_contrast_limits[contrast_key] = (0.0, 255.0)

# Convenience function for easy use
def create_navigation_mosaic(acquisition_dir: Path, timepoint: int = 0, 
                           tile_size: int = 75, cache_enabled: bool = True,
                           progress_callback: Optional[Callable] = None,
                           n_workers: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
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
    n_workers : int
        Number of worker threads/processes (default: CPU count)
    
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
        progress_callback=progress_callback,
        n_workers=n_workers
    )
    
    return navigator.create_mosaic(timepoint=timepoint)

# Convenience function for multi-channel navigation mosaics
def create_channel_navigation_mosaics(acquisition_dir: Path, region_name: str, 
                                    timepoint: int = 0, tile_size: int = 75, 
                                    cache_enabled: bool = True,
                                    progress_callback: Optional[Callable] = None,
                                    n_workers: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Create multi-channel navigation mosaics for a specific region.
    
    Parameters:
    -----------
    acquisition_dir : Path
        Path to the acquisition directory
    region_name : str
        Name of the region to create mosaics for
    timepoint : int
        Timepoint to process (default: 0, uses first available)
    tile_size : int
        Size of each tile in pixels (default: 75)
    cache_enabled : bool
        Whether to use caching (default: True)
    progress_callback : Callable
        Optional progress callback function
    n_workers : int
        Number of worker threads/processes (default: CPU count)
    
    Returns:
    --------
    channel_mosaics : Dict[str, np.ndarray]
        Dictionary mapping channel names to mosaic arrays
    metadata : dict
        Metadata including coordinate mappings and grid info for the region
    """
    navigator = DownsampledNavigator(
        acquisition_dir=acquisition_dir,
        tile_size=tile_size,
        cache_enabled=cache_enabled,
        progress_callback=progress_callback,
        n_workers=n_workers
    )
    
    return navigator.create_channel_mosaics_for_region(region_name=region_name, timepoint=timepoint)
