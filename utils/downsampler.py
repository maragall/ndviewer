#!/usr/bin/env python3
"""
Downsampler module for creating fast preview mosaics.
Provides a lightweight navigator view for large acquisitions.
"""

import re
import csv
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Callable, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing
import numpy as np
import tifffile as tf
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

try:
    from scipy import ndimage
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("[NAVIGATOR WARNING] scipy not available, using fallback interpolation methods")

# Pattern for acquisitions: {region}_{fov}_{z_layer}_{imaging_modality}_{channel_info}_{suffix}.tiff
FPATTERN = re.compile(
    r"(?P<region>[^_]+)_(?P<fov>\d+)_(?P<z>\d+)_(?P<modality>[^_]+)_(?P<channel>[^_]+)_.*\.tiff?", re.IGNORECASE
)

# Global variable to store the first FOV count for scaling calculations
# This is used to calculate the FOV layer size based on the formula:
# FOV_layer_size = floor(sqrt(((number_of_fovs/number_of_regions)*downsampler_tile_size)*desired_ratio))
FIRST_FOV_COUNT = None


@dataclass
class NavigatorConfig:
    """Configuration for navigator behavior"""
    tile_size: int = 75
    cache_enabled: bool = True
    n_workers: Optional[int] = None
    coordinate_tolerance: float = 0.001  # 1 micron
    downsample_factor: int = 20  # For intensity sampling
    sample_stride: int = 4  # For contrast sampling
    
    def __post_init__(self):
        if self.n_workers is None:
            self.n_workers = multiprocessing.cpu_count()


@dataclass
class NavigatorData:
    """Storage for navigator data"""
    coordinates: Dict[int, Tuple[float, float]] = field(default_factory=dict)
    regions: Dict[int, str] = field(default_factory=dict)
    channels: List[str] = field(default_factory=list)
    file_map: Dict[Tuple[str, str, int], List[Path]] = field(default_factory=dict)
    fov_grid: Dict[Tuple[int, int], int] = field(default_factory=dict)
    grid_dims: Tuple[int, int] = (0, 0)
    grid_bounds: Optional[Tuple[float, float, float, float]] = None
    
    # Per-region grid data for multi-region support
    region_grids: Dict[str, Dict[Tuple[int, int], int]] = field(default_factory=dict)
    region_grid_dims: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    region_grid_bounds: Dict[str, Tuple[float, float, float, float]] = field(default_factory=dict)
    region_grid_offsets: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    
    # Caches
    intensity_cache: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    global_contrast_limits: Dict[str, Tuple[float, float]] = field(default_factory=dict)


class FileHandler:
    """Handles file operations and caching"""
    
    def __init__(self, acquisition_dir: Path, config: NavigatorConfig):
        self.acquisition_dir = Path(acquisition_dir)
        self.config = config
        self.cache_dir = None
        self.regions = {}  # Store regions for access in scan_files
        self._region_fov_mapping = {}  # Maps (region, original_fov) -> unique_fov
        
        if config.cache_enabled:
            self.cache_dir = self.acquisition_dir / "tile_cache"
            self.cache_dir.mkdir(exist_ok=True)
            self._clear_cache()
    
    def _clear_cache(self):
        """Clear all cache files to ensure contrast changes take effect"""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        cache_patterns = [
            f"nav_*_{self.config.tile_size}.npy",
            f"nav_*_fov_*_{self.config.tile_size}.npy",
            f"nav_*_fov_*_ch_*_{self.config.tile_size}.npy"
        ]
        
        cache_files = []
        for pattern in cache_patterns:
            cache_files.extend(self.cache_dir.glob(pattern))
        
        if cache_files:
            print(f"[NAVIGATOR] Clearing {len(cache_files)} cached tiles...")
            for file in cache_files:
                try:
                    file.unlink()
                except Exception as e:
                    print(f"[NAVIGATOR WARNING] Failed to delete {file.name}: {e}")
    
    def find_timepoint_dir(self, timepoint: int = 0) -> Path:
        """Find the appropriate timepoint directory"""
        timepoint_dirs = [d for d in self.acquisition_dir.iterdir() 
                         if d.is_dir() and d.name.isdigit()]
        
        if not timepoint_dirs:
            raise ValueError(f"No timepoint directories found in {self.acquisition_dir}")
        
        if timepoint == 0 or not (self.acquisition_dir / str(timepoint)).exists():
            return sorted(timepoint_dirs, key=lambda x: int(x.name))[0]
        
        return self.acquisition_dir / str(timepoint)
    
    def load_coordinates(self, timepoint_dir: Path) -> Tuple[Dict[int, Tuple[float, float]], Dict[int, str]]:
        """Load FOV coordinates from CSV"""
        coord_file = timepoint_dir / "coordinates.csv"
        if not coord_file.exists():
            raise FileNotFoundError(f"coordinates.csv not found in {timepoint_dir}")
        
        coordinates = {}
        regions = {}
        
        # First pass: check if we have duplicate FOV numbers across regions
        fov_region_pairs = []
        with open(coord_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                fov = int(row['fov'])
                region = row.get('region', 'default')
                x_mm = float(row['x (mm)'])
                y_mm = float(row['y (mm)'])
                fov_region_pairs.append((region, fov, x_mm, y_mm))
        
        # Check for duplicate FOV numbers across different regions
        fov_counts = {}
        for region, fov, _, _ in fov_region_pairs:
            if fov not in fov_counts:
                fov_counts[fov] = set()
            fov_counts[fov].add(region)
        
        has_duplicates = any(len(regions_set) > 1 for regions_set in fov_counts.values())
        
        if has_duplicates:
            print(f"[NAVIGATOR] Detected duplicate FOV numbers across regions, creating unique IDs")
            
            # Create unique IDs by combining region and FOV
            unique_id = 0
            for region, fov, x_mm, y_mm in fov_region_pairs:
                coordinates[unique_id] = (x_mm, y_mm)
                regions[unique_id] = region
                self._region_fov_mapping[(region, fov)] = unique_id
                unique_id += 1
        else:
            # No duplicates - use original FOV numbers
            for region, fov, x_mm, y_mm in fov_region_pairs:
                coordinates[fov] = (x_mm, y_mm)
                regions[fov] = region
                self._region_fov_mapping[(region, fov)] = fov
        
        self.regions = regions  # Store for use in scan_files
        print(f"[NAVIGATOR] Loaded {len(coordinates)} FOV coordinates")
        return coordinates, regions
    
    def scan_files(self, timepoint_dir: Path, valid_fovs: Set[int]) -> Tuple[Dict, List[str]]:
        """Scan directory for TIFF files and organize by region/channel/fov"""
        tiff_files = list(timepoint_dir.glob("*.tif")) + list(timepoint_dir.glob("*.tiff"))
        print(f"[NAVIGATOR] Found {len(tiff_files)} TIFF files")
        
        file_map = {}
        channels = set()
        
        def process_file(filepath):
            match = FPATTERN.search(filepath.name)
            if not match:
                return None
            
            file_region = match.group("region")
            file_fov = int(match.group("fov"))
            
            # Look up the unique FOV ID using the mapping
            unique_fov = self._region_fov_mapping.get((file_region, file_fov))
            
            if unique_fov is None or unique_fov not in valid_fovs:
                return None
            
            modality = match.group("modality")
            channel_info = match.group("channel")
            
            channel = f"{channel_info}" if modality.lower() == "fluorescence" else f"{modality}_{channel_info}"
            
            # Get the region from our stored regions dict
            region = self.regions.get(unique_fov, file_region)
            
            return (region, channel, unique_fov, filepath)
        
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            results = list(executor.map(process_file, tiff_files))
        
        for result in results:
            if result is None:
                continue
            
            region, channel, unique_fov, filepath = result
            channels.add(channel)
            
            key = (region, channel, unique_fov)
            if key not in file_map:
                file_map[key] = []
            file_map[key].append(filepath)
        
        return file_map, sorted(list(channels))
    
    def get_middle_z_file(self, files: List[Path]) -> Path:
        """Select middle z-layer from list of files"""
        if len(files) == 1:
            return files[0]
        
        z_files = []
        for f in files:
            match = FPATTERN.search(f.name)
            if match:
                z = int(match.group("z"))
                z_files.append((z, f))
        
        if not z_files:
            return files[0]
        
        z_files.sort(key=lambda x: x[0])
        return z_files[len(z_files) // 2][1]
    
    def get_cached_tile(self, fov: int, channel: str, region: str) -> Optional[np.ndarray]:
        """Load cached tile if available"""
        if not self.cache_dir:
            return None
        
        cache_path = self.cache_dir / f"nav_{region}_fov_{fov}_ch_{channel}_{self.config.tile_size}.npy"
        
        if cache_path.exists():
            try:
                return np.load(cache_path)
            except Exception as e:
                print(f"[NAVIGATOR WARNING] Failed to load cache: {e}")
                try:
                    cache_path.unlink()
                except:
                    pass
        
        return None
    
    def save_cached_tile(self, fov: int, channel: str, region: str, tile: np.ndarray):
        """Save tile to cache"""
        if not self.cache_dir:
            return
        
        cache_path = self.cache_dir / f"nav_{region}_fov_{fov}_ch_{channel}_{self.config.tile_size}.npy"
        
        try:
            np.save(cache_path, tile)
        except Exception as e:
            print(f"[NAVIGATOR WARNING] Failed to cache tile: {e}")
    
    def determine_data_type(self, file_map: Dict) -> np.dtype:
        """Determine data type from first available image"""
        for key, files in file_map.items():
            if files:
                try:
                    with tf.TiffFile(files[0]) as tif:
                        return tif.asarray(out='memmap').dtype
                except:
                    continue
        
        return np.uint16


class ImageProcessor:
    """Handles image processing operations"""
    
    def __init__(self, config: NavigatorConfig):
        self.config = config
    
    def apply_contrast(self, img: np.ndarray, p1: Optional[float] = None, 
                      p99: Optional[float] = None) -> np.ndarray:
        """Apply minimal contrast adjustment to preserve original intensity"""
        # CONTRAST FIX: Preserve original intensity distribution
        print(f"[NAVIGATOR] Preserving original intensity distribution")
        return img.astype(img.dtype)
    
    def downsample_fast(self, img: np.ndarray, target_size: int) -> np.ndarray:
        """Fast image downsampling using area averaging"""
        h, w = img.shape
        dtype = img.dtype
        
        if h <= target_size and w <= target_size:
            result = np.zeros((target_size, target_size), dtype=dtype)
            result[:min(h, target_size), :min(w, target_size)] = img[:min(h, target_size), :min(w, target_size)]
            return result
        
        factor_h = h / target_size
        factor_w = w / target_size
        
        if factor_h > 4 and factor_w > 4:
            # Fast area averaging
            kernel_h = int(factor_h)
            kernel_w = int(factor_w)
            
            crop_h = (h // kernel_h) * kernel_h
            crop_w = (w // kernel_w) * kernel_w
            img_cropped = img[:crop_h, :crop_w]
            
            reshaped = img_cropped.reshape(crop_h // kernel_h, kernel_h,
                                         crop_w // kernel_w, kernel_w)
            downsampled = reshaped.mean(axis=(1, 3)).astype(dtype)
            
            return self._resize_to_exact(downsampled, target_size, dtype)
        else:
            # Simple decimation
            step_h = max(1, int(h / target_size))
            step_w = max(1, int(w / target_size))
            downsampled = img[::step_h, ::step_w]
            
            return self._resize_to_exact(downsampled, target_size, dtype)
    
    def _resize_to_exact(self, img: np.ndarray, target_size: int, dtype: np.dtype) -> np.ndarray:
        """Resize image to exact target size"""
        h, w = img.shape
        
        if SCIPY_AVAILABLE:
            zoom_h = target_size / h
            zoom_w = target_size / w
            resized = ndimage.zoom(img, (zoom_h, zoom_w), order=1, prefilter=False)
            
            if resized.shape != (target_size, target_size):
                result = np.zeros((target_size, target_size), dtype=dtype)
                h_min = min(resized.shape[0], target_size)
                w_min = min(resized.shape[1], target_size)
                result[:h_min, :w_min] = resized[:h_min, :w_min]
                return result.astype(dtype)
            
            return resized.astype(dtype)
        else:
            # Nearest neighbor fallback
            result = np.zeros((target_size, target_size), dtype=dtype)
            
            for i in range(target_size):
                for j in range(target_size):
                    src_i = int(i * h / target_size)
                    src_j = int(j * w / target_size)
                    result[i, j] = img[src_i, src_j]
            
            return result


class GridBuilder:
    """Handles grid construction and coordinate mapping"""
    
    def __init__(self, config: NavigatorConfig):
        self.config = config
    
    def build_grid(self, coordinates: Dict[int, Tuple[float, float]], 
                   region_fovs: Optional[List[int]] = None) -> Tuple[Dict, Tuple[int, int], Tuple]:
        """Build grid structure from coordinates"""
        if region_fovs:
            # Filter coordinates for specific region
            coords = {fov: coordinates[fov] for fov in region_fovs if fov in coordinates}
        else:
            coords = coordinates
        
        if not coords:
            return {}, (0, 0), None
        
        # Get unique positions
        x_positions = sorted(set(c[0] for c in coords.values()))
        y_positions = sorted(set(c[1] for c in coords.values()))
        
        n_cols = len(x_positions)
        n_rows = len(y_positions)
        
        # Create position mappings
        x_to_col = {x: i for i, x in enumerate(x_positions)}
        y_to_row = {y: i for i, y in enumerate(y_positions)}
        
        # Build FOV grid
        fov_grid = {}
        for fov, (x_mm, y_mm) in coords.items():
            col = self._find_closest_index(x_mm, x_to_col, self.config.coordinate_tolerance)
            row = self._find_closest_index(y_mm, y_to_row, self.config.coordinate_tolerance)
            
            if col is not None and row is not None:
                fov_grid[(row, col)] = fov
        
        grid_bounds = (min(x_positions), max(x_positions), 
                      min(y_positions), max(y_positions))
        
        print(f"[NAVIGATOR] Grid built: {n_rows}x{n_cols}, {len(fov_grid)} tiles")
        return fov_grid, (n_rows, n_cols), grid_bounds
    
    def build_multi_region_grid(self, coordinates: Dict[int, Tuple[float, float]], 
                               regions: Dict[int, str]) -> Tuple[Dict, Dict, Dict, Dict, Tuple[int, int]]:
        """Build separate grids for each region and compute layout"""
        # Group FOVs by region
        region_fovs = {}
        for fov, region in regions.items():
            if region not in region_fovs:
                region_fovs[region] = []
            region_fovs[region].append(fov)
        
        # Build individual grids for each region
        region_grids = {}
        region_grid_dims = {}
        region_grid_bounds = {}
        
        for region_name, fovs in region_fovs.items():
            grid, dims, bounds = self.build_grid(coordinates, fovs)
            region_grids[region_name] = grid
            region_grid_dims[region_name] = dims
            region_grid_bounds[region_name] = bounds
        
        # Compute optimal layout for regions
        region_grid_offsets = self._compute_region_layout(region_grid_dims)
        
        # Calculate total dimensions
        max_row = 0
        max_col = 0
        for region_name, (row_offset, col_offset) in region_grid_offsets.items():
            rows, cols = region_grid_dims[region_name]
            max_row = max(max_row, row_offset + rows)
            max_col = max(max_col, col_offset + cols)
        
        total_dims = (max_row, max_col)
        
        print(f"[NAVIGATOR] Multi-region grid: {len(region_grids)} regions, total size {total_dims}")
        return region_grids, region_grid_dims, region_grid_bounds, region_grid_offsets, total_dims
    
    def _compute_region_layout(self, region_grid_dims: Dict[str, Tuple[int, int]]) -> Dict[str, Tuple[int, int]]:
        """Compute optimal layout for multiple regions"""
        if not region_grid_dims:
            return {}
        
        # Simple layout: stack regions vertically with small gap
        region_offsets = {}
        current_row = 0
        
        for region_name in sorted(region_grid_dims.keys()):
            rows, cols = region_grid_dims[region_name]
            region_offsets[region_name] = (current_row, 0)
            current_row += rows + 1  # Add 1 row gap between regions
        
        return region_offsets
    
    def _find_closest_index(self, value: float, mapping: Dict[float, int], 
                           tolerance: float) -> Optional[int]:
        """Find index of closest value within tolerance"""
        for pos, idx in mapping.items():
            if abs(pos - value) < tolerance:
                return idx
        return None
    
    def calculate_pixel_to_mm_scale(self, grid_bounds: Tuple, grid_dims: Tuple, 
                                   tile_size: int) -> Tuple[float, float]:
        """Calculate mm per pixel for the mosaic"""
        if not grid_bounds or not grid_dims:
            return (1.0, 1.0)
        
        x_min, x_max, y_min, y_max = grid_bounds
        n_rows, n_cols = grid_dims
        
        mm_per_tile_x = (x_max - x_min) / (n_cols - 1) if n_cols > 1 else 1.0
        mm_per_tile_y = (y_max - y_min) / (n_rows - 1) if n_rows > 1 else 1.0
        
        return (mm_per_tile_x / tile_size, mm_per_tile_y / tile_size)


class DownsampledNavigator:
    """Fast downsampled mosaic generator for navigation."""
    
    def __init__(self, acquisition_dir: Path, tile_size: int = 75,
                 cache_enabled: bool = True, progress_callback: Optional[Callable] = None,
                 n_workers: Optional[int] = None):
        self.config = NavigatorConfig(
            tile_size=tile_size,
            cache_enabled=cache_enabled,
            n_workers=n_workers
        )
        self.data = NavigatorData()
        self.progress_callback = progress_callback
        
        # Helper classes
        self.file_handler = FileHandler(acquisition_dir, self.config)
        self.image_processor = ImageProcessor(self.config)
        self.grid_builder = GridBuilder(self.config)
    
    def create_mosaic(self, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """Create a downsampled mosaic for the specified timepoint"""
        timepoint_dir = self.file_handler.find_timepoint_dir(timepoint)
        print(f"[NAVIGATOR] Using timepoint directory: {timepoint_dir}")
        
        # Load and process data
        self._report_progress(5, "Loading coordinates...")
        self.data.coordinates, self.data.regions = self.file_handler.load_coordinates(timepoint_dir)
        
        self._report_progress(10, "Scanning files...")
        self.data.file_map, self.data.channels = self.file_handler.scan_files(
            timepoint_dir, set(self.data.coordinates.keys())
        )
        
        # Store the first FOV count globally for scaling calculations
        global FIRST_FOV_COUNT
        if FIRST_FOV_COUNT is None:
            # Count FOVs in the first region
            first_region = list(set(self.data.regions.values()))[0]
            fovs_in_first_region = sum(1 for region in self.data.regions.values() if region == first_region)
            FIRST_FOV_COUNT = fovs_in_first_region
            print(f"[NAVIGATOR] Stored first FOV count: {FIRST_FOV_COUNT} FOVs in region '{first_region}'")
        
        # Check if we have multiple regions
        unique_regions = set(self.data.regions.values())
        if len(unique_regions) > 1:
            print(f"[NAVIGATOR] Multi-region acquisition detected: {unique_regions}")
            self._report_progress(15, "Building multi-region grid structure...")
            
            # Build multi-region grid
            (self.data.region_grids, 
             self.data.region_grid_dims,
             self.data.region_grid_bounds,
             self.data.region_grid_offsets,
             self.data.grid_dims) = self.grid_builder.build_multi_region_grid(
                self.data.coordinates, self.data.regions
            )
            
            # Create unified grid for compatibility
            self.data.fov_grid = self._create_unified_grid()
            
            self._report_progress(20, "Creating multi-region mosaic...")
            mosaic_array = self._create_multi_region_mosaic_array()
        else:
            # Single region - use original logic
            self._report_progress(15, "Building grid structure...")
            self.data.fov_grid, self.data.grid_dims, self.data.grid_bounds = self.grid_builder.build_grid(
                self.data.coordinates
            )
            
            self._report_progress(20, "Creating mosaic...")
            mosaic_array = self._create_mosaic_array("full_mosaic")
        
        metadata = self._build_metadata()
        self._report_progress(100, "Complete!")
        return mosaic_array, metadata
    
    def create_mosaic_for_region(self, region_name: str, timepoint: int = 0) -> Tuple[np.ndarray, Dict]:
        """Create a downsampled mosaic for a specific region"""
        return self._create_region_mosaic(region_name, timepoint, single_channel=True)
    
    def create_channel_mosaics_for_region(self, region_name: str, 
                                         timepoint: int = 0) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Create downsampled mosaics for ALL channels in a specific region"""
        return self._create_region_mosaic(region_name, timepoint, single_channel=False)
    
    def _create_unified_grid(self) -> Dict[Tuple[int, int], int]:
        """Create unified grid from region grids with offsets"""
        unified_grid = {}
        
        for region_name, region_grid in self.data.region_grids.items():
            row_offset, col_offset = self.data.region_grid_offsets[region_name]
            
            for (row, col), fov in region_grid.items():
                unified_pos = (row + row_offset, col + col_offset)
                unified_grid[unified_pos] = fov
        
        return unified_grid
    
    def _create_multi_region_mosaic_array(self) -> np.ndarray:
        """Create mosaic array for multi-region acquisition"""
        n_rows, n_cols = self.data.grid_dims
        dtype = self.file_handler.determine_data_type(self.data.file_map)
        mosaic = np.zeros((n_rows * self.config.tile_size, n_cols * self.config.tile_size), dtype=dtype)
        
        # Pre-scan channels for each region separately
        for region_name, region_grid in self.data.region_grids.items():
            region_fovs = list(region_grid.values())
            print(f"[NAVIGATOR] Pre-scanning channels for region '{region_name}'...")
            self._prescan_channels_parallel(region_fovs, region_name)
        
        # Process tiles for each region
        total_tiles = sum(len(grid) for grid in self.data.region_grids.values())
        processed = 0
        
        for region_name, region_grid in self.data.region_grids.items():
            row_offset, col_offset = self.data.region_grid_offsets[region_name]
            
            # Get contrast limits for this region
            p1, p99 = self.data.global_contrast_limits.get(region_name, (None, None))
            
            # Process tiles for this region
            for (local_row, local_col), fov in region_grid.items():
                # Calculate global position
                global_row = local_row + row_offset
                global_col = local_col + col_offset
                
                # Generate tile
                tile = self._generate_single_tile(fov)
                
                if tile is not None:
                    y_start = global_row * self.config.tile_size
                    x_start = global_col * self.config.tile_size
                    
                    # Ensure correct size
                    if tile.shape != (self.config.tile_size, self.config.tile_size):
                        tile = self.image_processor.downsample_fast(tile, self.config.tile_size)
                    
                    # Place tile in mosaic
                    mosaic[y_start:y_start + self.config.tile_size,
                          x_start:x_start + self.config.tile_size] = tile
                
                processed += 1
                progress = 20 + int(80 * processed / total_tiles)
                self._report_progress(progress, f"Processing region '{region_name}': {processed}/{total_tiles} tiles")
        
        return mosaic
    
    def _create_region_mosaic(self, region_name: str, timepoint: int, 
                             single_channel: bool) -> Tuple[Any, Dict]:
        """Internal method to create region mosaics"""
        timepoint_dir = self.file_handler.find_timepoint_dir(timepoint)
        
        # Load full data
        self._report_progress(5, f"Loading coordinates for region {region_name}...")
        full_coordinates, full_regions = self.file_handler.load_coordinates(timepoint_dir)
        
        # Filter to region
        region_fovs = [fov for fov, region in full_regions.items() if region == region_name]
        if not region_fovs:
            raise ValueError(f"No FOVs found for region '{region_name}'")
        
        # Set filtered data
        self.data.coordinates = {fov: full_coordinates[fov] for fov in region_fovs if fov in full_coordinates}
        self.data.regions = {fov: full_regions[fov] for fov in region_fovs}
        
        print(f"[NAVIGATOR] Filtered to {len(region_fovs)} FOVs for region '{region_name}'")
        
        # Process files and grid
        self._report_progress(10, f"Scanning files for region {region_name}...")
        self.data.file_map, self.data.channels = self.file_handler.scan_files(
            timepoint_dir, set(self.data.coordinates.keys())
        )
        
        self._report_progress(15, f"Building grid for region {region_name}...")
        self.data.fov_grid, self.data.grid_dims, self.data.grid_bounds = self.grid_builder.build_grid(
            self.data.coordinates, region_fovs
        )
        
        # Create mosaic(s)
        self._report_progress(20, f"Creating {'mosaic' if single_channel else 'multi-channel mosaics'} for region {region_name}...")
        
        if single_channel:
            mosaic_array = self._create_mosaic_array(region_name)
            result = mosaic_array
        else:
            channel_mosaics = self._create_channel_mosaic_arrays(region_name)
            result = channel_mosaics
        
        # Build metadata
        region_mappings = self._create_region_specific_mappings(region_fovs)
        metadata = self._build_metadata(region_name=region_name, region_mappings=region_mappings)
        
        self._report_progress(100, f"Complete for region {region_name}!")
        return result, metadata
    
    def _create_mosaic_array(self, region_name: str) -> np.ndarray:
        """Create single channel mosaic array"""
        n_rows, n_cols = self.data.grid_dims
        dtype = self.file_handler.determine_data_type(self.data.file_map)
        mosaic = np.zeros((n_rows * self.config.tile_size, n_cols * self.config.tile_size), dtype=dtype)
        
        fov_to_pos = {fov: pos for pos, fov in self.data.fov_grid.items()}
        
        # Pre-scan for best channels
        print(f"[NAVIGATOR] Pre-scanning channels...")
        self._prescan_channels_parallel(list(fov_to_pos.keys()), region_name)
        
        # Process tiles
        self._process_tiles_parallel(mosaic, fov_to_pos, single_channel=True)
        
        return mosaic
    
    def _create_channel_mosaic_arrays(self, region_name: str) -> Dict[str, np.ndarray]:
        """Create multi-channel mosaic arrays"""
        n_rows, n_cols = self.data.grid_dims
        dtype = self.file_handler.determine_data_type(self.data.file_map)
        
        # Initialize mosaics for each channel
        channel_mosaics = {}
        for channel in self.data.channels:
            channel_mosaics[channel] = np.zeros(
                (n_rows * self.config.tile_size, n_cols * self.config.tile_size), 
                dtype=dtype
            )
        
        fov_to_pos = {fov: pos for pos, fov in self.data.fov_grid.items()}
        
        # Pre-scan all channels
        print(f"[NAVIGATOR] Pre-scanning ALL channels...")
        self._prescan_all_channels_parallel(list(fov_to_pos.keys()), region_name)
        
        # Process tiles for all channels
        self._process_tiles_parallel(channel_mosaics, fov_to_pos, single_channel=False)
        
        return channel_mosaics
    
    def _prescan_channels_parallel(self, fovs: List[int], region_name: str):
        """Pre-scan to find best channel for each FOV"""
        def scan_fov(fov):
            best_channel = None
            best_mean = -1
            best_samples = None
            
            fov_region = self.data.regions.get(fov, 'unknown')
            if fov_region == 'unknown':
                return (fov, None, -1, None)
            
            for channel in self.data.channels:
                key = (fov_region, channel, fov)
                if key not in self.data.file_map:
                    continue
                
                file_path = self.file_handler.get_middle_z_file(self.data.file_map[key])
                
                try:
                    with tf.TiffFile(file_path) as tif:
                        img_array = tif.asarray(out='memmap')
                        downsampled = img_array[::self.config.downsample_factor, ::self.config.downsample_factor]
                        mean_intensity = np.mean(downsampled)
                        
                        if mean_intensity > best_mean:
                            best_mean = mean_intensity
                            best_channel = channel
                            flat_samples = downsampled.flatten()
                            best_samples = flat_samples[::self.config.sample_stride]
                        
                        del img_array
                except:
                    continue
            
            return (fov, best_channel, best_mean, best_samples)
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            results = list(executor.map(scan_fov, fovs))
        
        # Store results and collect contrast samples
        region_samples = []
        for fov, channel, intensity, samples in results:
            if channel:
                self.data.intensity_cache[fov] = {'channel': channel, 'intensity': intensity}
                if samples is not None:
                    fov_region = self.data.regions.get(fov, 'unknown')
                    if region_name == "full_mosaic" or fov_region == region_name:
                        region_samples.extend(samples.tolist())
        
        # Calculate global contrast
        if region_samples:
            samples_array = np.array(region_samples)
            p1, p99 = np.percentile(samples_array, [1, 99])
            self.data.global_contrast_limits[region_name] = (p1, p99)
            print(f"[NAVIGATOR] Global contrast for '{region_name}': p1={p1:.1f}, p99={p99:.1f}")
    
    def _prescan_all_channels_parallel(self, fovs: List[int], region_name: str):
        """Pre-scan ALL channels for contrast calculation"""
        channel_samples = {channel: [] for channel in self.data.channels}
        
        def scan_fov_all_channels(fov):
            fov_region = self.data.regions.get(fov, 'unknown')
            if fov_region == 'unknown':
                return {}
            
            fov_samples = {}
            
            for channel in self.data.channels:
                key = (fov_region, channel, fov)
                if key not in self.data.file_map:
                    continue
                
                file_path = self.file_handler.get_middle_z_file(self.data.file_map[key])
                
                try:
                    with tf.TiffFile(file_path) as tif:
                        img_array = tif.asarray(out='memmap')
                        downsampled = img_array[::self.config.downsample_factor, ::self.config.downsample_factor]
                        flat_samples = downsampled.flatten()
                        fov_samples[channel] = flat_samples[::self.config.sample_stride].tolist()
                        del img_array
                except:
                    continue
            
            return fov_samples
        
        # Process in parallel
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            results = list(executor.map(scan_fov_all_channels, fovs))
        
        # Collect samples by channel
        for fov_samples in results:
            for channel, samples in fov_samples.items():
                if samples:
                    channel_samples[channel].extend(samples)
        
        # Calculate per-channel contrast
        for channel in self.data.channels:
            if channel_samples[channel]:
                samples_array = np.array(channel_samples[channel])
                p1, p99 = np.percentile(samples_array, [1, 99])
                contrast_key = f"{region_name}_{channel}"
                self.data.global_contrast_limits[contrast_key] = (p1, p99)
                print(f"[NAVIGATOR] Contrast for {contrast_key}: p1={p1:.1f}, p99={p99:.1f}")
    
    def _process_tiles_parallel(self, mosaic_target: Any, fov_to_pos: Dict[int, Tuple[int, int]], 
                               single_channel: bool):
        """Process tiles in parallel batches"""
        total_tiles = len(fov_to_pos)
        processed = 0
        batch_size = min(32 if single_channel else 16, total_tiles)
        
        def process_batch(batch_fovs):
            results = []
            for fov in batch_fovs:
                if fov in fov_to_pos:
                    row, col = fov_to_pos[fov]
                    
                    if single_channel:
                        tile = self._generate_single_tile(fov)
                        if tile is not None:
                            results.append((row, col, None, tile))
                    else:
                        tiles = self._generate_all_channel_tiles(fov)
                        for channel, tile in tiles.items():
                            if tile is not None:
                                results.append((row, col, channel, tile))
            
            return results
        
        # Process batches
        fov_list = list(fov_to_pos.keys())
        batches = [fov_list[i:i + batch_size] for i in range(0, len(fov_list), batch_size)]
        
        with ThreadPoolExecutor(max_workers=self.config.n_workers) as executor:
            future_to_batch = {executor.submit(process_batch, batch): i 
                             for i, batch in enumerate(batches)}
            
            for future in as_completed(future_to_batch):
                batch_idx = future_to_batch[future]
                try:
                    results = future.result()
                    
                    for row, col, channel, tile in results:
                        y_start = row * self.config.tile_size
                        x_start = col * self.config.tile_size
                        
                        # Ensure correct size
                        if tile.shape != (self.config.tile_size, self.config.tile_size):
                            tile = self.image_processor.downsample_fast(tile, self.config.tile_size)
                        
                        # Place tile
                        if single_channel:
                            mosaic_target[y_start:y_start + self.config.tile_size,
                                        x_start:x_start + self.config.tile_size] = tile
                        else:
                            mosaic_target[channel][y_start:y_start + self.config.tile_size,
                                                 x_start:x_start + self.config.tile_size] = tile
                        
                        processed += 1
                    
                    # Update progress
                    progress = 20 + int(80 * processed / (total_tiles * (1 if single_channel else len(self.data.channels))))
                    self._report_progress(progress, f"Processed batch {batch_idx + 1}/{len(batches)}")
                    
                except Exception as e:
                    print(f"[NAVIGATOR ERROR] Batch {batch_idx} failed: {e}")
        
        print(f"[NAVIGATOR] Mosaic processing complete: {processed} tiles")
    
    def _generate_single_tile(self, fov: int) -> Optional[np.ndarray]:
        """Generate a single tile using best channel"""
        fov_region = self.data.regions.get(fov, 'unknown')
        
        # Check cache first
        if fov in self.data.intensity_cache:
            best_channel = self.data.intensity_cache[fov]['channel']
            cached_tile = self.file_handler.get_cached_tile(fov, best_channel, fov_region)
            if cached_tile is not None:
                return cached_tile
            
            # Generate from best channel
            key = (fov_region, best_channel, fov)
            if key in self.data.file_map:
                tile = self._generate_tile_from_file(
                    self.file_handler.get_middle_z_file(self.data.file_map[key]),
                    fov_region
                )
                
                if tile is not None:
                    self.file_handler.save_cached_tile(fov, best_channel, fov_region, tile)
                
                return tile
        
        # Fallback: scan all channels
        return self._generate_fallback_tile(fov)
    
    def _generate_all_channel_tiles(self, fov: int) -> Dict[str, np.ndarray]:
        """Generate tiles for all channels of a FOV"""
        fov_region = self.data.regions.get(fov, 'unknown')
        if fov_region == 'unknown':
            return {}
        
        channel_tiles = {}
        
        for channel in self.data.channels:
            # Check cache
            cached_tile = self.file_handler.get_cached_tile(fov, channel, fov_region)
            if cached_tile is not None:
                channel_tiles[channel] = cached_tile
                continue
            
            # Generate tile
            key = (fov_region, channel, fov)
            if key in self.data.file_map:
                file_path = self.file_handler.get_middle_z_file(self.data.file_map[key])
                
                # Get channel-specific contrast
                contrast_key = f"{fov_region}_{channel}"
                p1, p99 = self.data.global_contrast_limits.get(contrast_key, (None, None))
                
                tile = self._generate_tile_from_file(file_path, fov_region, p1, p99)
                
                if tile is not None:
                    channel_tiles[channel] = tile
                    self.file_handler.save_cached_tile(fov, channel, fov_region, tile)
        
        return channel_tiles
    
    def _generate_tile_from_file(self, file_path: Path, region: str, 
                                p1: Optional[float] = None, p99: Optional[float] = None) -> Optional[np.ndarray]:
        """Generate a tile from a specific file"""
        try:
            with tf.TiffFile(file_path) as tif:
                img_array = tif.asarray(out='memmap')
                
                # Use region contrast if not specified
                if p1 is None or p99 is None:
                    p1, p99 = self.data.global_contrast_limits.get(region, (None, None))
                
                # Apply contrast and downsample
                img_processed = self.image_processor.apply_contrast(img_array, p1, p99)
                tile = self.image_processor.downsample_fast(img_processed, self.config.tile_size)
                
                del img_array
                return tile
                
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to generate tile from {file_path}: {e}")
            return None
    
    def _generate_fallback_tile(self, fov: int) -> Optional[np.ndarray]:
        """Generate tile by scanning all channels"""
        fov_region = self.data.regions.get(fov, 'unknown')
        best_channel = None
        best_mean = -1
        best_image = None
        
        for channel in self.data.channels:
            key = (fov_region, channel, fov)
            if key not in self.data.file_map:
                continue
            
            file_path = self.file_handler.get_middle_z_file(self.data.file_map[key])
            
            try:
                img_array = tf.imread(file_path)
                downsampled = img_array[::10, ::10]
                mean_intensity = np.mean(downsampled)
                
                if mean_intensity > best_mean:
                    best_mean = mean_intensity
                    best_channel = channel
                    best_image = img_array
                    
            except:
                continue
        
        if best_image is None:
            return None
        
        # Process best image
        p1, p99 = self.data.global_contrast_limits.get(fov_region, (None, None))
        img_processed = self.image_processor.apply_contrast(best_image, p1, p99)
        return self.image_processor.downsample_fast(img_processed, self.config.tile_size)
    
    def _build_metadata(self, region_name: Optional[str] = None, 
                       region_mappings: Optional[Dict] = None) -> Dict:
        """Build metadata dictionary"""
        # Handle multi-region case
        if self.data.region_grids:
            # Multi-region metadata
            metadata = {
                'grid_dims': self.data.grid_dims,
                'tile_size': self.config.tile_size,
                'coordinates': self.data.coordinates.copy(),
                'regions': self.data.regions.copy(),
                'channels': self.data.channels.copy(),
                'fov_grid': self.data.fov_grid,
                'fov_to_grid_pos': self._create_fov_to_grid_mapping(),
                'region_fov_mapping': self._create_region_fov_mapping(),
                'is_multi_region': True,
                'region_grids': self.data.region_grids,
                'region_grid_dims': self.data.region_grid_dims,
                'region_grid_bounds': self.data.region_grid_bounds,
                'region_grid_offsets': self.data.region_grid_offsets
            }
            
            # Add pixel scale for each region
            region_pixel_scales = {}
            for region, bounds in self.data.region_grid_bounds.items():
                dims = self.data.region_grid_dims[region]
                scale = self.grid_builder.calculate_pixel_to_mm_scale(bounds, dims, self.config.tile_size)
                region_pixel_scales[region] = scale
            metadata['region_pixel_scales'] = region_pixel_scales
            
        else:
            # Single region metadata (original)
            pixel_scale = self.grid_builder.calculate_pixel_to_mm_scale(
                self.data.grid_bounds, self.data.grid_dims, self.config.tile_size
            )
            
            metadata = {
                'grid_dims': self.data.grid_dims,
                'tile_size': self.config.tile_size,
                'coordinates': self.data.coordinates.copy(),
                'regions': self.data.regions.copy(),
                'channels': self.data.channels.copy(),
                'grid_bounds': self.data.grid_bounds,
                'pixel_to_mm_scale': pixel_scale,
                'fov_to_grid_pos': self._create_fov_to_grid_mapping(),
                'region_fov_mapping': self._create_region_fov_mapping(),
                'is_multi_region': False
            }
        
        if region_name:
            metadata['region_name'] = region_name
        
        if region_mappings:
            metadata['fov_grid'] = region_mappings['fov_grid']
            metadata['fov_to_grid_pos'] = region_mappings['fov_to_grid_pos']
        elif not self.data.region_grids:
            metadata['fov_grid'] = self.data.fov_grid
        
        return metadata
    
    def _create_fov_to_grid_mapping(self) -> Dict[int, Tuple[int, int]]:
        """Create mapping from FOV number to grid position"""
        if self.data.region_grids:
            # Multi-region: use unified grid
            return {fov: pos for pos, fov in self.data.fov_grid.items()}
        else:
            # Single region: use regular grid
            return {fov: pos for pos, fov in self.data.fov_grid.items()}
    
    def _create_region_fov_mapping(self) -> Dict[Tuple[str, int], Tuple[int, int]]:
        """Create mapping from (region_name, fov_num) to grid position"""
        mapping = {}
        
        if self.data.region_grids:
            # Multi-region case
            for region_name, region_grid in self.data.region_grids.items():
                row_offset, col_offset = self.data.region_grid_offsets[region_name]
                for (local_row, local_col), fov in region_grid.items():
                    global_pos = (local_row + row_offset, local_col + col_offset)
                    mapping[(region_name, fov)] = global_pos
        else:
            # Single region case
            for (row, col), fov in self.data.fov_grid.items():
                region_name = self.data.regions.get(fov, 'unknown')
                mapping[(region_name, fov)] = (row, col)
        
        return mapping
    
    def _create_region_specific_mappings(self, region_fovs: List[int]) -> Dict:
        """Create mappings that only include FOVs from the specified region"""
        region_fov_grid = {pos: fov for pos, fov in self.data.fov_grid.items() 
                          if fov in region_fovs}
        
        region_fov_to_pos = {fov: pos for pos, fov in region_fov_grid.items()}
        
        return {
            'fov_grid': region_fov_grid,
            'fov_to_grid_pos': region_fov_to_pos
        }
    
    def _report_progress(self, percent: int, message: str):
        """Report progress if callback is available"""
        if self.progress_callback:
            self.progress_callback(percent, message)
        print(f"[NAVIGATOR] {percent}% - {message}")
    
    def pixel_to_mm(self, pixel_x: int, pixel_y: int, region_name: Optional[str] = None) -> Tuple[float, float]:
        """Convert mosaic pixel coordinates to mm coordinates"""
        if self.data.region_grids and region_name:
            # Multi-region: use region-specific bounds
            if region_name not in self.data.region_grid_bounds:
                return (0.0, 0.0)
            
            bounds = self.data.region_grid_bounds[region_name]
            dims = self.data.region_grid_dims[region_name]
            row_offset, col_offset = self.data.region_grid_offsets[region_name]
            
            # Adjust pixel coordinates for region offset
            local_pixel_x = pixel_x - (col_offset * self.config.tile_size)
            local_pixel_y = pixel_y - (row_offset * self.config.tile_size)
            
            x_min, _, y_min, _ = bounds
            mm_per_pixel_x, mm_per_pixel_y = self.grid_builder.calculate_pixel_to_mm_scale(
                bounds, dims, self.config.tile_size
            )
            
            return (x_min + local_pixel_x * mm_per_pixel_x, y_min + local_pixel_y * mm_per_pixel_y)
        else:
            # Single region or no region specified
            if not self.data.grid_bounds:
                return (0.0, 0.0)
            
            x_min, _, y_min, _ = self.data.grid_bounds
            mm_per_pixel_x, mm_per_pixel_y = self.grid_builder.calculate_pixel_to_mm_scale(
                self.data.grid_bounds, self.data.grid_dims, self.config.tile_size
            )
            
            return (x_min + pixel_x * mm_per_pixel_x, y_min + pixel_y * mm_per_pixel_y)
    
    def mm_to_pixel(self, x_mm: float, y_mm: float, region_name: Optional[str] = None) -> Tuple[int, int]:
        """Convert mm coordinates to mosaic pixel coordinates"""
        if self.data.region_grids and region_name:
            # Multi-region: use region-specific bounds
            if region_name not in self.data.region_grid_bounds:
                return (0, 0)
            
            bounds = self.data.region_grid_bounds[region_name]
            dims = self.data.region_grid_dims[region_name]
            row_offset, col_offset = self.data.region_grid_offsets[region_name]
            
            x_min, _, y_min, _ = bounds
            mm_per_pixel_x, mm_per_pixel_y = self.grid_builder.calculate_pixel_to_mm_scale(
                bounds, dims, self.config.tile_size
            )
            
            local_pixel_x = int((x_mm - x_min) / mm_per_pixel_x)
            local_pixel_y = int((y_mm - y_min) / mm_per_pixel_y)
            
            # Add region offset
            return (local_pixel_x + col_offset * self.config.tile_size, 
                   local_pixel_y + row_offset * self.config.tile_size)
        else:
            # Single region or no region specified
            if not self.data.grid_bounds:
                return (0, 0)
            
            x_min, _, y_min, _ = self.data.grid_bounds
            mm_per_pixel_x, mm_per_pixel_y = self.grid_builder.calculate_pixel_to_mm_scale(
                self.data.grid_bounds, self.data.grid_dims, self.config.tile_size
            )
            
            return (int((x_mm - x_min) / mm_per_pixel_x), int((y_mm - y_min) / mm_per_pixel_y))


# Convenience functions
def create_navigation_mosaic(acquisition_dir: Path, timepoint: int = 0, 
                           tile_size: int = 75, cache_enabled: bool = True,
                           progress_callback: Optional[Callable] = None,
                           n_workers: Optional[int] = None) -> Tuple[np.ndarray, Dict]:
    """Create a navigation mosaic from an acquisition directory"""
    navigator = DownsampledNavigator(
        acquisition_dir=acquisition_dir,
        tile_size=tile_size,
        cache_enabled=cache_enabled,
        progress_callback=progress_callback,
        n_workers=n_workers
    )
    
    return navigator.create_mosaic(timepoint=timepoint)


def create_channel_navigation_mosaics(acquisition_dir: Path, region_name: str, 
                                    timepoint: int = 0, tile_size: int = 75, 
                                    cache_enabled: bool = True,
                                    progress_callback: Optional[Callable] = None,
                                    n_workers: Optional[int] = None) -> Tuple[Dict[str, np.ndarray], Dict]:
    """Create multi-channel navigation mosaics for a specific region"""
    navigator = DownsampledNavigator(
        acquisition_dir=acquisition_dir,
        tile_size=tile_size,
        cache_enabled=cache_enabled,
        progress_callback=progress_callback,
        n_workers=n_workers
    )
    
    return navigator.create_channel_mosaics_for_region(region_name=region_name, timepoint=timepoint)