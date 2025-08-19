import pandas as pd
import numpy as np
from skimage import io
from pathlib import Path
import sys
import subprocess
import tempfile
import os
import pickle
import re
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List, Sequence
import matplotlib.pyplot as plt
import random
import pandas as pd
from basicpy import BaSiC
from collections import defaultdict             

# Zarr and TiffSequence imports for FOV array creation
try:
    import tifffile as tf
    import zarr
    import xarray as xr
    import dask.array as da
    import dask
    from dask import delayed 
    
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False
    print("Warning: Zarr/tifffile/xarray/dask not available. Install with: pip install zarr tifffile xarray dask")

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QStatusBar, QSplitter)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, QTimer, QProcess, pyqtSignal, QPoint
from PIL import Image

# NDV imports for embedded viewer
try:
    import ndv
    NDV_AVAILABLE = True
except ImportError:
    NDV_AVAILABLE = False
    print("Warning: NDV not available. Install with: pip install ndv[vispy,pyqt]")

# Constants (same as original)
WELL_FORMATS = {4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12), 384: (16, 24), 1536: (32, 48)}
COLOR_MAPS = {'405': 'blue', '488': 'green', '561': 'yellow', '638': 'red', '640': 'red'}
COLOR_WEIGHTS = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1], 'yellow': [1, 1, 0], 'gray': [0.5, 0.5, 0.5]}

# Regular expression for the filename parsing (from simple.py)
fpattern = re.compile(
    r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.tiff?", re.IGNORECASE
)

def parse_filenames(filenames: Sequence[str]) -> tuple:
    """Parse a sequence of TIFF file paths to extract multi-dimensional metadata.
    
    From simple.py - adapted for our use case.
    """
    # List to hold (filename, metadata) tuples.
    parsed: List[Tuple[str, Tuple[int, str, int, int, str]]] = []

    for fname in filenames:
        path = Path(fname)
        try:
            t = int(path.parent.name)  # time from the immediate parent folder
        except ValueError as err:
            raise ValueError(
                f"Cannot parse time from parent folder of {fname}"
            ) from err

        if m := fpattern.search(path.name):
            # Extract groups; convert f and z to int.
            region = m.group("r")
            fov = int(m.group("f"))
            z_level = int(m.group("z"))
            channel = m.group("c")
            # Store full metadata as a tuple: (time, region, fov, z_level, channel)
            parsed.append((fname, (t, region, fov, z_level, channel)))

    # Sort by (time, region, fov, z_level, channel)
    parsed.sort(key=lambda x: x[1])

    # Separate filenames and metadata.
    sorted_files = [p[0] for p in parsed]
    metadata = [p[1] for p in parsed]

    # Gather unique values for each dimension.
    times = sorted({md[0] for md in metadata})
    regions = sorted({md[1] for md in metadata})
    fovs = sorted({md[2] for md in metadata})
    z_levels = sorted({md[3] for md in metadata})
    channels = sorted({md[4] for md in metadata})

    axes = ("time", "region", "fov", "z_level", "channel")
    shape = (len(times), len(regions), len(fovs), len(z_levels), len(channels))

    # Build indices as 5-tuples.
    indices = []
    for md in metadata:
        t_idx = times.index(md[0])
        r_idx = regions.index(md[1])
        f_idx = fovs.index(md[2])
        z_idx = z_levels.index(md[3])
        c_idx = channels.index(md[4])
        indices.append((t_idx, r_idx, f_idx, z_idx, c_idx))

    return axes, shape, indices, sorted_files

@dataclass
class TileData:
    image: np.ndarray
    x_mm: float
    y_mm: float
    file_path: str
    region: str
    fov: int
    wavelength: str

class ImageProcessor:
    @staticmethod
    def downsample_fast(img: np.ndarray, target_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h <= target_size and w <= target_size:
            result = np.zeros((target_size, target_size) + img.shape[2:], dtype=img.dtype)
            result[:min(h, target_size), :min(w, target_size)] = img[:min(h, target_size), :min(w, target_size)]
            return result
        
        factor = max(h, w) / target_size
        if factor > 4:
            kernel = int(factor)
            crop_h, crop_w = (h // kernel) * kernel, (w // kernel) * kernel
            cropped = img[:crop_h, :crop_w]
            if len(img.shape) == 3:
                reshaped = cropped.reshape(crop_h // kernel, kernel, crop_w // kernel, kernel, img.shape[2])
                return reshaped.mean(axis=(1, 3)).astype(img.dtype)
            else:
                reshaped = cropped.reshape(crop_h // kernel, kernel, crop_w // kernel, kernel)
                return reshaped.mean(axis=(1, 3)).astype(img.dtype)
        
        step = max(1, int(max(h, w) / target_size))
        return img[::step, ::step]

class TiffTileAssembler:
    def __init__(self, base_path: str, timepoint: int = 0, downsample_factor: float = 0.85):
        self.base_path = Path(base_path)
        self.timepoint = timepoint
        self.downsample_factor = downsample_factor
        self.cache_dir = self.base_path / "assembled_tiles_cache"
        self.cache_dir.mkdir(exist_ok=True)
        self.skip_downsampling = downsample_factor >= 0.98

    def _get_cache_key(self) -> str:
        suffix = "original" if self.skip_downsampling else f"ds{self.downsample_factor:.4f}"
        return f"t{self.timepoint}_{suffix}"

    def _get_flatfield_cache_key(self) -> str:
        return "flatfields_global"

    def _compute_flatfields(self) -> Dict[str, np.ndarray]:
        cache_key = self._get_flatfield_cache_key()
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            print("Loading cached flatfields...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        print("Computing flatfields from scratch...")
        wavelength_files = defaultdict(list)
        
        for timepoint_dir in self.base_path.iterdir():
            if not timepoint_dir.is_dir() or not timepoint_dir.name.isdigit():
                continue
                
            for filepath in timepoint_dir.glob("*.tiff"):
                parsed = self._parse_filename(filepath)
                if parsed:
                    wavelength = parsed[3]
                    wavelength_files[wavelength].append(filepath)
        
        flatfields = {}
        for wavelength, files in wavelength_files.items():
            print(f"Computing flatfield for {wavelength}nm...")
            sample_files = random.sample(files, min(48, len(files)))
            timepoint_count = len(set(f.parent.name for f in sample_files))
            print(f"  Using {len(sample_files)} images from {timepoint_count} timepoints")
            
            stack = []
            for i, filepath in enumerate(sample_files):
                if i % 10 == 0:
                    print(f"  Loading image {i+1}/{len(sample_files)}")
                img = io.imread(filepath).astype(np.float32)
                stack.append(img)
            
            stack = np.array(stack)
            print(f"  Fitting BaSiC model on stack shape: {stack.shape}")
            basic = BaSiC(get_darkfield=False, smoothness_flatfield=1)
            basic.fit(stack)
            flatfields[wavelength] = basic.flatfield.astype(np.float32)
            
            # Save flatfield as PNG for verification
            plt.figure(figsize=(10, 8))
            plt.imshow(flatfields[wavelength], cmap='viridis')
            plt.colorbar(label='Flatfield intensity')
            plt.title(f'Flatfield for {wavelength}nm')
            plt.tight_layout()
            png_path = self.cache_dir / f"flatfield_{wavelength}nm.png"
            plt.savefig(png_path, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"  Saved flatfield visualization: {png_path}")
            
            del stack
        
        with open(cache_file, 'wb') as f:
            pickle.dump(flatfields, f)
        print(f"Cached flatfields to {cache_file}")
        
        return flatfields

    def _apply_flatfield_correction(self, img: np.ndarray, flatfield: np.ndarray) -> np.ndarray:
        img_float = img.astype(np.float32)
        mean_ff = flatfield.mean()
        corrected = img_float / flatfield * mean_ff
        return np.clip(corrected, 0, np.iinfo(img.dtype).max).astype(img.dtype)

    def _cache_exists(self) -> bool:
        cache_key = self._get_cache_key()
        return all((self.cache_dir / f"{cache_key}_{name}").exists()
                  for name in ["pixel_map.pkl", "multichannel.tiff", "metadata.pkl"])

    def _load_from_cache(self) -> Tuple[Dict, Dict]:
        cache_key = self._get_cache_key()
        with open(self.cache_dir / f"{cache_key}_pixel_map.pkl", 'rb') as f:
            pixel_map = pickle.load(f)
        # Load the multichannel image and ensure correct shape
        multichannel_image = io.imread(self.cache_dir / f"{cache_key}_multichannel.tiff")
        print(f"Loaded cached image shape: {multichannel_image.shape}")
        # Check if the image needs reshaping (skimage might have transposed dimensions)
        if len(multichannel_image.shape) == 3:
            # If it's (height, width, channels), transpose to (channels, height, width)
            if multichannel_image.shape[2] <= 10:  # Assuming max 10 channels
                print(f"Transposing image from {multichannel_image.shape} to (channels, height, width)")
                multichannel_image = np.transpose(multichannel_image, (2, 0, 1))
                print(f"After transpose: {multichannel_image.shape}")
        with open(self.cache_dir / f"{cache_key}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        return {
            'multichannel': multichannel_image,
            'wavelengths': metadata['wavelengths'],
            'colormaps': metadata['colormaps']
        }, pixel_map

    def _save_to_cache(self, assembled_images: Dict, pixel_map: Dict):
        cache_key = self._get_cache_key()
        with open(self.cache_dir / f"{cache_key}_pixel_map.pkl", 'wb') as f:
            pickle.dump(pixel_map, f)
        multichannel_image = assembled_images['multichannel']
        print(f"Original multichannel image shape: {multichannel_image.shape}")
        # Ensure the image is in the correct format for saving
        # skimage.io.imsave expects (height, width, channels) for multichannel images
        if len(multichannel_image.shape) == 3 and multichannel_image.shape[0] <= 10:
            # If it's (channels, height, width), transpose to (height, width, channels)
            print(f"Transposing for saving from {multichannel_image.shape} to (height, width, channels)")
            multichannel_image = np.transpose(multichannel_image, (1, 2, 0))
            print(f"After transpose for saving: {multichannel_image.shape}")
        compression = 'lzw' if multichannel_image.dtype == np.uint16 else None
        io.imsave(self.cache_dir / f"{cache_key}_multichannel.tiff", multichannel_image, compression=compression)
        metadata = {k: assembled_images[k] for k in ['wavelengths', 'colormaps']}
        with open(self.cache_dir / f"{cache_key}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

    def _parse_filename(self, filepath: Path) -> Optional[Tuple]:
        match = re.match(r"([A-Z]+\d+)_(\d+)_(\d+)_.*_(\d+)_nm_Ex\.tiff", filepath.name)
        return match.groups() if match else None

    def _get_colormap(self, wavelength: str) -> str:
        return next((color for wl, color in COLOR_MAPS.items() if wl in wavelength), 'gray')

    def _load_tiles(self) -> Dict:
        # Compute/load flatfields first
        flatfields = self._compute_flatfields()
        
        timepoint_path = self.base_path / str(self.timepoint)
        coords_df = pd.read_csv(timepoint_path / "coordinates.csv")
        tiles = {}
        for filepath in timepoint_path.glob("*.tiff"):
            parsed = self._parse_filename(filepath)
            if not parsed:
                continue
            region, fov, z, wavelength = parsed
            coord_row = coords_df[(coords_df['region'] == region) & (coords_df['fov'] == int(fov))]
            if coord_row.empty:
                continue
                
            img = io.imread(filepath)
            
            # Apply flatfield correction BEFORE downsampling
            if wavelength in flatfields:
                img = self._apply_flatfield_correction(img, flatfields[wavelength])
            
            if not self.skip_downsampling and self.downsample_factor < 0.98:
                target_size = int(min(img.shape[:2]) * self.downsample_factor)
                if target_size < min(img.shape[:2]) - 1:
                    img = ImageProcessor.downsample_fast(img, target_size)
            tiles[(region, int(fov), wavelength)] = TileData(
                image=img,
                x_mm=coord_row['x (mm)'].iloc[0],
                y_mm=coord_row['y (mm)'].iloc[0],
                file_path=str(filepath),
                region=region,
                fov=int(fov),
                wavelength=wavelength
            )
        return tiles

    def _assemble_images(self, tiles: Dict) -> Tuple[Dict, Dict]:
        if not tiles:
            return {}, {}
        sample_img = next(iter(tiles.values())).image
        tile_h, tile_w = sample_img.shape[:2]
        unique_coords = {
            'x': sorted(set(t.x_mm for t in tiles.values())),
            'y': sorted(set(t.y_mm for t in tiles.values()))
        }
        coord_to_grid = {
            'x': {x: i for i, x in enumerate(unique_coords['x'])},
            'y': {y: i for i, y in enumerate(unique_coords['y'])}
        }
        canvas_w = len(unique_coords['x']) * tile_w
        canvas_h = len(unique_coords['y']) * tile_h
        wavelengths = sorted(set(key[2] for key in tiles.keys()))
        canvas = np.zeros((len(wavelengths), canvas_h, canvas_w), dtype=sample_img.dtype)
        pixel_map = {}
        for channel_idx, wavelength in enumerate(wavelengths):
            for (region, fov, wl), tile in tiles.items():
                if wl != wavelength:
                    continue
                x_pixel = coord_to_grid['x'][tile.x_mm] * tile_w
                y_pixel = coord_to_grid['y'][tile.y_mm] * tile_h
                canvas[channel_idx, y_pixel:y_pixel+tile_h, x_pixel:x_pixel+tile_w] = tile.image
                if channel_idx == 0:  # Create pixel mapping only once
                    for y in range(y_pixel, y_pixel + tile_h):
                        for x in range(x_pixel, x_pixel + tile_w):
                            pixel_map[(x, y)] = tile
        return {
            'multichannel': canvas,
            'wavelengths': wavelengths,
            'colormaps': [self._get_colormap(wl) for wl in wavelengths]
        }, pixel_map

    def get_assembled_images(self) -> Tuple[Dict, Dict]:
        if self._cache_exists():
            return self._load_from_cache()
        tiles = self._load_tiles()
        assembled_images, pixel_map = self._assemble_images(tiles)
        if assembled_images:
            self._save_to_cache(assembled_images, pixel_map)
        return assembled_images, pixel_map

    def clear_cache(self):
        """Clear the cache to force regeneration"""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        print(f"Cleared cache directory: {self.cache_dir}")
        self.cache_dir.mkdir(exist_ok=True)

class NDVLauncher:
    """Handles launching NDV in separate processes"""
    
    @staticmethod
    def launch_ndv_with_image(image_path: str):
        """Launch NDV with a specific image file"""
        try:
            # Create a Python script that will run NDV
            script_content = f'''
import ndv
from skimage import io
import sys

try:
    image = io.imread(r"{image_path}")
    ndv.imshow(image)
except Exception as e:
    print(f"Error in NDV: {{e}}")
    sys.exit(1)
'''
            
            # Create temporary script file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name
            
            # Launch the script in a separate Python process
            subprocess.Popen([sys.executable, script_path], 
                           creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == 'win32' else 0)
            
            # Clean up the temporary file after a delay
            QTimer.singleShot(2000, lambda: os.unlink(script_path) if os.path.exists(script_path) else None)
            
        except Exception as e:
            print(f"Error launching NDV: {e}")

class ClickableImageLabel(QLabel):
    """Custom QLabel that emits click events with coordinates"""
    clicked = pyqtSignal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
    
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.x(), event.y())

class TiffViewerWidget(QWidget):
    """Main widget for displaying the multichannel TIFF"""
    
    def __init__(self, base_path: str, timepoint: int = 0, downsample_factor: float = 0.85):
        super().__init__()
        self.base_path = base_path
        self.timepoint = timepoint
        self.downsample_factor = downsample_factor
        self.cache_dir = Path(base_path) / "assembled_tiles_cache"
        
        
        # Load data
        assembler = TiffTileAssembler(base_path, timepoint, downsample_factor)
        self.assembled_data, self.pixel_map = assembler.get_assembled_images()
        
        if not self.assembled_data or 'multichannel' not in self.assembled_data:
            print("No images found!")
            return
            
        self.multichannel_image = self.assembled_data['multichannel']
        self.wavelengths = self.assembled_data['wavelengths']
        self.colormaps = self.assembled_data['colormaps']
        
        # Store tile grid information for efficient coordinate lookup
        self._setup_tile_grid()
        
        self.setup_ui()
        self.update_image()

    def setup_ui(self):
        # Create main layout
        main_layout = QVBoxLayout()
        
        # Create splitter for side-by-side view
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Original image display
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignCenter)  # Center the entire left layout
        
        # Add top spacer to center the image vertically
        from PyQt5.QtWidgets import QSpacerItem, QSizePolicy
        top_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(top_spacer)
        
        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                background-color: transparent;
                border: 1px solid #333;
            }
        """)
        self.image_label.setMouseTracking(True)  # Ensure mouse tracking is enabled
        self.image_label.clicked.connect(self.on_image_click)
        left_layout.addWidget(self.image_label, 1)  # Add with stretch factor 1 to center it
        
        # Store the current clicked position for drawing the cyan dot
        self.clicked_position = None
        
        # Add bottom spacer to complete vertical centering
        bottom_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(bottom_spacer)
        
        # Status label
        self.status_label = QLabel("Click on a tile to open with NDV")
        left_layout.addWidget(self.status_label)
        
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # Right side: NDV embedded viewer
        if NDV_AVAILABLE:
            # Create NDV viewer with initial configuration
            # Start with a dummy array to initialize the viewer properly
            dummy_data = np.zeros((4, 100, 100), dtype=np.uint16)
            self.ndv_viewer = ndv.ArrayViewer(
                dummy_data,
                channel_axis=0,
                channel_mode="composite"
            )
            splitter.addWidget(self.ndv_viewer.widget())
        else:
            # Placeholder if NDV is not available
            placeholder = QLabel("NDV not available.\nInstall with: pip install ndv[vispy,pyqt]")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
            splitter.addWidget(placeholder)
        
        # Set splitter proportions (60% left, 40% right)
        splitter.setSizes([600, 400])
        
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def _setup_tile_grid(self):
        """Setup tile grid information for efficient coordinate lookup"""
        # Get tile dimensions from the first tile
        sample_tile = next(iter(self.pixel_map.values()))
        sample_img = sample_tile.image
        self.tile_h, self.tile_w = sample_img.shape[:2]
        
        # Get unique coordinates and create grid mapping
        unique_coords = {
            'x': sorted(set(t.x_mm for t in self.pixel_map.values())),
            'y': sorted(set(t.y_mm for t in self.pixel_map.values()))
        }
        
        self.coord_to_grid = {
            'x': {x: i for i, x in enumerate(unique_coords['x'])},
            'y': {y: i for i, y in enumerate(unique_coords['y'])}
        }
        
        self.grid_to_tile = {}
        for tile in self.pixel_map.values():
            grid_x = self.coord_to_grid['x'][tile.x_mm]
            grid_y = self.coord_to_grid['y'][tile.y_mm]
            self.grid_to_tile[(grid_x, grid_y)] = tile

    def _create_color_image(self) -> np.ndarray:
        channels, height, width = self.multichannel_image.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i, (channel, colormap) in enumerate(zip(self.multichannel_image, self.colormaps)):
            normalized = self._normalize_channel(channel)
            weights = COLOR_WEIGHTS.get(colormap, [0.5, 0.5, 0.5])
            for c in range(3):
                rgb_image[:, :, c] = np.clip(rgb_image[:, :, c] + normalized * weights[c], 0, 255)
        
        return rgb_image

    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        if channel.dtype == np.uint8:
            return channel
        elif channel.dtype == np.uint16:
            return (channel // 256).astype(np.uint8)
        else:
            p2, p98 = np.percentile(channel, (2, 98))
            if p98 > p2:
                normalized = np.clip((channel - p2) / (p98 - p2), 0, 1)
                return (normalized * 255).astype(np.uint8)
            return channel.astype(np.uint8)

    def update_image(self):
        display_image = self._create_color_image()
        
        # Convert to QImage and scale
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale image to fit display
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        
        # Add well plate labels to the scaled pixmap
        labeled_pixmap = self._add_well_plate_labels(scaled_pixmap)
        
        # Add cyan dot if there's a clicked position
        if self.clicked_position is not None:
            labeled_pixmap = self._add_cyan_dot(labeled_pixmap, self.clicked_position)
        
        self.image_label.setPixmap(labeled_pixmap)
        
        # Calculate scaling factors for coordinate mapping
        # Use the original scaled pixmap size (without labels) for scaling calculations
        self.scale_x = self.multichannel_image.shape[2] / scaled_pixmap.width()
        self.scale_y = self.multichannel_image.shape[1] / scaled_pixmap.height()

    def on_image_click(self, x, y):
        # Account for image centering offset and well plate labels
        pixmap = self.image_label.pixmap()
        if pixmap:
            # Calculate the offset due to centering
            label_size = self.image_label.size()
            pixmap_size = pixmap.size()
            
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            # Adjust coordinates for the offset
            adjusted_x = x - offset_x
            adjusted_y = y - offset_y
            
            # Account for well plate label offsets (50px left, 30px top)
            label_width = 50
            label_height = 30
            
            # Adjust for label offsets
            adjusted_x -= label_width
            adjusted_y -= label_height
            
            # Only process if we're within the actual image bounds
            if (0 <= adjusted_x < pixmap_size.width() - label_width and 
                0 <= adjusted_y < pixmap_size.height() - label_height):
                                # Convert click coordinates to image coordinates
                img_x = int(adjusted_x * self.scale_x)
                img_y = int(adjusted_y * self.scale_y)
                
                # Use tile grid for efficient lookup
                grid_x = img_x // self.tile_w
                grid_y = img_y // self.tile_h
                
                if (grid_x, grid_y) in self.grid_to_tile:
                    tile = self.grid_to_tile[(grid_x, grid_y)]
                    fov_number = tile.fov
                    region_name = tile.region
                    self.status_label.setText(f"Creating zarr array for FOV {fov_number} in region {region_name}")
                    
                    # Store the clicked position for drawing the cyan dot
                    self.clicked_position = (adjusted_x + label_width, adjusted_y + label_height)
                    
                    # Create zarr array for the FOV and load into NDV viewer
                    if NDV_AVAILABLE and hasattr(self, 'ndv_viewer'):
                        try:
                            fov_array = self.create_fov_zarr_array(fov_number, region_name)
                            if fov_array is not None:
                                self.set_ndv_data(fov_array)
                            else:
                                self.status_label.setText(f"Failed to create zarr array for FOV {fov_number}")
                        except Exception as e:
                            print(f"Error loading FOV zarr array into NDV viewer: {e}")
                            self.status_label.setText(f"Error loading FOV {fov_number}")
                    
                    # Update the image to show the cyan dot
                    self.update_image()
                else:
                    self.status_label.setText("No tile found at this location")
            else:
                self.status_label.setText("Click outside image area")
        else:
            self.status_label.setText("No image loaded")
    
    def _add_cyan_dot(self, pixmap, position):
        """Add a cyan dot at the specified position on the pixmap"""
        # Create a copy of the pixmap to draw on
        new_pixmap = pixmap.copy()
        painter = QPainter(new_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Set up the cyan pen and brush
        pen = QPen(Qt.red, 3)  # Cyan color, 3px width
        painter.setPen(pen)
        painter.setBrush(Qt.red)
        
        # Draw a circle at the clicked position
        x, y = position
        radius = 3 # Size of the dot
        painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
        
        painter.end()
        return new_pixmap
    
    def _calculate_well_format(self):
        """Calculate rows and columns based on region count following standard well plate formats"""
        # Get unique regions from pixel map
        regions = set()
        for tile in self.pixel_map.values():
            regions.add(tile.region)
        cell_count = len(regions)
        
        # Standard well plate formats
        format_map = {
            4: (2, 2),      # 4-well plate: 2x2
            6: (2, 3),      # 6-well plate: 2x3
            12: (3, 4),     # 12-well plate: 3x4
            24: (4, 6),     # 24-well plate: 4x6
            96: (8, 12),    # 96-well plate: 8x12
            384: (16, 24),  # 384-well plate: 16x24
            1536: (32, 48)  # 1536-well plate: 32x48
        }
        
        # Find the appropriate well plate format based on region count
        if cell_count <= 4:
            return format_map[4]  # 4-well plate
        elif cell_count <= 6:
            return format_map[6]  # 6-well plate
        elif cell_count <= 12:
            return format_map[12]  # 12-well plate
        elif cell_count <= 24:
            return format_map[24]  # 24-well plate
        elif cell_count <= 96:
            return format_map[96]  # 96-well plate
        elif cell_count <= 384:
            return format_map[384]  # 384-well plate
        elif cell_count <= 1536:
            return format_map[1536]  # 1536-well plate
        else:
            # For very large datasets, use 1536 format
            return format_map[1536]
    
    def _well_name_to_position(self, well_name):
        """Convert well name back to grid position"""
        # Extract row letters and column number
        import re
        match = re.match(r'([A-Z]+)(\d+)', well_name)
        if not match:
            return 0, 0
        row_letters, col_num = match.groups()
        col = int(col_num) - 1
        # Calculate row from letters
        if len(row_letters) == 1:
            row = ord(row_letters) - ord('A')
        else:
            # Handle AA, AB, etc.
            row = 0
            for i, char in enumerate(row_letters):
                row = row * 26 + (ord(char) - ord('A'))
            if len(row_letters) > 1:
                row += 26  # Offset for multi-letter combinations
        return row, col
    
    def _get_region_position(self, region_name):
        """Get the position for a region, assigning sequentially if not parseable"""
        # Try to parse existing region names first
        try:
            return self._well_name_to_position(region_name)
        except:
            # Assign sequential positions for unparseable names
            regions = sorted(list(set(tile.region for tile in self.pixel_map.values())))
            if region_name in regions:
                idx = regions.index(region_name)
                rows, cols = self._calculate_well_format()
                return idx // cols, idx % cols
            return 0, 0
    
    def _position_to_well_name(self, row, col):
        """Convert grid position to well name (A1, B3, AA5, etc.)"""
        # Handle row labels (A-Z, then AA-AZ, BA-BZ, etc.)
        if row < 26:
            row_label = chr(ord('A') + row)
        else:
            first_char = chr(ord('A') + (row // 26) - 1)
            second_char = chr(ord('A') + (row % 26))
            row_label = first_char + second_char
        # Column labels are 1-based numbers
        col_label = str(col + 1)
        return row_label + col_label
    
    def _get_well_name_for_region(self, region_name):
        """Get the well name (A1, B3, etc.) for a given region"""
        row, col = self._get_region_position(region_name)
        return self._position_to_well_name(row, col)
    
    def _add_well_plate_labels(self, pixmap):
        """Add well plate row and column labels to the pixmap"""
        # Calculate well plate format
        rows, cols = self._calculate_well_format()
        
        # Create a new pixmap with extra space for labels
        label_height = 30
        label_width = 50
        total_width = pixmap.width() + label_width
        total_height = pixmap.height() + label_height
        
        # Create new pixmap with transparent background
        labeled_pixmap = QPixmap(total_width, total_height)
        labeled_pixmap.fill(Qt.transparent)
        
        # Create painter to draw on the new pixmap
        painter = QPainter(labeled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw the original image in the center
        painter.drawPixmap(label_width, label_height, pixmap)
        
        # Calculate well dimensions
        well_width = pixmap.width() / cols
        well_height = pixmap.height() / rows
        
        # Set up font for labels
        font = painter.font()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(Qt.white, 2))
        
        # Draw column labels (numbers) at the top
        for col in range(cols):
            x = label_width + col * well_width + well_width / 2 + 10  # Offset to the right by 10px
            y = label_height / 2
            painter.drawText(
                int(x - 20), int(y - 12), 40, 24,
                Qt.AlignCenter,
                str(col + 1)
            )
        
        # Draw row labels (letters) on the left
        for row in range(rows):
            x = label_width / 2
            y = label_height + row * well_height + well_height / 2
            row_label = self._position_to_well_name(row, 0)[:-1]  # Remove column number
            painter.drawText(
                int(x - 25), int(y - 12), 50, 24,
                Qt.AlignCenter,
                row_label
            )
        
        painter.end()
        return labeled_pixmap
    
    def _get_channel_colormap(self, channel_idx):
        """Get appropriate colormap for each channel based on name/wavelength."""
        if channel_idx < len(self.wavelengths):
            channel_name = self.wavelengths[channel_idx].lower()
            
            # Map common wavelengths to colors
            if '405' in channel_name:
                return 'blue'
            elif '488' in channel_name:
                return 'green'  
            elif '561' in channel_name:
                return 'yellow'
            elif '638' in channel_name or '640' in channel_name:
                return 'red'
            else:
                return 'gray'
    
    def _create_ndv_luts(self):
        """Create LUT dictionary for NDV viewer based on channel wavelengths."""
        luts = {}
        for i, wavelength in enumerate(self.wavelengths):
            # Map wavelengths to specific NDV colormap names
            if '405' in wavelength:
                luts[i] = 'blue'
            elif '488' in wavelength:
                luts[i] = 'green'
            elif '561' in wavelength:
                luts[i] = 'yellow'
            elif '638' in wavelength or '640' in wavelength:
                luts[i] = 'red'
            else:
                # For unknown wavelengths, use wavelength-specific colors
                luts[i] = self._get_channel_colormap(i)
        return luts
    

    def set_ndv_data(self, data):
        """Enhanced version that handles lazy arrays properly."""
        if NDV_AVAILABLE and hasattr(self, 'ndv_viewer'):
            try:
                # Create LUTs for the channels
                luts = self._create_ndv_luts()
                
                # Find channel axis
                channel_axis = None
                if hasattr(data, 'dims'):
                    try:
                        channel_axis = data.dims.index('channel')
                    except ValueError:
                        pass
                
                print(f"Setting data with shape: {data.shape}")
                print(f"Channel axis: {channel_axis}")
                print(f"LUTs: {luts}")
                
                # Store current viewer state before updating
                current_state = self._capture_viewer_state()
                
                # TRY TO UPDATE EXISTING VIEWER FIRST (like original implementation)
                try:
                    # Try to update the data directly if the viewer supports it
                    if hasattr(self.ndv_viewer, 'set_data'):
                        self.ndv_viewer.set_data(data)
                    elif hasattr(self.ndv_viewer, 'data'):
                        self.ndv_viewer.data = data
                    else:
                        # Fallback: recreate viewer but preserve state
                        if hasattr(data, 'chunks'):  # It's a dask array
                            self._recreate_viewer_with_lazy_support(data, channel_axis, luts, current_state)
                        else:
                            self._recreate_viewer_with_state(data, channel_axis, luts, current_state)
                except Exception as e:
                    print(f"Direct data update failed, recreating viewer: {e}")
                    # Only recreate as fallback
                    if hasattr(data, 'chunks'):  # It's a dask array
                        self._recreate_viewer_with_lazy_support(data, channel_axis, luts, current_state)
                    else:
                        self._recreate_viewer_with_state(data, channel_axis, luts, current_state)
                
                # Restore viewer state and ensure proper configuration
                self._restore_viewer_state(current_state)
                self._ensure_composite_mode_and_luts(luts)
                
                print(f"Updated NDV viewer with channel_axis={channel_axis}, luts={luts}")
                
            except Exception as e:
                print(f"Error setting lazy data in NDV viewer: {e}")
                import traceback
                traceback.print_exc()
    
    
    def _capture_viewer_state(self):
        """Capture current viewer state (zoom, contrast, channel visibility, etc.)"""
        state = {}
        try:
            if hasattr(self.ndv_viewer, 'viewer'):
                viewer = self.ndv_viewer.viewer
                # Capture camera state (zoom, pan)
                if hasattr(viewer, 'camera'):
                    camera = viewer.camera
                    state['camera'] = {
                        'center': camera.center.copy() if hasattr(camera, 'center') else None,
                        'scale': camera.scale.copy() if hasattr(camera, 'scale') else None,
                        'rect': camera.rect.copy() if hasattr(camera, 'rect') else None
                    }
                
                # Capture layer visibility and contrast
                if hasattr(viewer, 'layers'):
                    state['layers'] = {}
                    for i, layer in enumerate(viewer.layers):
                        if hasattr(layer, 'visible'):
                            state['layers'][i] = {'visible': layer.visible}
                        if hasattr(layer, 'contrast_limits'):
                            state['layers'][i] = state['layers'].get(i, {})
                            state['layers'][i]['contrast_limits'] = layer.contrast_limits.copy()
                
                # Capture channel visibility if available
                if hasattr(viewer, 'channel_visibility'):
                    state['channel_visibility'] = viewer.channel_visibility.copy()
                
        except Exception as e:
            print(f"Error capturing viewer state: {e}")
        
        return state
    
    def _restore_viewer_state(self, state):
        """Restore viewer state from captured state"""
        try:
            if hasattr(self.ndv_viewer, 'viewer') and state:
                viewer = self.ndv_viewer.viewer
                
                # Restore camera state
                if 'camera' in state and hasattr(viewer, 'camera'):
                    camera = viewer.camera
                    camera_state = state['camera']
                    if camera_state.get('center') is not None and hasattr(camera, 'center'):
                        camera.center = camera_state['center']
                    if camera_state.get('scale') is not None and hasattr(camera, 'scale'):
                        camera.scale = camera_state['scale']
                    if camera_state.get('rect') is not None and hasattr(camera, 'rect'):
                        camera.rect = camera_state['rect']
                
                # Restore layer visibility and contrast
                if 'layers' in state and hasattr(viewer, 'layers'):
                    for i, layer_state in state['layers'].items():
                        if i < len(viewer.layers):
                            layer = viewer.layers[i]
                            if 'visible' in layer_state and hasattr(layer, 'visible'):
                                layer.visible = layer_state['visible']
                            if 'contrast_limits' in layer_state and hasattr(layer, 'contrast_limits'):
                                layer.contrast_limits = layer_state['contrast_limits']
                
                # Restore channel visibility
                if 'channel_visibility' in state and hasattr(viewer, 'channel_visibility'):
                    viewer.channel_visibility = state['channel_visibility']
                
        except Exception as e:
            print(f"Error restoring viewer state: {e}")
    
    def _recreate_viewer_with_lazy_support(self, data, channel_axis, luts, state):
        """Create viewer that understands the full dataset dimensions."""
        from PyQt5.QtWidgets import QVBoxLayout
        
        # Remove old viewer
        if hasattr(self, 'ndv_viewer'):
            old_widget = self.ndv_viewer.widget()
            old_widget.setParent(None)
        
        # Create new viewer with explicit dimension information
        if channel_axis is not None:
            self.ndv_viewer = ndv.ArrayViewer(
                data,
                channel_axis=channel_axis,
                channel_mode="composite",
                luts=luts
            )
            
            # If the viewer supports it, explicitly set dimension ranges
            # This ensures sliders are created even for lazy data
            if hasattr(self.ndv_viewer, 'set_dimension_ranges'):
                dim_ranges = {}
                for dim_name, coord in data.coords.items():
                    if dim_name in ['time', 'z_level']:
                        dim_ranges[dim_name] = (0, len(coord) - 1)
                self.ndv_viewer.set_dimension_ranges(dim_ranges)
        
        # Re-add to splitter
        splitter = self.parent().findChild(QSplitter)
        if splitter:
            splitter.addWidget(self.ndv_viewer.widget())
            splitter.setSizes([600, 400])
    
    def _ensure_composite_mode_and_luts(self, luts):
        """Ensure the viewer is in composite mode and has the correct LUTs applied"""
        try:
            if hasattr(self.ndv_viewer, 'viewer'):
                viewer = self.ndv_viewer.viewer
                
                # Set composite mode
                if hasattr(viewer, 'mode'):
                    viewer.mode = 'composite'
                elif hasattr(viewer, 'composite_mode'):
                    viewer.composite_mode = True
                
                # Apply LUTs to layers
                if hasattr(viewer, 'layers') and luts:
                    for i, layer in enumerate(viewer.layers):
                        if i in luts:
                            # Try different ways to set the colormap
                            if hasattr(layer, 'colormap'):
                                layer.colormap = luts[i]
                            elif hasattr(layer, 'cmap'):
                                layer.cmap = luts[i]
                            elif hasattr(layer, 'lut'):
                                layer.lut = luts[i]
                
                # Also try to set channel visibility to show all channels
                if hasattr(viewer, 'channel_visibility'):
                    viewer.channel_visibility = [True] * len(luts)
                
        except Exception as e:
            print(f"Error ensuring composite mode and LUTs: {e}")

    def create_fov_zarr_array(self, target_fov: int, target_region: str = None, flatfields: bool = True) -> Optional[xr.DataArray]:
        """Create a lazy-loading xarray DataArray that maintains full dimensions."""
        if not ZARR_AVAILABLE:
            print("Zarr/tifffile/xarray not available")
            return None
        
        try:
            # Load flatfields if requested and available
            flatfield_data = None
            wavelength_mapping = {}
            if flatfields:
                flatfields_file = self.cache_dir / "flatfields_global.pkl"
                print(f"Looking for flatfields at: {flatfields_file}")
                if flatfields_file.exists():
                    try:
                        with open(flatfields_file, 'rb') as f:
                            flatfield_data = pickle.load(f)
                        print(f"Loaded flatfields for wavelengths: {list(flatfield_data.keys())}")
                        
                        # Create wavelength mapping for flatfield correction
                        if hasattr(self, 'wavelengths'):
                            print(f"Available wavelengths in self.wavelengths: {self.wavelengths}")
                            for i, wavelength_name in enumerate(self.wavelengths):
                                # Extract numeric wavelength from names like "Fluorescence_561_nm_Ex"
                                import re
                                match = re.search(r'(\d{3})', wavelength_name)
                                if match:
                                    wavelength_str = match.group(1)  # Keep as string to match flatfield keys
                                    wavelength_mapping[i] = wavelength_str
                                    print(f"Mapped channel {i} ({wavelength_name}) -> flatfield key '{wavelength_str}'")
                            print(f"Final wavelength mapping: {wavelength_mapping}")
                        else:
                            print("Warning: self.wavelengths not found, cannot create wavelength mapping")
                            
                    except Exception as e:
                        print(f"Error loading flatfields: {e}")
                        flatfield_data = None
                else:
                    print("No flatfields_global.pkl found, proceeding without flatfield correction")
            else:
                print("Flatfield correction disabled")
                
            # Get all TIFF files from the base path (same as original)
            base_path = Path(self.base_path)
            all_tiffs = list(base_path.rglob("*.tiff"))
            if not all_tiffs:
                print(f"No TIFF files found in {base_path}")
                return None
            # Filter files for the target FOV and region (same as original)
            fov_files = []
            for tiff_path in all_tiffs:
                if m := fpattern.search(tiff_path.name):
                    fov = int(m.group("f"))
                    region = m.group("r")
                    if fov == target_fov:
                        if target_region is None or region == target_region:
                            fov_files.append(str(tiff_path))
            if not fov_files:
                print(f"No files found for FOV {target_fov}" + (f" in region {target_region}" if target_region else ""))
                return None
            # Use your original parse_filenames function to get the proper structure
            axes, shape, indices, sorted_files = parse_filenames(fov_files)
            print(f"Original axes: {axes}")
            print(f"Original shape: {shape}")
            print(f"Number of files: {len(sorted_files)}")
            # Get a sample file to determine spatial dimensions
            sample_tiff = tf.TiffFile(sorted_files[0])
            sample_shape = sample_tiff.pages[0].shape
            height, width = sample_shape[-2:]
            # Extract dimension information from the parsed results
            dim_info = {}
            for i, axis in enumerate(axes):
                dim_info[axis] = shape[i]
            print(f"Parsed dimensions: {dim_info}")
            print(f"Spatial dimensions: y={height}, x={width}")
            # Create a mapping from multi-dimensional indices to file paths
            file_lookup = {}
            for idx, file_path in enumerate(sorted_files):
                # Convert flat index to multi-dimensional coordinates
                coords = np.unravel_index(idx, shape)
                file_lookup[coords] = file_path
            print(f"Created file lookup with {len(file_lookup)} entries")
            
            # Create a lazy-loading function that returns delayed objects
            def load_single_file(file_path, coords=None):
                """Load a single file and return the data, with optional flatfield correction."""
                def _load():
                    try:
                        tiff = tf.TiffFile(file_path)
                        data = np.array(tiff.pages[0].asarray())
                        
                        # Apply flatfield correction if available and coords provided
                        if flatfield_data and coords is not None and len(axes) >= 5:
                            # Extract channel index from coordinates
                            # Assuming axes order is (time, region, fov, z_level, channel)
                            channel_idx = coords[4] if 'channel' in axes else None
                            
                            if channel_idx is not None and channel_idx in wavelength_mapping:
                                wavelength_str = wavelength_mapping[channel_idx]
                                if wavelength_str in flatfield_data:
                                    flatfield = flatfield_data[wavelength_str]
                                    print(f"Applying flatfield correction: channel {channel_idx} -> wavelength '{wavelength_str}'")
                                    # Apply correction: corrected = raw / flatfield
                                    data = data.astype(np.float32) / flatfield.astype(np.float32)
                                    # Convert back to original dtype, clipping to prevent overflow
                                    data = np.clip(data, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                                else:
                                    print(f"No flatfield found for wavelength '{wavelength_str}' (channel {channel_idx})")
                            else:
                                print(f"No wavelength mapping for channel {channel_idx}")
                        
                        return data
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        return np.zeros((height, width), dtype=np.uint16)
                return _load
                
            # Build the dask array structure properly
            # Create a function that constructs the array for given coordinates
            def build_dask_array():
                # Remove singleton dimensions (except time and z_level which we want to keep)
                non_singleton_axes = []
                non_singleton_shape = []
                for i, (axis, size) in enumerate(zip(axes, shape)):
                    if size > 1 or axis in ['time', 'z_level']:
                        non_singleton_axes.append(axis)
                        non_singleton_shape.append(size)
                # Add spatial dimensions
                non_singleton_axes.extend(['y', 'x'])
                non_singleton_shape.extend([height, width])
                print(f"Building array with axes: {non_singleton_axes}")
                print(f"Building array with shape: {non_singleton_shape}")
                # Create delayed arrays for each file
                delayed_arrays = []
                for coords, file_path in file_lookup.items():
                    # Create a delayed function for this file, passing coords for flatfield correction
                    delayed_load = delayed(load_single_file(file_path, coords))
                    # Convert to dask array
                    dask_chunk = da.from_delayed(
                        delayed_load(),
                        shape=(height, width),
                        dtype=np.uint16
                    )
                    delayed_arrays.append((coords, dask_chunk))
                # Now we need to arrange these into the proper multidimensional structure
                # Create a nested structure that matches the dimensions
                # For simplicity, let's build this step by step
                # First, let's create a dictionary indexed by the coordinates
                array_dict = {}
                for coords, dask_chunk in delayed_arrays:
                    array_dict[coords] = dask_chunk
                # Now build the nested structure
                if len(axes) == 5:  # time, region, fov, z_level, channel
                    time_arrays = []
                    for t in range(shape[0]):  # time
                        region_arrays = []
                        for r in range(shape[1]):  # region  
                            fov_arrays = []
                            for f in range(shape[2]):  # fov
                                z_arrays = []
                                for z in range(shape[3]):  # z_level
                                    channel_arrays = []
                                    for c in range(shape[4]):  # channel
                                        coord_key = (t, r, f, z, c)
                                        if coord_key in array_dict:
                                            channel_arrays.append(array_dict[coord_key])
                                        else:
                                            # Create a zero array if file doesn't exist
                                            zero_array = da.zeros((height, width), dtype=np.uint16)
                                            channel_arrays.append(zero_array)
                                    if channel_arrays:
                                        z_arrays.append(da.stack(channel_arrays, axis=0))
                                if z_arrays:
                                    fov_arrays.append(da.stack(z_arrays, axis=0))
                            if fov_arrays:
                                region_arrays.append(da.stack(fov_arrays, axis=0))
                        if region_arrays:
                            time_arrays.append(da.stack(region_arrays, axis=0))
                    if time_arrays:
                        full_array = da.stack(time_arrays, axis=0)
                    else:
                        # Fallback
                        full_array = da.zeros(shape + (height, width), dtype=np.uint16)
                # Remove singleton dimensions after construction
                squeeze_axes = []
                final_axes = []
                for i, (axis, size) in enumerate(zip(axes, shape)):
                    if size == 1 and axis not in ['time', 'z_level']:
                        squeeze_axes.append(i)
                    else:
                        final_axes.append(axis)
                # Squeeze out singleton dimensions
                if squeeze_axes:
                    full_array = da.squeeze(full_array, axis=tuple(squeeze_axes))
                final_axes.extend(['y', 'x'])
                return full_array, final_axes
            # Build the actual dask array
            full_array, final_axes = build_dask_array()
            print(f"Final array shape: {full_array.shape}")
            print(f"Final axes: {final_axes}")
            # Create coordinates
            coords_dict = {}
            for i, axis in enumerate(final_axes):
                if axis == 'time':
                    coords_dict[axis] = list(range(full_array.shape[i]))
                elif axis == 'z_level':
                    coords_dict[axis] = list(range(full_array.shape[i]))
                elif axis == 'channel':
                    coords_dict[axis] = [f"ch_{j}" for j in range(full_array.shape[i])]
                elif axis in ['y', 'x']:
                    size = height if axis == 'y' else width
                    coords_dict[axis] = list(range(size))
                else:
                    coords_dict[axis] = list(range(full_array.shape[i]))
            # Create xarray DataArray
            xarray_data = xr.DataArray(
                full_array,
                dims=final_axes,
                coords=coords_dict,
                name="image_data"
            )
            
            correction_status = "with flatfield correction" if flatfield_data else "without flatfield correction"
            print(f"Created lazy xarray DataArray with shape: {xarray_data.shape} {correction_status}")
            return xarray_data
        except Exception as e:
            print(f"Error creating lazy FOV xarray: {e}")
            import traceback
            traceback.print_exc()
            return None
            
class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self, base_path: str, timepoint: int = 0, downsample_factor: float = 0.85):
        super().__init__()
        self.setWindowTitle("Multi-Channel TIFF Viewer with Embedded NDV")
        self.setGeometry(100, 100, 1400, 800)
        
        # Create central widget
        self.tiff_viewer = TiffViewerWidget(base_path, timepoint, downsample_factor)
        self.setCentralWidget(self.tiff_viewer)
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Click on tiles to view in embedded NDV viewer")

def main():
    app = QApplication(sys.argv)
    
    # Configuration
    base_path = "/Volumes/Extreme Pro/broad_6_well_plate_10x_Nt_=_10_2025-07-26_22-43-02.153728"
    
    # Create main window
    
    main_window = MainWindow(base_path, timepoint=0, downsample_factor=0.01)
    main_window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
