"""Viewer widget with NDV integration for viewing assembled plates"""

import pickle
import re
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
from skimage import io
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSplitter, 
                             QSpacerItem, QSizePolicy, QMainWindow, QStatusBar)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal

from .common import (COLOR_WEIGHTS, parse_filenames, fpattern, fpattern_ome, 
                     detect_acquisition_format, detect_hcs_vs_normal_tissue)
from .plate_stack import PlateStackManager, StackBuilderThread, NDVSyncController, NDVContrastSyncController

# NDV and zarr imports
try:
    import ndv
    NDV_AVAILABLE = True
except ImportError:
    NDV_AVAILABLE = False

try:
    import tifffile as tf
    import xarray as xr
    import dask.array as da
    from dask import delayed
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


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
        # Use consistent cache directory - in downsampled_image folder
        self.cache_dir = Path(base_path) / "downsampled_image" / "assembled_tiles_cache"
        
        # Detect dataset type: HCS/wellplate vs normal tissue
        self.is_hcs_dataset = detect_hcs_vs_normal_tissue(Path(base_path))
        print(f"\n{'='*60}")
        print(f"Dataset Type: {'HCS/Wellplate' if self.is_hcs_dataset else 'Normal Tissue'}")
        print(f"Plate View: {'ENABLED' if self.is_hcs_dataset else 'DISABLED'}")
        print(f"{'='*60}\n")
        
        # Initialize plate-related attributes (only used for HCS datasets)
        self.plate_stack = None
        self.ndv_sync = None
        self.ndv_contrast_sync = None
        self.stack_loaded = False
        self._plate_contrast_limits = {}
        self.assembled_data = None
        self.tile_map = None
        self.multichannel_image = None
        self.wavelengths = None
        self.colormaps = None
        self.clicked_position = None
        
        # Only assemble plate view for HCS datasets
        if self.is_hcs_dataset:
            # Initialize plate stack manager
            self.plate_stack = PlateStackManager(base_path, self.cache_dir)
            
            # Load data from cache
            from .preprocessing import PlateAssembler
            assembler = PlateAssembler(base_path, timepoint)
            target_px = int(assembler._get_original_pixel_size() * downsample_factor)
            self.assembled_data, self.tile_map = assembler.assemble_plate(target_px)
            
            if not self.assembled_data or 'multichannel' not in self.assembled_data:
                print("No images found!")
                return
            
            self.multichannel_image = self.assembled_data['multichannel']
            self.wavelengths = self.assembled_data['wavelengths']
            self.colormaps = self.assembled_data['colormaps']
            
            self._setup_tile_grid()
            self.setup_ui()
            self.update_image()
            
            # Check if Z×T stack exists, if not build it silently in background
            if not self.plate_stack.exists(downsample_factor):
                print("Z×T stack not found, building in background...")
                self._build_stack_async()
            else:
                self._load_stack()
        else:
            # Normal tissue: Skip plate assembly, show NDV only
            print("Normal tissue dataset detected - showing FOV viewer only (no plate view)")
            self.setup_ui_simple()
            # Optionally load first FOV automatically
            self._load_first_fov()

    def setup_ui(self):
        """Setup UI for HCS datasets with plate view on left, NDV on right"""
        main_layout = QVBoxLayout()
        splitter = QSplitter(Qt.Horizontal)
        
        # Left side: Original image display
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setAlignment(Qt.AlignCenter)
        
        top_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(top_spacer)
        
        self.image_label = ClickableImageLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("QLabel { background-color: transparent; border: 1px solid #333; }")
        self.image_label.setMouseTracking(True)
        self.image_label.clicked.connect(self.on_image_click)
        left_layout.addWidget(self.image_label, 1)
        
        bottom_spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        left_layout.addItem(bottom_spacer)
        
        self.status_label = QLabel("Click on a tile to open with NDV")
        left_layout.addWidget(self.status_label)
        
        left_widget.setLayout(left_layout)
        splitter.addWidget(left_widget)
        
        # Right side: NDV embedded viewer
        if NDV_AVAILABLE:
            dummy_data = np.zeros((4, 100, 100), dtype=np.uint16)
            # Set visible_axes to the spatial dims only (not channel)
            self.ndv_viewer = ndv.ArrayViewer(dummy_data, channel_axis=0, channel_mode="composite", visible_axes=(-2, -1))
            splitter.addWidget(self.ndv_viewer.widget())
        else:
            placeholder = QLabel("NDV not available.\nInstall with: pip install ndv[vispy,pyqt]")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
            splitter.addWidget(placeholder)
        
        splitter.setSizes([600, 400])
        main_layout.addWidget(splitter)
        self.setLayout(main_layout)
    
    def setup_ui_simple(self):
        """Setup simple UI for normal tissue datasets (NDV viewer only)"""
        main_layout = QVBoxLayout()
        
        # Status label at top
        self.status_label = QLabel("Normal Tissue Dataset - FOV Viewer Mode")
        self.status_label.setStyleSheet("QLabel { padding: 10px; background-color: #2a2a2a; color: white; }")
        main_layout.addWidget(self.status_label)
        
        # NDV viewer takes full space
        if NDV_AVAILABLE:
            dummy_data = np.zeros((4, 100, 100), dtype=np.uint16)
            self.ndv_viewer = ndv.ArrayViewer(dummy_data, channel_axis=0, channel_mode="composite", visible_axes=(-2, -1))
            main_layout.addWidget(self.ndv_viewer.widget())
        else:
            placeholder = QLabel("NDV not available.\nInstall with: pip install ndv[vispy,pyqt]")
            placeholder.setAlignment(Qt.AlignCenter)
            placeholder.setStyleSheet("background-color: #f0f0f0; border: 1px solid #ccc;")
            main_layout.addWidget(placeholder)
        
        self.setLayout(main_layout)
    
    def _load_first_fov(self):
        """Load ALL FOVs for normal tissue datasets to enable FOV slider navigation"""
        if not NDV_AVAILABLE:
            return
        
        try:
            self.status_label.setText("Loading all FOVs for navigation...")
            
            # Load all FOVs into a multi-FOV xarray
            all_fovs_array = self.create_all_fovs_zarr_array()
            
            if all_fovs_array is not None:
                self.set_ndv_data(all_fovs_array)
                
                # Get FOV count from array
                fov_dim_idx = all_fovs_array.dims.index('fov') if 'fov' in all_fovs_array.dims else None
                if fov_dim_idx is not None:
                    n_fovs = all_fovs_array.shape[fov_dim_idx]
                    self.status_label.setText(f"Loaded {n_fovs} FOVs - Use FOV slider to navigate")
                else:
                    self.status_label.setText("Dataset loaded - Use sliders to navigate")
            else:
                self.status_label.setText("Failed to load dataset")
                
        except Exception as e:
            print(f"Error loading FOVs: {e}")
            import traceback
            traceback.print_exc()
            self.status_label.setText("Error loading dataset")
    
    def _setup_tile_grid(self):
        """Setup tile grid information, FOV boundaries, and well boundaries"""
        from types import SimpleNamespace
        
        # Extract tile dimensions from tile_map
        self.tile_h, self.tile_w = self.tile_map['tile_dimensions']
        
        # Build FOV boundary map: (region, fov) -> {x, y, w, h, center_x, center_y}
        self.fov_boundaries = {}
        
        # Track region bounds for well-level grid
        region_bounds = {}  # region -> {'min_x', 'max_x', 'min_y', 'max_y'}
        
        # Convert tile metadata and build spatial index
        self.grid_to_tile = {}
        for (grid_x, grid_y), tile_info in self.tile_map['grid_to_tile'].items():
            tile_obj = SimpleNamespace(
                region=tile_info['region'],
                fov=tile_info['fov'],
                x_mm=tile_info['x_mm'],
                y_mm=tile_info['y_mm'],
                x_pixel=tile_info['x_pixel'],
                y_pixel=tile_info['y_pixel']
            )
            self.grid_to_tile[(grid_x, grid_y)] = tile_obj
            
            # Build FOV boundary
            key = (tile_info['region'], tile_info['fov'])
            x, y = tile_info['x_pixel'], tile_info['y_pixel']
            self.fov_boundaries[key] = {
                'x': x,
                'y': y,
                'w': self.tile_w,
                'h': self.tile_h,
                'center_x': x + self.tile_w // 2,
                'center_y': y + self.tile_h // 2
            }
            
            # Update region bounds
            region = tile_info['region']
            if region not in region_bounds:
                region_bounds[region] = {
                    'min_x': x,
                    'max_x': x + self.tile_w,
                    'min_y': y,
                    'max_y': y + self.tile_h
                }
            else:
                region_bounds[region]['min_x'] = min(region_bounds[region]['min_x'], x)
                region_bounds[region]['max_x'] = max(region_bounds[region]['max_x'], x + self.tile_w)
                region_bounds[region]['min_y'] = min(region_bounds[region]['min_y'], y)
                region_bounds[region]['max_y'] = max(region_bounds[region]['max_y'], y + self.tile_h)
        
        # Store well boundaries (one bounding box per region)
        self.well_boundaries = region_bounds
        
        # Store FOV file index for O(1) lookup
        self.fov_to_files = self.tile_map['fov_to_files']

    def _create_color_image(self) -> np.ndarray:
        """Create RGB composite image from multichannel data (HCS only)"""
        if self.multichannel_image is None:
            return np.zeros((100, 100, 3), dtype=np.uint8)
        
        channels, height, width = self.multichannel_image.shape
        rgb_image = np.zeros((height, width, 3), dtype=np.float32)
        
        for i, (channel, colormap) in enumerate(zip(self.multichannel_image, self.colormaps)):
            # Keep in uint16 space, apply contrast limits
            # Try to get contrast limits - check cache first, then read from NDV
            if i not in self._plate_contrast_limits and hasattr(self, 'ndv_viewer'):
                try:
                    lut = self.ndv_viewer.display_model.luts.get(i)
                    if lut and hasattr(lut, 'clims'):
                        clims = lut.clims
                        if hasattr(clims, 'computed'):
                            vmin, vmax = clims.computed()
                            self._plate_contrast_limits[i] = (float(vmin), float(vmax))
                except:
                    pass
            
            if i in self._plate_contrast_limits:
                vmin, vmax = self._plate_contrast_limits[i]
                # Clip to contrast limits (uint16 range)
                clipped = np.clip(channel.astype(np.float32), vmin, vmax)
                # Normalize to [0, 1] in 16-bit space
                if vmax > vmin:
                    normalized_float = (clipped - vmin) / (vmax - vmin)
                else:
                    normalized_float = np.zeros_like(clipped, dtype=np.float32)
            else:
                # Auto-contrast in uint16 space
                if channel.dtype == np.uint16:
                    p2, p98 = np.percentile(channel, (2, 98))
                    if p98 > p2:
                        normalized_float = np.clip((channel.astype(np.float32) - p2) / (p98 - p2), 0, 1)
                    else:
                        normalized_float = channel.astype(np.float32) / 65535.0
                else:
                    normalized_float = channel.astype(np.float32) / 255.0
            
            # Apply color weights - NDV uses additive blending
            weights = COLOR_WEIGHTS.get(colormap, [0.5, 0.5, 0.5])
            for c in range(3):
                rgb_image[:, :, c] += normalized_float * weights[c]
        
        # Clip and convert to uint8 for display
        return (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)

    def _normalize_channel_with_limits_uint16(self, channel: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
        """
        Normalize uint16 channel with explicit contrast limits from NDV.
        NDV provides limits in uint16 range (0-65535), we map to uint8 display (0-255).
        """
        # Ensure we're working with the right data type
        if channel.dtype != np.uint16:
            channel = channel.astype(np.uint16)
        
        # Clip to the uint16 contrast range
        clipped = np.clip(channel, vmin, vmax)
        
        # Normalize to [0, 1]
        if vmax > vmin:
            normalized = (clipped.astype(np.float32) - vmin) / (vmax - vmin)
        else:
            normalized = np.zeros_like(clipped, dtype=np.float32)
        
        # Scale to [0, 255] for RGB display
        return (normalized * 255).astype(np.uint8)
    
    def _normalize_channel(self, channel: np.ndarray) -> np.ndarray:
        """Auto-contrast normalization (fallback when no NDV limits set)"""
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
        """Update plate image display (HCS only)"""
        if not self.is_hcs_dataset or self.multichannel_image is None:
            return
        
        display_image = self._create_color_image()
        height, width, channel = display_image.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_image.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        labeled_pixmap = self._add_well_plate_labels(scaled_pixmap)
        
        if self.clicked_position is not None:
            labeled_pixmap = self._add_cyan_dot(labeled_pixmap, self.clicked_position)
        
        self.image_label.setPixmap(labeled_pixmap)
        self.scale_x = self.multichannel_image.shape[2] / scaled_pixmap.width()
        self.scale_y = self.multichannel_image.shape[1] / scaled_pixmap.height()

    def on_image_click(self, x, y):
        """Handle click on plate image - find FOV and center red dot (HCS only)"""
        if not self.is_hcs_dataset:
            return
        
        pixmap = self.image_label.pixmap()
        if not pixmap:
            self.status_label.setText("No image loaded")
            return
        
        label_size = self.image_label.size()
        pixmap_size = pixmap.size()
        
        offset_x = (label_size.width() - pixmap_size.width()) // 2
        offset_y = (label_size.height() - pixmap_size.height()) // 2
        
        adjusted_x = x - offset_x - 50  # Account for well plate labels
        adjusted_y = y - offset_y - 30
        
        if not (0 <= adjusted_x < pixmap_size.width() - 50 and 
                0 <= adjusted_y < pixmap_size.height() - 30):
            self.status_label.setText("Click outside image area")
            return
        
        # Map to full-resolution image coordinates
        img_x = int(adjusted_x * self.scale_x)
        img_y = int(adjusted_y * self.scale_y)
        
        # Find which FOV was clicked using boundaries
        clicked_fov = None
        for (region, fov), bounds in self.fov_boundaries.items():
            if (bounds['x'] <= img_x < bounds['x'] + bounds['w'] and
                bounds['y'] <= img_y < bounds['y'] + bounds['h']):
                clicked_fov = (region, fov)
                break
        
        if not clicked_fov:
            self.status_label.setText("No tile found at this location")
            return
        
        region_name, fov_number = clicked_fov
        bounds = self.fov_boundaries[clicked_fov]
        
        # Calculate display-space center position
        center_x_scaled = bounds['center_x'] / self.scale_x + 50
        center_y_scaled = bounds['center_y'] / self.scale_y + 30
        self.clicked_position = (center_x_scaled, center_y_scaled)
        
        self.status_label.setText(f"Loading FOV {fov_number} in region {region_name}")
        
        if NDV_AVAILABLE and hasattr(self, 'ndv_viewer'):
            try:
                fov_array = self.create_fov_zarr_array(fov_number, region_name)
                if fov_array is not None:
                    self.set_ndv_data(fov_array)
                else:
                    self.status_label.setText(f"Failed to load FOV {fov_number}")
            except Exception as e:
                print(f"Error loading FOV: {e}")
                self.status_label.setText(f"Error loading FOV {fov_number}")
        
        self.update_image()
    
    def _add_cyan_dot(self, pixmap, position):
        """Add a red dot at the specified position on the pixmap"""
        new_pixmap = pixmap.copy()
        painter = QPainter(new_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        
        pen = QPen(Qt.red, 3)
        painter.setPen(pen)
        painter.setBrush(Qt.red)
        
        x, y = position
        radius = 3
        painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
        painter.end()
        return new_pixmap
    
    def _calculate_well_format(self):
        """Calculate rows and columns based on region count following standard well plate formats"""
        regions = set(tile.region for tile in self.grid_to_tile.values())
        cell_count = len(regions)
        
        if cell_count <= 4: return (2, 2)
        elif cell_count <= 6: return (2, 3)
        elif cell_count <= 12: return (3, 4)
        elif cell_count <= 24: return (4, 6)
        elif cell_count <= 96: return (8, 12)
        elif cell_count <= 384: return (16, 24)
        else: return (32, 48)
    
    def _position_to_well_name(self, row, col):
        """Convert grid position to well name (A1, B3, AA5, etc.)"""
        if row < 26:
            row_label = chr(ord('A') + row)
        else:
            first_char = chr(ord('A') + (row // 26) - 1)
            second_char = chr(ord('A') + (row % 26))
            row_label = first_char + second_char
        return row_label + str(col + 1)
    
    def _add_well_plate_labels(self, pixmap):
        """Add well plate row/column labels and FOV grid lines to the pixmap"""
        rows, cols = self._calculate_well_format()
        label_height, label_width = 30, 50
        total_width = pixmap.width() + label_width
        total_height = pixmap.height() + label_height
        
        labeled_pixmap = QPixmap(total_width, total_height)
        labeled_pixmap.fill(Qt.transparent)
        
        painter = QPainter(labeled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(label_width, label_height, pixmap)
        
        # Calculate scale factor for drawing
        scale_x = pixmap.width() / self.multichannel_image.shape[2]
        scale_y = pixmap.height() / self.multichannel_image.shape[1]
        
        # Draw white grid lines at well (region) boundaries
        painter.setPen(QPen(Qt.white, 1))
        drawn_vertical = set()
        drawn_horizontal = set()
        
        for region, bounds in self.well_boundaries.items():
            # Right edge of well
            x_right = int(bounds['max_x'] * scale_x) + label_width
            if x_right not in drawn_vertical and x_right < total_width:
                painter.drawLine(x_right, label_height, x_right, total_height)
                drawn_vertical.add(x_right)
            
            # Bottom edge of well
            y_bottom = int(bounds['max_y'] * scale_y) + label_height
            if y_bottom not in drawn_horizontal and y_bottom < total_height:
                painter.drawLine(label_width, y_bottom, total_width, y_bottom)
                drawn_horizontal.add(y_bottom)
        
        # Draw well labels
        well_width = pixmap.width() / cols
        well_height = pixmap.height() / rows
        
        font = painter.font()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QPen(Qt.white, 2))
        
        # Column labels
        for col in range(cols):
            x = label_width + col * well_width + well_width / 2 + 10
            y = label_height / 2
            painter.drawText(int(x - 20), int(y - 12), 40, 24, Qt.AlignCenter, str(col + 1))
        
        # Row labels
        for row in range(rows):
            x = label_width / 2
            y = label_height + row * well_height + well_height / 2
            row_label = self._position_to_well_name(row, 0)[:-1]
            painter.drawText(int(x - 25), int(y - 12), 50, 24, Qt.AlignCenter, row_label)
        
        painter.end()
        return labeled_pixmap
    
    def _create_ndv_luts(self):
        """Create LUT dictionary for NDV viewer based on channel wavelengths"""
        luts = {}
        if self.wavelengths is None:
            # Default LUTs for datasets without wavelength info
            # Use standard matplotlib colormaps that NDV recognizes
            default_colors = ['blue', 'green', 'yellow', 'red', 'Reds']
            for i in range(5):  # Support up to 5 channels by default
                luts[i] = default_colors[i]
            return luts
        
        for i, wavelength in enumerate(self.wavelengths):
            luts[i] = self._get_channel_colormap_from_name(wavelength, i)
        return luts
    
    def _get_channel_colormap_from_name(self, channel_name: str, index: int) -> str:
        """Get colormap for a channel based on its name or wavelength.
        
        Based on wavelength detection:
        - 405nm -> blue
        - 488nm -> green  
        - 561nm -> yellow
        - 638/640nm -> red
        - 730nm -> magenta
        - _B suffix -> blue
        - _G suffix -> green
        - _R suffix -> red
        """
        name_upper = channel_name.upper()
        
        # Check for wavelength numbers
        if '405' in name_upper:
            return 'blue'
        elif '488' in name_upper:
            return 'green'
        elif '561' in name_upper:
            return 'yellow'
        elif '638' in name_upper or '640' in name_upper:
            return 'red'
        elif '730' in name_upper:
            return 'Reds'  # Sequential colormap for far-red/near-infrared
        # Check for suffix patterns
        elif name_upper.endswith('_B'):
            return 'blue'
        elif name_upper.endswith('_G'):
            return 'green'
        elif name_upper.endswith('_R'):
            return 'red'
        # Default color based on index
        else:
            default_colors = ['blue', 'green', 'yellow', 'red', 'Reds']
            return default_colors[index] if index < len(default_colors) else 'gray'

    def set_ndv_data(self, data):
        """Enhanced version that handles lazy arrays properly"""
        if NDV_AVAILABLE and hasattr(self, 'ndv_viewer'):
            try:
                # Get LUTs - prefer from xarray attrs (OME), fallback to colormaps (single-TIFF)
                if hasattr(data, 'attrs') and 'luts' in data.attrs:
                    luts = data.attrs['luts']
                else:
                    luts = self._create_ndv_luts()
                
                channel_axis = None
                if hasattr(data, 'dims'):
                    try:
                        channel_axis = data.dims.index('channel')
                    except ValueError:
                        pass
                
                
                current_state = self._capture_viewer_state()
                
                try:
                    if hasattr(self.ndv_viewer, 'set_data'):
                        self.ndv_viewer.set_data(data)
                    elif hasattr(self.ndv_viewer, 'data'):
                        self.ndv_viewer.data = data
                    else:
                        # Always recreate viewer to ensure proper dimension handling
                        self._recreate_viewer_with_lazy_support(data, channel_axis, luts, current_state)
                except Exception as e:
                    print(f"Direct data update failed, recreating viewer: {e}")
                    self._recreate_viewer_with_lazy_support(data, channel_axis, luts, current_state)
                
                self._restore_viewer_state(current_state)
                self._ensure_composite_mode_and_luts(luts)
                
                # Initialize NDV sync controller if not already done
                if self.ndv_sync is None and self.stack_loaded:
                    self._init_ndv_sync()
                
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
                if hasattr(viewer, 'camera'):
                    camera = viewer.camera
                    state['camera'] = {
                        'center': camera.center.copy() if hasattr(camera, 'center') else None,
                        'scale': camera.scale.copy() if hasattr(camera, 'scale') else None,
                        'rect': camera.rect.copy() if hasattr(camera, 'rect') else None
                    }
                if hasattr(viewer, 'layers'):
                    state['layers'] = {}
                    for i, layer in enumerate(viewer.layers):
                        if hasattr(layer, 'visible'):
                            state['layers'][i] = {'visible': layer.visible}
                        if hasattr(layer, 'contrast_limits'):
                            state['layers'][i] = state['layers'].get(i, {})
                            state['layers'][i]['contrast_limits'] = layer.contrast_limits.copy()
        except Exception as e:
            print(f"Error capturing viewer state: {e}")
        return state
    
    def _restore_viewer_state(self, state):
        """Restore viewer state from captured state"""
        try:
            if hasattr(self.ndv_viewer, 'viewer') and state:
                viewer = self.ndv_viewer.viewer
                if 'camera' in state and hasattr(viewer, 'camera'):
                    camera = viewer.camera
                    camera_state = state['camera']
                    if camera_state.get('center') is not None and hasattr(camera, 'center'):
                        camera.center = camera_state['center']
                    if camera_state.get('scale') is not None and hasattr(camera, 'scale'):
                        camera.scale = camera_state['scale']
                    if camera_state.get('rect') is not None and hasattr(camera, 'rect'):
                        camera.rect = camera_state['rect']
                if 'layers' in state and hasattr(viewer, 'layers'):
                    for i, layer_state in state['layers'].items():
                        if i < len(viewer.layers):
                            layer = viewer.layers[i]
                            if 'visible' in layer_state and hasattr(layer, 'visible'):
                                layer.visible = layer_state['visible']
                            if 'contrast_limits' in layer_state and hasattr(layer, 'contrast_limits'):
                                layer.contrast_limits = layer_state['contrast_limits']
        except Exception as e:
            print(f"Error restoring viewer state: {e}")
    
    def _recreate_viewer_with_lazy_support(self, data, channel_axis, luts, state):
        """Create viewer that understands the full dataset dimensions"""
        if hasattr(self, 'ndv_viewer'):
            old_widget = self.ndv_viewer.widget()
            old_widget.setParent(None)
        
        # Determine visible_axes (should be just the 2D spatial display dimensions)
        # This tells NDV which axes to display in the canvas.
        # All other axes (time, z_level, etc.) will automatically get sliders.
        visible_axes = ('y', 'x')  # Default to 2D display
        if hasattr(data, 'dims'):
            # Always use just y, x for 2D display so time and z get sliders
            visible_axes = tuple(d for d in ['y', 'x'] if d in data.dims)
            slider_dims = [d for d in data.dims if d not in visible_axes]
            if channel_axis is not None and channel_axis < len(data.dims):
                channel_dim_name = data.dims[channel_axis]
                slider_dims = [d for d in slider_dims if d != channel_dim_name]
        
        if channel_axis is not None:
            self.ndv_viewer = ndv.ArrayViewer(data, channel_axis=channel_axis, 
                                             channel_mode="composite", luts=luts,
                                             visible_axes=visible_axes)
        
        # Add viewer back to UI - handle both HCS and normal tissue layouts
        if self.is_hcs_dataset:
            # HCS mode: viewer in right panel of splitter
            parent = self.parent()
            if parent:
                splitter = parent.findChild(QSplitter)
                if splitter:
                    splitter.addWidget(self.ndv_viewer.widget())
                    splitter.setSizes([600, 400])
        else:
            # Normal tissue mode: viewer takes full layout (after status label)
            layout = self.layout()
            if layout and layout.count() >= 1:
                # Insert viewer after status label (which should be at index 0)
                layout.insertWidget(1, self.ndv_viewer.widget())
        
        # Initialize NDV sync after viewer is created (if stack is loaded and HCS mode)
        if self.is_hcs_dataset and self.stack_loaded and self.ndv_sync is None:
            self._init_ndv_sync()
    
    def _ensure_composite_mode_and_luts(self, luts):
        """Ensure the viewer is in composite mode and has the correct LUTs applied"""
        try:
            if hasattr(self.ndv_viewer, 'viewer'):
                viewer = self.ndv_viewer.viewer
                if hasattr(viewer, 'mode'):
                    viewer.mode = 'composite'
                elif hasattr(viewer, 'composite_mode'):
                    viewer.composite_mode = True
                if hasattr(viewer, 'layers') and luts:
                    for i, layer in enumerate(viewer.layers):
                        if i in luts:
                            if hasattr(layer, 'colormap'):
                                layer.colormap = luts[i]
                            elif hasattr(layer, 'cmap'):
                                layer.cmap = luts[i]
                            elif hasattr(layer, 'lut'):
                                layer.lut = luts[i]
        except Exception as e:
            print(f"Error ensuring composite mode and LUTs: {e}")

    def create_all_fovs_zarr_array(self) -> Optional[xr.DataArray]:
        """
        Create a xarray DataArray for ALL FOVs.
        Builds a single unified dask graph - NO files opened during construction.
        
        Returns xarray with dimensions: (time, fov, z_level, channel, y, x)
        """
        if not ZARR_AVAILABLE:
            print("Zarr/tifffile/xarray not available")
            return None
        
        try:
            from bioio import BioImage
            base_path = Path(self.base_path)
            format_type = detect_acquisition_format(base_path)
            
            # Discover all available FOVs (just filenames - no opening!)
            all_fovs = self._discover_all_fovs()
            
            if not all_fovs:
                print("No FOVs found in dataset")
                return None
            
            
            if format_type == 'ome_tiff':
                return self._create_unified_ome_lazy_array(all_fovs)
            else:
                return self._create_unified_single_tiff_lazy_array(all_fovs)
            
        except Exception as e:
            print(f"Error creating multi-FOV array: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_unified_ome_lazy_array(self, all_fovs: List[Dict]) -> Optional[xr.DataArray]:
        """
        Create unified lazy array for OME-TIFF - opens ONE file to get dimensions,
        then builds lazy graph for all FOVs.
        """
        try:
            from bioio import BioImage
            base_path = Path(self.base_path)
            
            # Build file index: {(region, fov): filepath}
            ome_dir = base_path / "ome_tiff" if (base_path / "ome_tiff").exists() else base_path / "0"
            file_index = {}
            for fov_info in all_fovs:
                region = fov_info['region']
                fov = fov_info['fov']
                # Find the file
                for ome_path in ome_dir.glob("*.ome.tif*"):
                    if m := fpattern_ome.search(ome_path.name):
                        if m.group("r") == region and int(m.group("f")) == fov:
                            file_index[(region, fov)] = str(ome_path)
                            break
            
            if not file_index:
                return None
            
            # Extract metadata using tifffile only (no BioImage during graph construction)
            first_file = next(iter(file_index.values()))
            with tf.TiffFile(first_file) as tif:
                series = tif.series[0]
                axes = series.axes
                shape = series.shape
                shape_dict = dict(zip(axes, shape))
                
                n_t = shape_dict.get('T', 1)
                n_c = shape_dict.get('C', 1) 
                n_z = shape_dict.get('Z', 1)
                height = shape_dict.get('Y', shape[-2])
                width = shape_dict.get('X', shape[-1])
                
                # Extract channel names from OME-XML
                channel_names = []
                try:
                    if tif.ome_metadata:
                        import xml.etree.ElementTree as ET
                        root = ET.fromstring(tif.ome_metadata)
                        ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                        for ch in root.findall('.//ome:Channel', ns):
                            name = ch.get('Name') or ch.get('ID', '')
                            if name:
                                channel_names.append(name)
                except:
                    pass
            
            # Build LUTs
            luts = {}
            for i in range(n_c):
                name = channel_names[i] if i < len(channel_names) else f'Channel_{i}'
                luts[i] = self._get_channel_colormap_from_name(name, i)
            
            n_fov = len(all_fovs)
            
            # Memory protection: Estimate graph size and limit if needed
            total_chunks = n_t * n_fov
            chunk_size_mb = (n_z * n_c * height * width * 2) / (1024**2)
            estimated_graph_mb = total_chunks * 0.001  # ~1KB metadata per chunk
            
            print(f"Dataset: {n_t}T × {n_fov}FOV × {n_z}Z × {n_c}C")
            print(f"Chunks: {total_chunks} ({chunk_size_mb:.1f} MB each)")
            print(f"Graph overhead: ~{estimated_graph_mb:.1f} MB")
            
            # If graph would be huge, limit to first timepoint only
            if total_chunks > 10000:  # >10k chunks = risky
                print(f"WARNING: {total_chunks} chunks detected - limiting to first timepoint")
                n_t = 1
            
            # Efficient chunking: Load entire FOV volumes (Z×C×Y×X) as single chunks
            # Reduces graph from T×FOV×Z×C chunks to just T×FOV chunks (1000x reduction!)
            def load_fov_volume(fov_idx, t_idx):
                """Load entire FOV volume (all Z-slices and channels) in one operation"""
                def _load():
                    bio = None
                    try:
                        region, fov = all_fovs[fov_idx]['region'], all_fovs[fov_idx]['fov']
                        filepath = file_index.get((region, fov))
                        if not filepath:
                            return np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                        
                        bio = BioImage(filepath)
                        # Load entire volume at once: (Z, C, Y, X)
                        volume = np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                        for z_idx in range(n_z):
                            for c_idx in range(n_c):
                                volume[z_idx, c_idx] = bio.get_image_data("YX", T=t_idx, C=c_idx, Z=z_idx)
                        del bio
                        return volume
                    except:
                        if bio:
                            del bio
                        return np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                return _load
            
            # Build graph with FOV-level chunks (much smaller graph)
            time_arrays = []
            for t_idx in range(n_t):
                fov_arrays = []
                for fov_idx in range(n_fov):
                    delayed_load = delayed(load_fov_volume(fov_idx, t_idx))
                    # Single chunk per FOV: (Z, C, Y, X)
                    dask_chunk = da.from_delayed(delayed_load(), shape=(n_z, n_c, height, width), dtype=np.uint16)
                    fov_arrays.append(dask_chunk)
                time_arrays.append(da.stack(fov_arrays, axis=0))
            
            # Stack: (T, FOV, Z, C, Y, X)
            full_array = da.stack(time_arrays, axis=0)
            
            # Create xarray
            xarr = xr.DataArray(
                full_array,
                dims=['time', 'fov', 'z_level', 'channel', 'y', 'x'],
                coords={
                    'time': list(range(n_t)),
                    'fov': list(range(n_fov)),
                    'z_level': list(range(n_z)),
                    'channel': list(range(n_c))
                }
            )
            
            xarr.attrs['luts'] = luts
            return xarr
            
        except Exception as e:
            print(f"Error in unified OME loader: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_unified_single_tiff_lazy_array(self, all_fovs: List[Dict]) -> Optional[xr.DataArray]:
        """
        Create unified lazy array for Single-TIFF - scans files ONCE to build index,
        then creates lazy graph.
        """
        try:
            base_path = Path(self.base_path)
            
            # Build complete file index in ONE scan
            file_index = {}  # {(t, region, fov, z, channel): filepath}
            timepoint_set = set()
            z_set = set()
            channel_set = set()
            
            for tp_dir in sorted(base_path.iterdir()):
                if not (tp_dir.is_dir() and tp_dir.name.isdigit()):
                    continue
                t = int(tp_dir.name)
                timepoint_set.add(t)
                
                for tiff_path in tp_dir.glob("*.tiff"):
                    if m := fpattern.search(tiff_path.name):
                        region = m.group("r")
                        fov = int(m.group("f"))
                        z = int(m.group("z"))
                        channel = m.group("c")
                        
                        z_set.add(z)
                        channel_set.add(channel)
                        file_index[(t, region, fov, z, channel)] = str(tiff_path)
            
            # Organize dimensions
            timepoints = sorted(timepoint_set)
            z_levels = sorted(z_set)
            channels = sorted(channel_set)
            
            n_t = len(timepoints)
            n_fov = len(all_fovs)
            n_z = len(z_levels)
            n_c = len(channels)
            
            # Get dimensions from one sample file
            sample_file = next(iter(file_index.values()))
            sample_tiff = tf.TiffFile(sample_file)
            height, width = sample_tiff.pages[0].shape[-2:]
            sample_tiff.close()
            
            # Efficient chunking: Load entire FOV volumes (Z×C×Y×X) as single chunks
            def load_fov_volume_tiff(fov_idx, t_idx):
                """Load entire FOV volume (all Z-slices and channels) in one operation"""
                def _load():
                    try:
                        t = timepoints[t_idx]
                        region, fov = all_fovs[fov_idx]['region'], all_fovs[fov_idx]['fov']
                        
                        # Load entire volume: (Z, C, Y, X)
                        volume = np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                        for z_idx in range(n_z):
                            for c_idx in range(n_c):
                                z = z_levels[z_idx]
                                channel = channels[c_idx]
                                filepath = file_index.get((t, region, fov, z, channel))
                                if filepath:
                                    with tf.TiffFile(filepath) as tiff:
                                        volume[z_idx, c_idx] = tiff.pages[0].asarray()
                        return volume
                    except:
                        return np.zeros((n_z, n_c, height, width), dtype=np.uint16)
                return _load
            
            # Build graph with FOV-level chunks
            time_arrays = []
            for t_idx in range(n_t):
                fov_arrays = []
                for fov_idx in range(n_fov):
                    delayed_load = delayed(load_fov_volume_tiff(fov_idx, t_idx))
                    dask_chunk = da.from_delayed(delayed_load(), shape=(n_z, n_c, height, width), dtype=np.uint16)
                    fov_arrays.append(dask_chunk)
                time_arrays.append(da.stack(fov_arrays, axis=0))
            
            full_array = da.stack(time_arrays, axis=0)
            
            # Create xarray
            xarr = xr.DataArray(
                full_array,
                dims=['time', 'fov', 'z_level', 'channel', 'y', 'x'],
                coords={
                    'time': timepoints,
                    'fov': list(range(n_fov)),
                    'z_level': z_levels,
                    'channel': channels
                }
            )
            
            return xarr
            
        except Exception as e:
            print(f"Error in unified single-TIFF loader: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _discover_all_fovs(self) -> List[Dict]:
        """
        Discover all unique FOVs in the dataset.
        
        Returns:
            List of dicts: [{'fov': 0, 'region': 'A1'}, {'fov': 1, 'region': 'A1'}, ...]
        """
        base_path = Path(self.base_path)
        format_type = detect_acquisition_format(base_path)
        fov_set = set()
        
        # Look in first timepoint directory
        first_tp = next((d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()), None)
        if not first_tp:
            return []
        
        if format_type == 'ome_tiff':
            # Scan OME-TIFF directory
            ome_dir = base_path / "ome_tiff" if (base_path / "ome_tiff").exists() else first_tp
            for ome_file in ome_dir.glob("*.ome.tif*"):
                if m := fpattern_ome.search(ome_file.name):
                    region = m.group("r")
                    fov = int(m.group("f"))
                    fov_set.add((region, fov))
        else:
            # Scan single-TIFF files
            for tiff_file in first_tp.glob("*.tiff"):
                if m := fpattern.search(tiff_file.name):
                    region = m.group("r")
                    fov = int(m.group("f"))
                    fov_set.add((region, fov))
        
        # Convert to sorted list of dicts
        fov_list = [{'region': region, 'fov': fov} for region, fov in sorted(fov_set)]
        return fov_list
    
    def create_fov_zarr_array(self, target_fov: int, target_region: str = None) -> Optional[xr.DataArray]:
        """Create a lazy-loading xarray DataArray using O(1) file lookup (for single FOV, HCS mode)"""
        if not ZARR_AVAILABLE:
            print("Zarr/tifffile/xarray not available")
            return None
        
        # Detect format and route to appropriate loader
        format_type = detect_acquisition_format(Path(self.base_path))
        if format_type == 'ome_tiff':
            return self._create_fov_from_ome(target_fov, target_region)
        else:
            return self._create_fov_from_single_tiff(target_fov, target_region)
    
    def _create_fov_from_single_tiff(self, target_fov: int, target_region: str = None) -> Optional[xr.DataArray]:
        """Load FOV from single-TIFF files (current format)"""
        try:
            # Scan for files from ALL timepoints for this FOV
            # Note: fov_to_files only contains current timepoint, so we need to scan all timepoints
            base_path = Path(self.base_path)
            fov_files = []
            
            # Look through all timepoint directories
            for timepoint_dir in base_path.iterdir():
                if timepoint_dir.is_dir() and timepoint_dir.name.isdigit():
                    # Scan this timepoint for matching FOV files
                    for tiff_path in timepoint_dir.glob("*.tiff"):
                        if m := fpattern.search(tiff_path.name):
                            fov = int(m.group("f"))
                            region = m.group("r")
                            if fov == target_fov and (target_region is None or region == target_region):
                                fov_files.append(str(tiff_path))
            
            if not fov_files:
                return None
            
            axes, shape, indices, sorted_files = parse_filenames(fov_files)
            sample_tiff = tf.TiffFile(sorted_files[0])
            sample_shape = sample_tiff.pages[0].shape
            height, width = sample_shape[-2:]
            sample_tiff.close()  # ✓ CRITICAL: Close after getting dimensions
            
            file_lookup = {}
            for idx, file_path in enumerate(sorted_files):
                coords = np.unravel_index(idx, shape)
                file_lookup[coords] = file_path
            
            def load_single_file(file_path, coords=None):
                def _load():
                    tiff = None
                    try:
                        tiff = tf.TiffFile(file_path)
                        data = np.array(tiff.pages[0].asarray())
                        tiff.close()  # ✓ CRITICAL: Close file immediately
                        return data
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                        if tiff:
                            tiff.close()
                        return np.zeros((height, width), dtype=np.uint16)
                return _load
            
            # Build dask array structure
            if len(axes) == 5:
                time_arrays = []
                for t in range(shape[0]):
                    region_arrays = []
                    for r in range(shape[1]):
                        fov_arrays = []
                        for f in range(shape[2]):
                            z_arrays = []
                            for z in range(shape[3]):
                                channel_arrays = []
                                for c in range(shape[4]):
                                    coord_key = (t, r, f, z, c)
                                    if coord_key in file_lookup:
                                        delayed_load = delayed(load_single_file(file_lookup[coord_key], coord_key))
                                        dask_chunk = da.from_delayed(delayed_load(), shape=(height, width), dtype=np.uint16)
                                        channel_arrays.append(dask_chunk)
                                    else:
                                        channel_arrays.append(da.zeros((height, width), dtype=np.uint16))
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
                    full_array = da.zeros(shape + (height, width), dtype=np.uint16)
                
                # Remove singleton dimensions
                squeeze_axes = []
                final_axes = []
                for i, (axis, size) in enumerate(zip(axes, shape)):
                    if size == 1 and axis not in ['time', 'z_level']:
                        squeeze_axes.append(i)
                    else:
                        final_axes.append(axis)
                if squeeze_axes:
                    full_array = da.squeeze(full_array, axis=tuple(squeeze_axes))
                final_axes.extend(['y', 'x'])
                
                # Create coordinates
                coords_dict = {}
                for i, axis in enumerate(final_axes):
                    if axis in ['time', 'z_level']:
                        coords_dict[axis] = list(range(full_array.shape[i]))
                    elif axis == 'channel':
                        coords_dict[axis] = [f"ch_{j}" for j in range(full_array.shape[i])]
                    elif axis in ['y', 'x']:
                        size = height if axis == 'y' else width
                        coords_dict[axis] = list(range(size))
                    else:
                        coords_dict[axis] = list(range(full_array.shape[i]))
                
                xarray_data = xr.DataArray(full_array, dims=final_axes, coords=coords_dict, name="image_data")
                return xarray_data
        except Exception as e:
            print(f"Error creating lazy FOV xarray: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_fov_from_ome(self, target_fov: int, target_region: str = None) -> Optional[xr.DataArray]:
        """Load FOV from OME-TIFF files using bioio for lazy loading"""
        try:
            from bioio import BioImage
            base_path = Path(self.base_path)
            
            # OME-TIFF files contain all timepoints internally
            # Check for OME-TIFF in ome_tiff/ directory or fallback to 0/
            first_tp = base_path / "ome_tiff" if (base_path / "ome_tiff").exists() else base_path / "0"
            ome_file = None
            
            for ome_path in first_tp.glob("*.ome.tif*"):
                if m := fpattern_ome.search(ome_path.name):
                    region, fov = m.group("r"), int(m.group("f"))
                    if fov == target_fov and (target_region is None or region == target_region):
                        ome_file = str(ome_path)
                        break
            
            if not ome_file:
                return None
            
            # Load with bioio - handles all dimension complexity
            bio_img = BioImage(ome_file)
            
            # Get dimensions
            n_channels = bio_img.dims.C
            n_z = bio_img.dims.Z
            n_timepoints = bio_img.dims.T
            height = bio_img.dims.Y
            width = bio_img.dims.X
            
            # Extract channel names from bioio metadata
            channel_names = []
            try:
                # bioio provides channel names through physical_pixel_sizes or ome metadata
                if hasattr(bio_img, 'channel_names') and bio_img.channel_names:
                    channel_names = list(bio_img.channel_names)
            except:
                pass
            
            # Build colormap LUTs from actual channel names
            luts = {}
            
            for i in range(n_channels):
                name = channel_names[i] if i < len(channel_names) else f'Channel_{i}'
                colormap = self._get_channel_colormap_from_name(name, i)
                luts[i] = colormap
            
            bio_img.close()  # ✓ CRITICAL: Close after getting metadata
            
            # Build lazy dask array structure using bioio: (time, z_level, channel, y, x)
            def load_ome_slice_bioio(filepath, c_idx, z_idx, t_idx):
                def _load():
                    bio = None
                    try:
                        bio = BioImage(filepath)
                        # bioio handles ALL dimension ordering automatically!
                        data = bio.get_image_data("YX", T=t_idx, C=c_idx, Z=z_idx)
                        del bio  # Release reference
                        return data
                    except Exception as e:
                        print(f"Error loading OME slice: {e}")
                        return np.zeros((height, width), dtype=np.uint16)
                return _load
            
            time_arrays = []
            for t in range(n_timepoints):
                z_arrays = []
                for z in range(n_z):
                    channel_arrays = []
                    for c in range(n_channels):
                        delayed_load = delayed(load_ome_slice_bioio(ome_file, c, z, t))
                        dask_chunk = da.from_delayed(delayed_load(), shape=(height, width), dtype=np.uint16)
                        channel_arrays.append(dask_chunk)
                    z_arrays.append(da.stack(channel_arrays, axis=0))
                time_arrays.append(da.stack(z_arrays, axis=0))
            
            full_array = da.stack(time_arrays, axis=0)
            
            # Create xarray with proper dimensions
            xarr = xr.DataArray(
                full_array,
                dims=['time', 'z_level', 'channel', 'y', 'x'],
                coords={
                    'time': list(range(n_timepoints)),
                    'z_level': list(range(n_z)),
                    'channel': list(range(n_channels))
                }
            )
            
            # Store LUTs as attribute for set_ndv_data to use
            xarr.attrs['luts'] = luts
            
            return xarr
            
        except Exception as e:
            print(f"Error creating OME FOV xarray: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_stack_async(self):
        """Build stack in background thread (non-blocking, HCS only)"""
        if not self.is_hcs_dataset or self.plate_stack is None:
            return
        
        # Create builder thread
        self.builder_thread = StackBuilderThread(
            self.base_path,
            self.cache_dir,
            self.downsample_factor
        )
        
        # Connect signals
        self.builder_thread.progress.connect(self._on_build_progress)
        self.builder_thread.finished.connect(self._on_build_finished)
        
        # Start building
        self.builder_thread.start()
        
        # Update status
        if hasattr(self, 'status_label'):
            self.status_label.setText("Building Z×T stack in background...")
    
    def _on_build_progress(self, message: str):
        """Update status label with build progress"""
        if hasattr(self, 'status_label'):
            self.status_label.setText(f"Building stack: {message}")
    
    def _on_build_finished(self, success: bool, message: str):
        """Handle completion of stack building"""
        if success:
            self._load_stack()
        else:
            print(f"Error: Stack build failed - {message}")
            if hasattr(self, 'status_label'):
                self.status_label.setText("Z×T stack build failed - single plane mode")
    
    def _load_stack(self):
        """Load the pre-built Z×T stack"""
        if self.plate_stack.load_stack(self.downsample_factor):
            self.stack_loaded = True
            
            # Extract colormaps from stack metadata
            metadata = self.plate_stack.get_metadata()
            if metadata and 'shape_info' in metadata and metadata['shape_info']:
                self.wavelengths = metadata['shape_info']['wavelengths']
                self.colormaps = metadata['shape_info']['colormaps']
            
            # Update status
            if hasattr(self, 'status_label'):
                if metadata:
                    n_t = len(metadata['timepoints'])
                    n_z = len(metadata['z_levels'])
                    self.status_label.setText(
                        f"Z×T stack ready: {n_t} timepoints × {n_z} z-levels. "
                        "Click tile, use sliders to navigate."
                    )
        else:
            print("Warning: Could not load Z×T stack")
            self.stack_label = False
    
    def _init_ndv_sync(self):
        """Initialize NDV synchronization controllers"""
        if not NDV_AVAILABLE or not hasattr(self, 'ndv_viewer'):
            return
        
        try:
            # Initialize plate contrast limits from NDV's initial state
            self._sync_initial_contrast_from_ndv()
            
            # Z/T slider sync
            self.ndv_sync = NDVSyncController(self.ndv_viewer, self)
            self.ndv_sync.indices_changed.connect(self._on_ndv_indices_changed)
            self.ndv_sync.connect_to_viewer()
            
            # Contrast sync
            self.ndv_contrast_sync = NDVContrastSyncController(self.ndv_viewer, self)
            self.ndv_contrast_sync.contrast_changed.connect(self._on_ndv_contrast_changed)
            self.ndv_contrast_sync.connect_to_viewer()
            
        except Exception as e:
            print(f"Warning: Could not initialize NDV sync: {e}")
            import traceback
            traceback.print_exc()
    
    def _sync_initial_contrast_from_ndv(self):
        """Read NDV's initial contrast limits and apply to plate image"""
        from PyQt5.QtCore import QTimer
        
        def _delayed_sync():
            """Sync after a delay to ensure all channel data is loaded"""
            try:
                if not hasattr(self.ndv_viewer, 'display_model'):
                    return
                
                dm = self.ndv_viewer.display_model
                if not hasattr(dm, 'luts'):
                    return
                
                luts = dm.luts
                synced_count = 0
                
                for ch_idx, lut_model in luts.items():
                    if hasattr(lut_model, 'clims'):
                        clims = lut_model.clims
                        # Get actual contrast values - use computed() for auto, or min/max for manual
                        try:
                            if hasattr(clims, 'computed'):
                                vmin, vmax = clims.computed()
                            elif hasattr(clims, 'min') and hasattr(clims, 'max'):
                                vmin, vmax = float(clims.min), float(clims.max)
                            else:
                                continue
                            
                            self._plate_contrast_limits[ch_idx] = (float(vmin), float(vmax))
                            synced_count += 1
                        except:
                            pass
                
                # Update plate display with NDV's initial contrast
                if synced_count > 0:
                    self.update_image()
                    
            except Exception as e:
                print(f"Warning: Could not sync initial contrast from NDV: {e}")
        
        # Delay to let NDV compute contrast for all channels
        QTimer.singleShot(500, _delayed_sync)
    
    def _on_ndv_indices_changed(self, t_idx: int, z_idx: int):
        """
        Callback when NDV z/t sliders change.
        Update the left-side plate image to match.
        """
        if not self.stack_loaded:
            return
        
        # Get corresponding plate page from stack
        plate_page = self.plate_stack.get_page(t_idx, z_idx)
        
        if plate_page is not None:
            # Update the displayed multichannel image
            self.multichannel_image = plate_page
            self.update_image()
            
            # Update status
            metadata = self.plate_stack.get_metadata()
            if metadata:
                actual_t = metadata['timepoints'][t_idx]
                actual_z = metadata['z_levels'][z_idx]
                self.status_label.setText(
                    f"Viewing: T={actual_t}, Z={actual_z}"
                )
    
    def _on_ndv_contrast_changed(self, channel_idx: int, vmin: float, vmax: float):
        """
        Callback when NDV contrast sliders change.
        Update plate view contrast for this channel.
        """
        if vmin == -1.0 and vmax == -1.0:
            if channel_idx in self._plate_contrast_limits:
                del self._plate_contrast_limits[channel_idx]
        else:
            self._plate_contrast_limits[channel_idx] = (vmin, vmax)
        
        # Trigger plate image refresh
        self.update_image()


class ViewerMainWindow(QMainWindow):
    """Main application window for viewer"""
    
    def __init__(self, base_path: str, timepoint: int = 0, downsample_factor: float = 0.85):
        super().__init__()
        self.setWindowTitle("Multi-Channel TIFF Viewer with Embedded NDV")
        
        # Detect dataset type to set appropriate window size
        from .common import detect_hcs_vs_normal_tissue
        is_hcs = detect_hcs_vs_normal_tissue(Path(base_path))
        
        if is_hcs:
            # HCS: Wide window for split-panel (plate + NDV)
            self.setGeometry(100, 100, 1400, 800)
        else:
            # Normal tissue: Narrower window (NDV only)
            self.setGeometry(100, 100, 600, 800)
        
        # Apply dark theme
        from PyQt5.QtWidgets import QStyleFactory
        self.setStyle(QStyleFactory.create("Fusion"))
        self._set_dark_palette()
        
        self.tiff_viewer = TiffViewerWidget(base_path, timepoint, downsample_factor)
        self.setCentralWidget(self.tiff_viewer)
    
    def _set_dark_palette(self):
        """Set dark color palette"""
        from PyQt5.QtGui import QPalette, QColor
        dark_palette = self.palette()
        dark_palette.setColor(QPalette.Window, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Base, QColor(35, 35, 35))
        dark_palette.setColor(QPalette.AlternateBase, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ToolTipBase, QColor(25, 25, 25))
        dark_palette.setColor(QPalette.ToolTipText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Text, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.Button, QColor(53, 53, 53))
        dark_palette.setColor(QPalette.ButtonText, QColor(255, 255, 255))
        dark_palette.setColor(QPalette.BrightText, QColor(255, 0, 0))
        dark_palette.setColor(QPalette.Link, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.Highlight, QColor(42, 130, 218))
        dark_palette.setColor(QPalette.HighlightedText, QColor(35, 35, 35))
        self.setPalette(dark_palette)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Click on tiles to view in embedded NDV viewer")

