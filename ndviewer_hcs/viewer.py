"""Viewer widget with NDV integration for viewing assembled plates"""

import pickle
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from skimage import io
from PyQt5.QtWidgets import (QWidget, QVBoxLayout, QLabel, QSplitter, 
                             QSpacerItem, QSizePolicy, QMainWindow, QStatusBar)
from PyQt5.QtGui import QPixmap, QImage, QPainter, QPen
from PyQt5.QtCore import Qt, pyqtSignal

from .common import COLOR_WEIGHTS, parse_filenames, fpattern, fpattern_ome, detect_acquisition_format
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
        
        # Initialize plate stack manager
        self.plate_stack = PlateStackManager(base_path, self.cache_dir)
        self.ndv_sync = None  # Will be initialized after NDV viewer is created
        self.ndv_contrast_sync = None  # Contrast sync controller
        self.stack_loaded = False
        self._plate_contrast_limits = {}  # {channel_idx: (vmin, vmax)}
        
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
        self.clicked_position = None
        
        self._setup_tile_grid()
        self.setup_ui()
        self.update_image()
        
        # Check if Z×T stack exists, if not build it silently in background
        if not self.plate_stack.exists(downsample_factor):
            print("Z×T stack not found, building in background...")
            self._build_stack_async()
        else:
            self._load_stack()

    def setup_ui(self):
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
        """Handle click on plate image - find FOV and center red dot"""
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
        for i, wavelength in enumerate(self.wavelengths):
            if '405' in wavelength: luts[i] = 'blue'
            elif '488' in wavelength: luts[i] = 'green'
            elif '561' in wavelength: luts[i] = 'yellow'
            elif '638' in wavelength or '640' in wavelength: luts[i] = 'red'
            else: luts[i] = 'gray'
        return luts

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
                
                print(f"Setting data with shape: {data.shape}, channel_axis: {channel_axis}, LUTs: {luts}")
                
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
            print(f"Data dimensions: {data.dims}, visible_axes: {visible_axes}")
            print(f"Dimensions that will have sliders: {slider_dims}")
        
        if channel_axis is not None:
            self.ndv_viewer = ndv.ArrayViewer(data, channel_axis=channel_axis, 
                                             channel_mode="composite", luts=luts,
                                             visible_axes=visible_axes)
        
        splitter = self.parent().findChild(QSplitter)
        if splitter:
            splitter.addWidget(self.ndv_viewer.widget())
            splitter.setSizes([600, 400])
        
        # Initialize NDV sync after viewer is created (if stack is loaded)
        if self.stack_loaded and self.ndv_sync is None:
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

    def create_fov_zarr_array(self, target_fov: int, target_region: str = None) -> Optional[xr.DataArray]:
        """Create a lazy-loading xarray DataArray using O(1) file lookup"""
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
            # # Load flatfields if available - DISABLED (Less is more)
            # flatfield_data = None
            # wavelength_mapping = {}
            # flatfields_file = self.cache_dir / "flatfields_global.pkl"
            # if flatfields_file.exists():
            #     try:
            #         with open(flatfields_file, 'rb') as f:
            #             flatfield_data = pickle.load(f)
            #         for i, wavelength_name in enumerate(self.wavelengths):
            #             match = re.search(r'(\d{3})', wavelength_name)
            #             if match:
            #                 wavelength_mapping[i] = match.group(1)
            #     except Exception as e:
            #         print(f"Error loading flatfields: {e}")
            #         flatfield_data = None
            
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
                print(f"No files found for FOV {target_fov} in region {target_region}")
                return None
            
            print(f"Found {len(fov_files)} files for FOV {target_fov} in region {target_region}")
            axes, shape, indices, sorted_files = parse_filenames(fov_files)
            print(f"Parsed dimensions - axes: {axes}, shape: {shape}")
            sample_tiff = tf.TiffFile(sorted_files[0])
            sample_shape = sample_tiff.pages[0].shape
            height, width = sample_shape[-2:]
            
            file_lookup = {}
            for idx, file_path in enumerate(sorted_files):
                coords = np.unravel_index(idx, shape)
                file_lookup[coords] = file_path
            
            def load_single_file(file_path, coords=None):
                def _load():
                    try:
                        tiff = tf.TiffFile(file_path)
                        data = np.array(tiff.pages[0].asarray())
                        
                        # # Apply flatfield correction - DISABLED (Less is more)
                        # if flatfield_data and coords is not None and len(axes) >= 5:
                        #     channel_idx = coords[4] if 'channel' in axes else None
                        #     if channel_idx is not None and channel_idx in wavelength_mapping:
                        #         wavelength_str = wavelength_mapping[channel_idx]
                        #         if wavelength_str in flatfield_data:
                        #             flatfield = flatfield_data[wavelength_str]
                        #             data = data.astype(np.float32) / flatfield.astype(np.float32)
                        #             data = np.clip(data, 0, np.iinfo(np.uint16).max).astype(np.uint16)
                        return data
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
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
                print(f"Created lazy xarray DataArray with shape: {xarray_data.shape}")
                return xarray_data
        except Exception as e:
            print(f"Error creating lazy FOV xarray: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _create_fov_from_ome(self, target_fov: int, target_region: str = None) -> Optional[xr.DataArray]:
        """Load FOV from OME-TIFF files (new format)"""
        try:
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
                print(f"No OME-TIFF file found for FOV {target_fov} in region {target_region}")
                return None
            
            # Read metadata from the OME file
            with tf.TiffFile(ome_file) as tif:
                sample_data = tif.asarray()
                
                # Parse OME metadata to understand dimension order
                dim_order = 'XYCZT'  # Default
                size_c, size_z, size_t = 1, 1, 1
                
                if tif.is_ome and tif.ome_metadata:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(tif.ome_metadata)
                    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels = root.find('.//ome:Pixels', ns)
                    if pixels:
                        dim_order = pixels.get('DimensionOrder', 'XYCZT')
                        size_c = int(pixels.get('SizeC', 1))
                        size_z = int(pixels.get('SizeZ', 1))
                        size_t = int(pixels.get('SizeT', 1))
                
                # Tifffile returns array based on dimension order
                # For XYCZT order with shape (T, Z, C, Y, X), we need to transpose to (C, Z, T, Y, X)
                if dim_order == 'XYCZT' and len(sample_data.shape) == 5:
                    # Shape is (T, Z, C, Y, X), transpose to (C, Z, T, Y, X)
                    sample_data = np.transpose(sample_data, (2, 1, 0, 3, 4))
                elif len(sample_data.shape) == 3:
                    # (Z, Y, X) - single channel, single T
                    sample_data = sample_data[np.newaxis, :, np.newaxis, :, :]
                elif len(sample_data.shape) == 4:
                    # Could be (C, Z, Y, X) or (Z, C, Y, X) - add T dimension
                    sample_data = sample_data[:, :, np.newaxis, :, :]
                
                n_channels, n_z, n_timepoints, height, width = sample_data.shape
                
                # Extract channel names from OME metadata
                channel_names = []
                if tif.is_ome and tif.ome_metadata:
                    import xml.etree.ElementTree as ET
                    root = ET.fromstring(tif.ome_metadata)
                    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
                    pixels = root.find('.//ome:Pixels', ns)
                    if pixels:
                        for ch in pixels.findall('ome:Channel', ns):
                            channel_names.append(ch.get('Name', ''))
            
            print(f"Found OME-TIFF for FOV {target_fov}: C={n_channels}, Z={n_z}, T={n_timepoints}")
            
            # Build colormap LUTs from channel names
            from .common import COLOR_MAPS
            luts = {}
            default_colors = ['blue', 'green', 'yellow', 'red', 'darkred']  # Ch0-4, then gray
            
            for i in range(n_channels):
                name = channel_names[i] if i < len(channel_names) else f'Channel_{i}'
                colormap = None
                
                # Try to extract wavelength from name
                for wl_key, color in COLOR_MAPS.items():
                    if wl_key in name:
                        colormap = color
                        break
                
                # If no wavelength found, use default color palette
                if colormap is None:
                    if i < len(default_colors):
                        colormap = default_colors[i]
                    else:
                        colormap = 'gray'  # Ch5+ get gray
                
                luts[i] = colormap
            
            # Build lazy dask array structure: (time, z_level, channel, y, x)
            def load_ome_slice(filepath, c_idx, z_idx, t_idx, dim_order):
                def _load():
                    try:
                        with tf.TiffFile(filepath) as tif:
                            data = tif.asarray()
                            
                            # Transpose based on dimension order
                            if dim_order == 'XYCZT' and len(data.shape) == 5:
                                # (T, Z, C, Y, X) -> (C, Z, T, Y, X)
                                data = np.transpose(data, (2, 1, 0, 3, 4))
                            elif len(data.shape) == 3:
                                # (Z, Y, X) - single channel
                                data = data[np.newaxis, :, np.newaxis, :, :]
                            elif len(data.shape) == 4:
                                # (C, Z, Y, X) or similar - add T dim
                                data = data[:, :, np.newaxis, :, :]
                            
                            # Now data is (C, Z, T, Y, X)
                            return data[c_idx, z_idx, t_idx, :, :]
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
                        delayed_load = delayed(load_ome_slice(ome_file, c, z, t, dim_order))
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
            
            print(f"Created OME lazy xarray with shape: {xarr.shape}")
            
            # Store LUTs as attribute for set_ndv_data to use
            xarr.attrs['luts'] = luts
            
            return xarr
            
        except Exception as e:
            print(f"Error creating OME FOV xarray: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _build_stack_async(self):
        """Build stack in background thread (non-blocking)"""
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
                print(f"[Stack] Using {len(self.colormaps)} channel colormaps: {self.colormaps}")
            
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
                            print(f"[Init] Ch{ch_idx} contrast from NDV: [{vmin:.0f}, {vmax:.0f}]")
                            synced_count += 1
                        except Exception as e:
                            print(f"[Init] Ch{ch_idx} contrast failed: {e}")
                
                # Update plate display with NDV's initial contrast
                if synced_count > 0:
                    print(f"[Init] Synced {synced_count}/{len(luts)} channels, updating plate")
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
        self.setGeometry(100, 100, 1400, 800)
        
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

