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

from .common import COLOR_WEIGHTS, WELL_FORMATS, parse_filenames, fpattern

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
        self.cache_dir = Path(base_path) / "assembled_tiles_cache"
        
        # Load data from cache
        from .preprocessing import PlateAssembler
        assembler = PlateAssembler(base_path, timepoint)
        target_px = int(assembler._get_original_pixel_size() * downsample_factor)
        self.assembled_data, self.pixel_map = assembler.assemble_plate(target_px)
        
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
            self.ndv_viewer = ndv.ArrayViewer(dummy_data, channel_axis=0, channel_mode="composite")
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
        """Setup tile grid information for efficient coordinate lookup"""
        sample_tile = next(iter(self.pixel_map.values()))
        sample_img = sample_tile.image
        self.tile_h, self.tile_w = sample_img.shape[:2]
        
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
        pixmap = self.image_label.pixmap()
        if pixmap:
            label_size = self.image_label.size()
            pixmap_size = pixmap.size()
            
            offset_x = (label_size.width() - pixmap_size.width()) // 2
            offset_y = (label_size.height() - pixmap_size.height()) // 2
            
            adjusted_x = x - offset_x - 50  # Account for well plate labels
            adjusted_y = y - offset_y - 30
            
            if (0 <= adjusted_x < pixmap_size.width() - 50 and 
                0 <= adjusted_y < pixmap_size.height() - 30):
                img_x = int(adjusted_x * self.scale_x)
                img_y = int(adjusted_y * self.scale_y)
                
                grid_x = img_x // self.tile_w
                grid_y = img_y // self.tile_h
                
                if (grid_x, grid_y) in self.grid_to_tile:
                    tile = self.grid_to_tile[(grid_x, grid_y)]
                    fov_number = tile.fov
                    region_name = tile.region
                    self.status_label.setText(f"Creating zarr array for FOV {fov_number} in region {region_name}")
                    
                    self.clicked_position = (adjusted_x + 50, adjusted_y + 30)
                    
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
                    
                    self.update_image()
                else:
                    self.status_label.setText("No tile found at this location")
            else:
                self.status_label.setText("Click outside image area")
        else:
            self.status_label.setText("No image loaded")
    
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
        regions = set(tile.region for tile in self.pixel_map.values())
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
        """Add well plate row and column labels to the pixmap"""
        rows, cols = self._calculate_well_format()
        label_height, label_width = 30, 50
        total_width = pixmap.width() + label_width
        total_height = pixmap.height() + label_height
        
        labeled_pixmap = QPixmap(total_width, total_height)
        labeled_pixmap.fill(Qt.transparent)
        
        painter = QPainter(labeled_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.drawPixmap(label_width, label_height, pixmap)
        
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
                        if hasattr(data, 'chunks'):
                            self._recreate_viewer_with_lazy_support(data, channel_axis, luts, current_state)
                        else:
                            self._recreate_viewer_with_state(data, channel_axis, luts, current_state)
                except Exception as e:
                    print(f"Direct data update failed, recreating viewer: {e}")
                    if hasattr(data, 'chunks'):
                        self._recreate_viewer_with_lazy_support(data, channel_axis, luts, current_state)
                
                self._restore_viewer_state(current_state)
                self._ensure_composite_mode_and_luts(luts)
                
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
        
        if channel_axis is not None:
            self.ndv_viewer = ndv.ArrayViewer(data, channel_axis=channel_axis, 
                                             channel_mode="composite", luts=luts)
        
        splitter = self.parent().findChild(QSplitter)
        if splitter:
            splitter.addWidget(self.ndv_viewer.widget())
            splitter.setSizes([600, 400])
    
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
        """Create a lazy-loading xarray DataArray that maintains full dimensions"""
        if not ZARR_AVAILABLE:
            print("Zarr/tifffile/xarray not available")
            return None
        
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
            
            # Get all TIFF files
            base_path = Path(self.base_path)
            all_tiffs = list(base_path.rglob("*.tiff"))
            if not all_tiffs:
                return None
            
            # Filter files for target FOV and region
            fov_files = []
            for tiff_path in all_tiffs:
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

