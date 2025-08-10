#!/usr/bin/env python3
"""
Navigator Module - Refactored
Handles navigator overlay functionality for napari viewer.
Provides region-specific multi-channel navigator overlays with click handling.
"""

import os
import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Callable
from functools import partial
import re

# Try to import downsampler from utils directory
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.downsampler import DownsampledNavigator
    NAVIGATOR_AVAILABLE = True
    print("Successfully imported downsampler module from utils in navigator")
except ImportError as e:
    print(f"Warning: Navigator module not available: {e}")
    NAVIGATOR_AVAILABLE = False

# Import scaling manager
from scaling_manager import ScalingManager


@dataclass
class NavigatorConfig:
    """Configuration for navigator behavior"""
    tile_size: int = 75  # Will be overridden by scaling manager
    tile_cache_enabled: bool = True
    pixel_gap_to_viewer: int = 1
    box_edge_width: int = 9
    border_edge_width: int = 12
    contrast_tolerance: float = 0.05 #synch contrast if deviates by 5% from the fov_viewer
    contrast_drift_check_delay: int = 2000  #ms
    selection_verify_delay: int = 200  # ms


@dataclass
class NavigatorState:
    """Mutable state for navigator"""
    updating_navigator: bool = False
    updating_contrast: bool = False
    forcing_sync: bool = False
    last_dims_state: Optional[tuple] = None
    last_known_contrasts: Dict[str, List[float]] = field(default_factory=dict)
    contrast_check_timer: Any = None
    visibility_handlers: Dict[Tuple[int, int], Callable] = field(default_factory=dict)


class LayerManager:
    """Manages layer creation and manipulation"""
    
    def __init__(self, viewer, config: NavigatorConfig, scaling_manager: ScalingManager = None):
        self.viewer = viewer
        self.config = config
        self.scaling_manager = scaling_manager
        
    def create_navigator_layer(self, mosaic_array: np.ndarray, region_name: str, 
                             channel_name: str, channel_idx: int, region_idx: int,
                             nav_channel_idx: int, colormap: str, position: Tuple[float, float],
                             is_visible: bool) -> Any:
        """Create a single navigator layer"""
        # Get navigator scale from scaling manager if available
        navigator_scale = (1, 1)  # Default
        if self.scaling_manager:
            navigator_scale = self.scaling_manager.get_navigator_scale()
        
        # Calculate FOV-style contrast limits for Navigator layer
        nav_contrast_limits = self._get_fov_layer_contrast_limits(channel_idx, mosaic_array)
        
        layer = self.viewer.add_image(
            mosaic_array,
            name=f'_{region_name} - Fluorescence_{self._extract_wavelength(channel_name)}_nm_EX',
            opacity=1.0,
            colormap=colormap,
            blending='additive',
            visible=is_visible,
            scale=navigator_scale,
            translate=position,
            metadata={
                'region_idx': region_idx,
                'region_name': region_name,
                'channel_idx': channel_idx,
                'channel_name': channel_name,
                'navigator_channel_idx': nav_channel_idx
            },
            contrast_limits=nav_contrast_limits
        )
        
        # Set interactive property after creation
        layer.interactive = True
        return layer
    
    def create_border_layer(self, position: Tuple[float, float], dimensions: Tuple[int, int],
                          region_name: str, is_visible: bool) -> Any:
        """Create border layer around navigator"""
        height, width = dimensions
        y_pos, x_pos = position
        
        coords = [
            [y_pos, x_pos],
            [y_pos, x_pos + width],
            [y_pos + height, x_pos + width],
            [y_pos + height, x_pos]
        ]
        
        layer = self.viewer.add_shapes(
            [coords],
            shape_type='rectangle',
            edge_color='white',
            face_color='transparent',
            edge_width=self.config.border_edge_width,
            name=f'_Navigator Border {region_name} <hidden>',
            opacity=1.0,
            visible=is_visible
        )
        
        self._make_non_interactive(layer)
        return layer
    
    def create_box_layer(self, position: Tuple[float, float], size: float,
                        region_name: str) -> Any:
        """Create FOV position indicator box"""
        y, x = position
        half_size = size // 2
        
        coords = [
            [y - half_size, x - half_size],
            [y - half_size, x + half_size],
            [y + half_size, x + half_size],
            [y + half_size, x - half_size]
        ]
        
        layer = self.viewer.add_shapes(
            [coords],
            shape_type='rectangle',
            edge_color='red',
            face_color='transparent',
            edge_width=self.config.box_edge_width,
            name=f'_Navigator Box {region_name} <hidden>',
            opacity=1.0
        )
        
        self._make_non_interactive(layer)
        return layer
    
    def _make_non_interactive(self, layer):
        """Make a layer non-interactive"""
        layer.interactive = False
        layer.editable = False
        layer.mouse_pan = False
        layer.mouse_zoom = False
    
    def _extract_wavelength(self, channel_name: str) -> str:
        """Extract wavelength number from channel name"""
        match = re.search(r'(\d{3})', channel_name)
        return match.group(1) if match else "000"
    
    def _get_fov_layer_contrast_limits(self, channel_idx: int, mosaic_array: np.ndarray) -> List[float]:
        """Get contrast limits from existing FOV layer or calculate FOV-style limits"""
        try:
            # First, try to get contrast limits from existing FOV layer for same channel
            for layer in self.viewer.layers:
                if ('Channel:' in layer.name and 
                    hasattr(layer, 'metadata') and 
                    layer.metadata.get('channel_index') == channel_idx):
                    
                    fov_contrast = layer.contrast_limits
                    print(f"[NAVIGATOR] Using FOV layer contrast limits for channel {channel_idx}: {fov_contrast}")
                    return list(fov_contrast)
            
            # If no FOV layer found, calculate FOV-style contrast limits
            print(f"[NAVIGATOR] No FOV layer found for channel {channel_idx}, calculating FOV-style limits")
            return self._calculate_fov_style_contrast_limits(mosaic_array)
            
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to get FOV layer contrast limits: {e}")
            return self._calculate_fov_style_contrast_limits(mosaic_array)
    
    def _calculate_fov_style_contrast_limits(self, data: np.ndarray) -> List[float]:
        """Calculate contrast limits using the same method as napari's auto-calculation for FOV layers"""
        try:
            # Mimic napari's auto-contrast calculation
            # Napari typically uses 2nd and 98th percentiles for auto-contrast
            
            # For large arrays, sample to speed up calculation
            if data.size > 1000000:  # Sample if > 1M pixels
                # Take every 10th pixel in each dimension
                sample = data[::10, ::10]
            else:
                sample = data
            
            # Remove zeros and very low values that might be background
            non_zero_data = sample[sample > 0]
            if len(non_zero_data) == 0:
                # Fallback to full data if no non-zero values
                non_zero_data = sample
            
            # Calculate percentiles like napari does
            # Use 2nd and 98th percentiles to avoid outliers
            p2, p98 = np.percentile(non_zero_data, [2, 98])
            
            # Ensure valid range
            if p98 <= p2:
                p2 = float(sample.min())
                p98 = float(sample.max())
                # If still equal, add small offset
                if p98 <= p2:
                    p98 = p2 + 1.0
            
            contrast_limits = [float(p2), float(p98)]
            print(f"[NAVIGATOR] Calculated FOV-style contrast limits: {contrast_limits}")
            return contrast_limits
            
        except Exception as e:
            print(f"[NAVIGATOR ERROR] Failed to calculate FOV-style contrast limits: {e}")
            # Fallback to data range
            return [float(data.min()), float(data.max())]
    
    def remove_navigator_boxes(self):
        """Remove all navigator box layers"""
        to_remove = [layer for layer in self.viewer.layers 
                    if hasattr(layer, 'name') and '_Navigator Box' in layer.name]
        for layer in to_remove:
            try:
                self.viewer.layers.remove(layer)
            except:
                pass


class ContrastSynchronizer:
    """Handles contrast synchronization between layers"""
    
    def __init__(self, state: NavigatorState):
        self.state = state
        
    def sync_fov_to_navigator(self, fov_layer, nav_layer, channel_name: str):
        """Sync contrast from FOV to navigator layer"""
        if self.state.updating_contrast:
            return
            
        try:
            self.state.updating_contrast = True
            fov_contrast = fov_layer.contrast_limits
            
            if fov_layer.data.dtype == nav_layer.data.dtype:
                synced_contrast = fov_contrast
            else:
                synced_contrast = self._convert_contrast_proportionally(
                    fov_contrast,
                    self._get_layer_data_range(fov_layer),
                    self._get_layer_data_range(nav_layer)
                )
            
            nav_layer.contrast_limits = synced_contrast
            print(f"Synced FOV→Navigator contrast for {channel_name}: {fov_contrast} → {synced_contrast}")
            
        finally:
            self.state.updating_contrast = False
    
    def sync_navigator_to_fov(self, nav_layer, fov_layer, channel_name: str):
        """Sync contrast from navigator to FOV layer"""
        if self.state.updating_contrast:
            return
            
        try:
            self.state.updating_contrast = True
            nav_contrast = nav_layer.contrast_limits
            
            if nav_layer.data.dtype == fov_layer.data.dtype:
                synced_contrast = nav_contrast
            else:
                synced_contrast = self._convert_contrast_proportionally(
                    nav_contrast,
                    self._get_layer_data_range(nav_layer),
                    self._get_layer_data_range(fov_layer)
                )
            
            fov_layer.contrast_limits = synced_contrast
            print(f"Synced Navigator→FOV contrast for {channel_name}: {nav_contrast} → {synced_contrast}")
            
        finally:
            self.state.updating_contrast = False
    
    def _get_layer_data_range(self, layer) -> Tuple[float, float]:
        """Get the full data range for a layer using actual data percentiles"""
        try:
            data = layer.data
            
            # Both Navigator and FOV layers now use actual data range
            # This eliminates the mismatch that caused contrast amplification
            if hasattr(data, 'compute'):
                sample = data[..., ::5, ::5] if data.ndim >= 2 else data
                sample_computed = sample.compute()
                p1, p99 = np.percentile(sample_computed, [1, 99])
                actual_min, actual_max = sample_computed.min(), sample_computed.max()
                
                range_min = max(actual_min, p1 - (p99 - p1) * 0.1)
                range_max = min(actual_max, p99 + (p99 - p1) * 0.1)
                return (float(range_min), float(range_max))
            else:
                return (float(data.min()), float(data.max()))
                
        except Exception as e:
            print(f"Error getting data range: {e}")
            return layer.contrast_limits
    
    def _convert_contrast_proportionally(self, source_contrast: List[float], 
                                       source_range: Tuple[float, float],
                                       target_range: Tuple[float, float]) -> List[float]:
        """Convert contrast limits proportionally between different data ranges"""
        source_min, source_max = source_range
        target_min, target_max = target_range
        contrast_min, contrast_max = source_contrast
        
        source_span = source_max - source_min
        if source_span == 0:
            return list(target_range)
        
        # Calculate proportions
        min_prop = np.clip((contrast_min - source_min) / source_span, 0, 1)
        max_prop = np.clip((contrast_max - source_min) / source_span, 0, 1)
        
        # Ensure valid range
        if max_prop < min_prop:
            max_prop = min_prop
        
        # Apply to target range
        target_span = target_max - target_min
        new_min = target_min + (min_prop * target_span)
        new_max = target_min + (max_prop * target_span)
        
        # Final safety checks
        new_min = np.clip(new_min, target_min, target_max)
        new_max = np.clip(new_max, target_min, target_max)
        
        if new_max <= new_min:
            new_max = new_min + 1.0
        
        return [new_min, new_max]


class NavigatorOverlay:
    """Navigator overlay manager for napari viewer - Refactored"""
    
    def __init__(self, viewer, metadata, directory, fov_viewer=None):
        self.viewer = viewer
        self.metadata = metadata
        self.directory = directory
        self.fov_viewer = fov_viewer
        
        # Get scaling manager from FOV viewer if available
        self.scaling_manager = None
        if fov_viewer and hasattr(fov_viewer, 'scaling_manager'):
            self.scaling_manager = fov_viewer.scaling_manager
        
        # Configuration
        self.config = NavigatorConfig()
        
        # Update config with scaling manager values if available
        if self.scaling_manager:
            self.config.tile_size = self.scaling_manager.get_navigator_tile_size()
        
        # State management
        self.state = NavigatorState()
        
        # Helper classes
        self.layer_manager = LayerManager(viewer, self.config, self.scaling_manager)
        self.contrast_sync = ContrastSynchronizer(self.state)
        
        # Extract metadata
        self.dims = metadata['dimensions']
        self.region_names = metadata['regions']
        self.channel_names = metadata['channels']
        self.fov_names = metadata['fovs']
        
        # Storage
        self.region_channel_navigators = {}  # {(region_idx, channel_idx): (navigator, mosaic, metadata)}
        self.nav_channel_layers = {}         # {(region_idx, channel_idx): layer}
        self.nav_border_layers = {}          # {region_idx: layer}
        self.nav_box_layers = {}             # {region_idx: layer}
        
        # Channel mappings
        self.navigator_channel_names = None
        self.channel_mappings = self._initialize_channel_mappings()
        
        print("NavigatorOverlay initialized")
    
    def _initialize_channel_mappings(self) -> Dict[str, Dict]:
        """Initialize empty channel mapping dictionaries"""
        return {
            'navigator_to_idx': {},     # navigator channel name -> idx
            'idx_to_navigator': {},     # idx -> navigator channel name
            'napari_to_navigator': {}   # napari idx -> navigator idx
        }
    
    def is_available(self) -> bool:
        """Check if navigator functionality is available"""
        return NAVIGATOR_AVAILABLE and self.dims.get('region', 0) > 0 and self.dims.get('fov', 0) > 0
    
    def create_navigators(self) -> bool:
        """Create region-specific multi-channel navigator overlays"""
        if not self.is_available():
            print("Navigator not available - skipping creation")
            return False
        
        print("Creating region-specific multi-channel navigator overlays...")
        
        try:
            # Create downsampled navigators
            if not self._create_downsampled_navigators():
                return False
            
            # Create visualization layers
            self._create_combined_navigator_layers()
            
            # Setup interactivity
            self._setup_navigator_interactivity()
            
            # Initialize state
            self._initialize_navigator_state()
            
            print(f"Navigator overlays created successfully!")
            return True
            
        except Exception as e:
            print(f"Failed to create navigators: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_downsampled_navigators(self) -> bool:
        """Create downsampled navigator mosaics for each region"""
        def progress_callback(percent, message):
            print(f"Navigator: {percent}% - {message}")
        
        # Use scaling manager tile size if available, otherwise use config
        tile_size = self.config.tile_size
        if self.scaling_manager:
            tile_size = self.scaling_manager.get_navigator_tile_size()
        
        navigator = DownsampledNavigator(
            Path(self.directory),
            tile_size=tile_size,
            cache_enabled=self.config.tile_cache_enabled,
            progress_callback=progress_callback
        )
        
        current_time = self.viewer.dims.current_step[0] if self.viewer.dims.current_step else 0
        
        for region_idx, region_name in enumerate(self.region_names):
            print(f"Creating navigator for region {region_idx}:'{region_name}'")
            
            try:
                channel_mosaics, nav_metadata = navigator.create_channel_mosaics_for_region(
                    region_name, current_time
                )
                
                # Build channel mappings on first region
                if self.navigator_channel_names is None:
                    self._build_channel_mappings(channel_mosaics.keys())
                
                # Store navigator data
                for channel_name, mosaic_array in channel_mosaics.items():
                    if np.count_nonzero(mosaic_array) == 0:
                        print(f"WARNING: Empty mosaic for {region_name}, {channel_name}")
                        continue
                    
                    nav_idx = self.channel_mappings['navigator_to_idx'][channel_name]
                    key = (region_idx, nav_idx)
                    self.region_channel_navigators[key] = (navigator, mosaic_array, nav_metadata)
                    
            except Exception as e:
                print(f"Failed to create navigator for {region_name}: {e}")
                # Store empty metadata to prevent issues later
                if self.navigator_channel_names:
                    for nav_idx in range(len(self.navigator_channel_names)):
                        key = (region_idx, nav_idx)
                        self.region_channel_navigators[key] = (navigator, np.zeros((100, 100)), {})
                continue
        
        return len(self.region_channel_navigators) > 0
    
    def _build_channel_mappings(self, navigator_channels):
        """Build efficient channel index mappings"""
        self.navigator_channel_names = list(navigator_channels)
        
        # Navigator channel mappings
        for idx, ch in enumerate(self.navigator_channel_names):
            self.channel_mappings['navigator_to_idx'][ch] = idx
            self.channel_mappings['idx_to_navigator'][idx] = ch
        
        # Map napari to navigator channels by wavelength
        wavelengths = ['405', '488', '561', '638']
        for napari_idx, napari_ch in enumerate(self.channel_names):
            for nav_idx, nav_ch in enumerate(self.navigator_channel_names):
                if any(wl in napari_ch and wl in nav_ch for wl in wavelengths):
                    self.channel_mappings['napari_to_navigator'][napari_idx] = nav_idx
                    break
        
        print(f"Channel mappings built: {len(self.channel_mappings['napari_to_navigator'])} mapped")
    
    def _create_combined_navigator_layers(self):
        """Create napari layers for navigator overlays"""
        # Get positioning from main FOV layer
        fov_right, fov_top = self._get_fov_positioning_bounds()
        
        for region_idx, region_name in enumerate(self.region_names):
            is_first_region = (region_idx == 0)
            
            # Get available channels for this region
            available_channels = self._get_available_channels_for_region(region_idx)
            if not available_channels:
                continue
            
            # Create navigator layers for each channel
            for napari_idx, nav_idx, mosaic_array in available_channels:
                channel_name = self.channel_names[napari_idx]
                colormap = self._get_channel_colormap(napari_idx)
                
                position = (fov_top + self.config.pixel_gap_to_viewer, fov_right + self.config.pixel_gap_to_viewer)
                
                nav_layer = self.layer_manager.create_navigator_layer(
                    mosaic_array, region_name, channel_name, napari_idx,
                    region_idx, nav_idx, colormap, position, is_first_region
                )
                
                self.nav_channel_layers[(region_idx, napari_idx)] = nav_layer
            
            # Create border layer
            if available_channels:
                _, _, first_mosaic = available_channels[0]
                dimensions = first_mosaic.shape
                position = (fov_top + self.config.pixel_gap_to_viewer, fov_right + self.config.pixel_gap_to_viewer)
                
                border_layer = self.layer_manager.create_border_layer(
                    position, dimensions, region_name, is_first_region
                )
                self.nav_border_layers[region_idx] = border_layer
    
    def _get_fov_positioning_bounds(self) -> Tuple[float, float]:
        """Get bounds for positioning navigator relative to FOV"""
        # Find main FOV layer
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and 'Channel:' in layer.name:
                extent = layer.extent
                world_bounds = extent.world
                return world_bounds[1, -1], world_bounds[0, -2]  # right, top
        
        # Fallback
        return self.dims.get('x', 5000), 0
    
    def _get_available_channels_for_region(self, region_idx: int) -> List[Tuple[int, int, np.ndarray]]:
        """Get available navigator channels for a region"""
        available = []
        
        for napari_idx in range(len(self.channel_names)):
            nav_idx = self.channel_mappings['napari_to_navigator'].get(napari_idx, napari_idx)
            key = (region_idx, nav_idx)
            
            if key in self.region_channel_navigators:
                _, mosaic, _ = self.region_channel_navigators[key]
                available.append((napari_idx, nav_idx, mosaic))
        
        return available
    
    def _get_channel_colormap(self, channel_idx: int) -> str:
        """Get appropriate colormap for channel"""
        if channel_idx >= len(self.channel_names):
            return 'gray'
            
        channel_name = self.channel_names[channel_idx].lower()
        
        wavelength_colors = {
            '405': 'blue',
            '488': 'green',
            '561': 'yellow',
            '638': 'red',
            '640': 'red'
        }
        
        for wavelength, color in wavelength_colors.items():
            if wavelength in channel_name:
                return color
        
        return 'gray'
    
    def _setup_navigator_interactivity(self):
        """Setup all interactive behaviors"""
        # Click handlers
        for (region_idx, channel_idx), nav_layer in self.nav_channel_layers.items():
            nav_idx = self.channel_mappings['napari_to_navigator'].get(channel_idx, channel_idx)
            key = (region_idx, nav_idx)
            
            if key in self.region_channel_navigators:
                _, _, metadata = self.region_channel_navigators[key]
                self._add_click_handler(nav_layer, region_idx, channel_idx, metadata)
        
        # Dimension change handlers
        self.viewer.dims.events.current_step.connect(self._on_dimension_change)
        
        # Layer selection handlers
        self.viewer.layers.selection.events.active.connect(self._on_layer_selection_changed)
        
        # Contrast synchronization
        self._setup_contrast_sync()
        
        # Visibility synchronization
        self._setup_visibility_sync()
        
        # FOV viewer callback
        if self.fov_viewer:
            self.fov_viewer.set_navigator_visibility_callback(self._on_channel_visibility_changed)
    
    def _add_click_handler(self, nav_layer, region_idx: int, channel_idx: int, metadata: dict):
        """Add click handler to navigator layer"""
        @nav_layer.mouse_drag_callbacks.append
        def on_click(layer, event):
            if event.button != 1 or not nav_layer.visible:
                return
            
            # Convert click to grid position
            pos = event.position
            if len(pos) < 2:
                return
                
            viewer_y, viewer_x = pos[-2], pos[-1]
            nav_y, nav_x = nav_layer.translate
            
            local_x = viewer_x - nav_x
            local_y = viewer_y - nav_y
            
            height, width = nav_layer.data.shape
            if not (0 <= local_x < width and 0 <= local_y < height):
                return
            
            # Find FOV at grid position
            tile_size = metadata['tile_size']
            grid_col = int(local_x / tile_size)
            grid_row = int(local_y / tile_size)
            
            fov_grid = metadata.get('fov_grid', {})
            if (grid_row, grid_col) in fov_grid:
                clicked_fov = fov_grid[(grid_row, grid_col)]
                # clicked_fov is the actual FOV number for this region
                self._jump_to_fov(clicked_fov, region_idx)
    
    def _get_fovs_for_region(self, region_idx: int) -> List[str]:
        """Get list of FOV names that exist for a specific region"""
        print(f"DEBUG: _get_fovs_for_region called for region_idx={region_idx}")
        
        if region_idx >= len(self.region_names):
            print(f"DEBUG: region_idx {region_idx} >= len(self.region_names) {len(self.region_names)}")
            return []
        
        region_name = self.region_names[region_idx]
        print(f"DEBUG: Checking FOVs for region '{region_name}'")
        
        # Get metadata for this region
        metadata = self._get_navigator_metadata_for_region(region_idx)
        if not metadata:
            print(f"DEBUG: No metadata found for region {region_idx}")
            return []
        
        # Get the fov_to_grid_pos mapping from metadata (correct field name)
        fov_to_grid = metadata.get('fov_to_grid_pos', {})
        print(f"DEBUG: fov_to_grid_pos from metadata = {fov_to_grid}")
        
        # Get the region_fov_mapping to map between global FOV indices and region-specific FOV numbers
        region_fov_mapping = metadata.get('region_fov_mapping', {})
        print(f"DEBUG: region_fov_mapping = {region_fov_mapping}")
        
        # Debug: Print first few fov_names to understand the format
        print(f"DEBUG: self.fov_names (first 10) = {self.fov_names[:10]}")
        print(f"DEBUG: self.fov_names (last 10) = {self.fov_names[-10:]}")
        
        # Map between global FOV indices and region-specific FOV numbers
        # Use the actual FOV numbers from the navigator metadata instead of hardcoded values
        region_fovs = []
        
        # Get all FOVs that exist in this region from the navigator metadata
        available_fovs = list(fov_to_grid.keys())
        if available_fovs:
            # Sort the available FOVs to get consistent ordering
            available_fovs.sort()
            
            # Map each available FOV to the corresponding global FOV name
            for i, region_fov_number in enumerate(available_fovs):
                if i < len(self.fov_names):
                    region_fovs.append(self.fov_names[i])
                else:
                    # If we have more FOVs in the region than global FOV names, stop
                    break
        
        print(f"DEBUG: Final region_fovs = {region_fovs}")
        return region_fovs
    
    def _get_region_fov_index(self, global_fov_idx: int, region_idx: int) -> int:
        """Convert global FOV index to region-specific FOV index"""
        if global_fov_idx >= len(self.fov_names):
            return 0
        
        global_fov_name = self.fov_names[global_fov_idx]
        region_fovs = self._get_fovs_for_region(region_idx)
        
        # Find the region-specific index
        try:
            return region_fovs.index(global_fov_name)
        except ValueError:
            return 0  # Default to first FOV if not found
    
    def _get_global_fov_index(self, region_fov_idx: int, region_idx: int) -> int:
        """Convert region-specific FOV index to global FOV index"""
        region_fovs = self._get_fovs_for_region(region_idx)
        
        if region_fov_idx < len(region_fovs):
            region_fov_name = region_fovs[region_fov_idx]
            return self.fov_names.index(region_fov_name)
        
        return 0  # Default to first FOV if not found

    def _jump_to_fov(self, fov_name: str, region_idx: int):
        """Jump viewer to specified FOV from navigator click"""
        try:
            # fov_name here is the actual FOV number from the navigator
            fov_number = int(fov_name) if isinstance(fov_name, str) else fov_name
            
            # Get the navigator metadata for this region to determine FOV mapping
            metadata = self._get_navigator_metadata_for_region(region_idx)
            if not metadata:
                print(f"No navigator metadata found for region {region_idx}")
                return
            
            # Get the fov_to_grid_pos mapping to verify this FOV exists in the region
            fov_to_grid = metadata.get('fov_to_grid_pos', {})
            if fov_number not in fov_to_grid:
                print(f"FOV {fov_name} not found in region {region_idx} navigator")
                return
            
            # Convert the region-specific FOV number to a global FOV index
            # We need to find which position in self.fov_names corresponds to this FOV
            global_fov_idx = None
            
            # Method 1: Try to find the FOV number directly in fov_names (works for region 0)
            if str(fov_number) in self.fov_names:
                global_fov_idx = self.fov_names.index(str(fov_number))
            elif fov_number < len(self.fov_names):
                # Method 2: Direct index mapping (works for region 0 with integer indices)
                global_fov_idx = fov_number
            else:
                # Method 3: For regions with offset FOV numbers
                # We need to map the region FOV number to the corresponding global index
                # This assumes the global fov_names are ordered sequentially starting from 0
                
                # Get all FOVs for this region from the navigator metadata
                region_fovs = list(fov_to_grid.keys())
                if region_fovs:
                    # Sort the region FOVs to get consistent ordering
                    region_fovs.sort()
                    
                    # Find the position of our FOV in the sorted region FOVs
                    try:
                        region_fov_position = region_fovs.index(fov_number)
                        # Map this position to the global FOV index
                        # This assumes global fov_names are sequential starting from 0
                        global_fov_idx = region_fov_position
                    except ValueError:
                        print(f"FOV {fov_name} not found in region {region_idx} FOV list")
                        return
            
            # Ensure the index is valid
            if global_fov_idx is None or global_fov_idx < 0 or global_fov_idx >= len(self.fov_names):
                print(f"Invalid FOV index {global_fov_idx} for FOV {fov_name}")
                return
            
            # Check if this FOV is valid for the target region (using fov_viewer's fovs_per_region if available)
            if hasattr(self.fov_viewer, 'fovs_per_region'):
                num_fovs_in_region = self.fov_viewer.fovs_per_region.get(region_idx, len(self.fov_names))
                if global_fov_idx >= num_fovs_in_region:
                    print(f"FOV {fov_name} (index {global_fov_idx}) is beyond region {region_idx} limit ({num_fovs_in_region} FOVs)")
                    return
            
            # Temporarily disable the update flag to allow box update
            was_updating = self.state.updating_navigator
            self.state.updating_navigator = False
            
            try:
                current = list(self.viewer.dims.current_step)
                if len(current) >= 3:
                    current[1] = region_idx
                    current[2] = global_fov_idx
                    self.viewer.dims.current_step = current
                    print(f"Jumped to Region {region_idx}, FOV {fov_name} (global index {global_fov_idx})")
            finally:
                self.state.updating_navigator = was_updating
                
        except Exception as e:
            print(f"Error jumping to FOV: {e}")
    
    def _on_dimension_change(self):
        """Handle dimension changes"""
        if self.state.updating_navigator:
            return
        
        current_dims = tuple(self.viewer.dims.current_step) if self.viewer.dims.current_step else None
        
        if current_dims == self.state.last_dims_state:
            return
        
        self.state.last_dims_state = current_dims
        
        try:
            self.state.updating_navigator = True
            self._update_navigator_visibility()
            self._update_navigator_box(force_update=True)  # Force update on dimension change
        finally:
            self.state.updating_navigator = False
    
    def _on_layer_selection_changed(self, event):
        """Handle layer selection changes"""
        if self.state.updating_navigator:
            return
        
        layer = event.value if hasattr(event, 'value') else None
        if not layer or not hasattr(layer, 'name'):
            return
        
        # Prevent navigator box selection
        if '_Navigator Box' in layer.name:
            self._switch_to_main_layer()
            return
        
        # Handle specific layer types
        if 'Channel:' in layer.name and hasattr(layer, 'metadata'):
            self._handle_fov_channel_selection(layer)
        elif layer.name.startswith('_') and 'Fluorescence_' in layer.name:
            self._handle_navigator_selection(layer)
    
    def _switch_to_main_layer(self):
        """Switch selection to main data layer"""
        try:
            self.state.updating_navigator = True
            for layer in self.viewer.layers:
                if 'Multi-channel FOV Data' in layer.name:
                    self.viewer.layers.selection.active = layer
                    break
        finally:
            self.state.updating_navigator = False
    
    def _handle_fov_channel_selection(self, fov_layer):
        """Handle FOV channel selection"""
        try:
            self.state.updating_navigator = True
            
            channel_idx = fov_layer.metadata.get('channel_index')
            if channel_idx is None:
                return
            
            current_region = self._get_current_region()
            nav_layer = self.nav_channel_layers.get((current_region, channel_idx))
            
            if nav_layer and fov_layer.visible:
                nav_layer.visible = True
                self.contrast_sync.sync_fov_to_navigator(
                    fov_layer, nav_layer, self.channel_names[channel_idx]
                )
            
            self._update_navigator_box()
            
        finally:
            self.state.updating_navigator = False
    
    def _handle_navigator_selection(self, nav_layer):
        """Handle navigator layer selection"""
        if not hasattr(nav_layer, 'metadata'):
            return
        
        channel_idx = nav_layer.metadata.get('channel_idx')
        if channel_idx is None:
            return
        
        # Find corresponding FOV layer
        for layer in self.viewer.layers:
            if ('Channel:' in layer.name and 
                hasattr(layer, 'metadata') and
                layer.metadata.get('channel_index') == channel_idx):
                
                if nav_layer.visible and layer.visible:
                    self.contrast_sync.sync_navigator_to_fov(
                        nav_layer, layer, self.channel_names[channel_idx]
                    )
                break
    
    def _setup_contrast_sync(self):
        """Setup contrast synchronization between FOV and navigator layers"""
        # Find FOV channel layers
        fov_layers = {}
        for layer in self.viewer.layers:
            if 'Channel:' in layer.name and hasattr(layer, 'metadata'):
                idx = layer.metadata.get('channel_index')
                if idx is not None:
                    fov_layers[idx] = layer
        
        # Setup bidirectional sync
        for channel_idx, fov_layer in fov_layers.items():
            # Get all navigator layers for this channel
            nav_layers = [(r, l) for (r, c), l in self.nav_channel_layers.items() if c == channel_idx]
            
            if nav_layers:
                self._setup_contrast_sync_pair(channel_idx, fov_layer, nav_layers)
    
    def _setup_contrast_sync_pair(self, channel_idx: int, fov_layer, nav_layers: List[Tuple[int, Any]]):
        """Setup contrast sync between FOV and navigator layers"""
        channel_name = self.channel_names[channel_idx]
        
        # FOV -> Navigator
        @fov_layer.events.contrast_limits.connect
        def on_fov_contrast_change(event):
            if self.state.updating_contrast:
                return
            
            current_region = self._get_current_region()
            for region_idx, nav_layer in nav_layers:
                if region_idx == current_region and nav_layer.visible:
                    self.contrast_sync.sync_fov_to_navigator(fov_layer, nav_layer, channel_name)
        
        # Navigator -> FOV
        for region_idx, nav_layer in nav_layers:
            @nav_layer.events.contrast_limits.connect
            def on_nav_contrast_change(event, r_idx=region_idx, n_layer=nav_layer):
                if self.state.updating_contrast or not n_layer.visible:
                    return
                self.contrast_sync.sync_navigator_to_fov(n_layer, fov_layer, channel_name)
    
    def _setup_visibility_sync(self):
        """Setup visibility synchronization"""
        for (region_idx, channel_idx), nav_layer in self.nav_channel_layers.items():
            def create_handler(r_idx, c_idx, n_layer):
                @n_layer.events.visible.connect
                def on_visibility_change(event):
                    if self.state.updating_navigator:
                        return
                    
                    current_region = self._get_current_region()
                    if r_idx != current_region:
                        return
                    
                    # Sync to FOV layer
                    if self.fov_viewer:
                        fov_layer = self.fov_viewer.get_channel_layer(c_idx)
                        if fov_layer and fov_layer.visible != n_layer.visible:
                            try:
                                self.state.updating_navigator = True
                                if hasattr(self.fov_viewer, '_updating_visibility'):
                                    self.fov_viewer._updating_visibility = True
                                fov_layer.visible = n_layer.visible
                            finally:
                                self.state.updating_navigator = False
                                if hasattr(self.fov_viewer, '_updating_visibility'):
                                    self.fov_viewer._updating_visibility = False
                
                return on_visibility_change
            
            handler = create_handler(region_idx, channel_idx, nav_layer)
            self.state.visibility_handlers[(region_idx, channel_idx)] = handler
    
    def _initialize_navigator_state(self):
        """Initialize navigator to match current viewer state"""
        print("Initializing navigator state...")
        self._update_navigator_visibility()
        self._update_navigator_box()
        print("Navigator state initialized successfully")
    
    def _on_channel_visibility_changed(self):
        """Callback when FOV channel visibility changes"""
        if self.state.updating_navigator:
            return
        
        try:
            self.state.updating_navigator = True
            self._update_navigator_visibility()
        finally:
            self.state.updating_navigator = False
    
    def _update_navigator_visibility(self):
        """Update navigator visibility to match current region"""
        current_region = self._get_current_region()
        
        # Update navigator channel layers - only show current region's layers
        for (region_idx, channel_idx), nav_layer in self.nav_channel_layers.items():
            should_be_visible = (region_idx == current_region)
            if nav_layer.visible != should_be_visible:
                nav_layer.visible = should_be_visible
        
        # Update border layers
        for region_idx, border_layer in self.nav_border_layers.items():
            should_be_visible = (region_idx == current_region)
            if border_layer.visible != should_be_visible:
                border_layer.visible = should_be_visible
        
        # Update box position after visibility change
        self._update_navigator_box(force_update=True)
    
    def _update_navigator_box(self, force_update=False):
        """Update navigator box to show current FOV position"""
        if self.state.updating_navigator and not force_update:
            return
            
        try:
            # Remove existing boxes
            self.layer_manager.remove_navigator_boxes()
            self.nav_box_layers.clear()
            
            # Get current position
            current_step = self.viewer.dims.current_step
            if len(current_step) < 3:
                return
                
            region_idx = current_step[1]
            fov_idx = current_step[2]
            
            if region_idx >= len(self.region_names) or fov_idx >= len(self.fov_names):
                return
            
            # Get the current FOV name from the global list
            current_fov_name = self.fov_names[fov_idx]
            
            # Check if any navigator is visible for this region
            visible_nav = self._get_visible_navigator_for_region(region_idx)
            if not visible_nav:
                return
            
            # Get metadata for box positioning
            metadata = self._get_navigator_metadata_for_region(region_idx)
            if not metadata:
                return
            
            # Find FOV position in grid
            # Convert the global FOV index to the region-specific FOV number used in the navigator
            fov_to_grid = metadata.get('fov_to_grid_pos', {})
            
            # Get all FOVs available in this region from the navigator metadata
            available_fovs = list(fov_to_grid.keys())
            if not available_fovs:
                return
            
            # Sort the available FOVs to get consistent ordering
            available_fovs.sort()
            
            # Map the global FOV index to the region-specific FOV number
            if fov_idx < len(available_fovs):
                region_fov_number = available_fovs[fov_idx]
                grid_row, grid_col = fov_to_grid[region_fov_number]
            else:
                # FOV index is beyond what's available in this region
                return
            
            tile_size = metadata['tile_size']
            
            # Calculate viewer position
            nav_local_x = grid_col * tile_size + tile_size // 2
            nav_local_y = grid_row * tile_size + tile_size // 2
            
            nav_y_pos, nav_x_pos = visible_nav.translate
            viewer_x = nav_local_x + nav_x_pos
            viewer_y = nav_local_y + nav_y_pos
            
            # Create box
            box_size = tile_size * 0.8
            box_layer = self.layer_manager.create_box_layer(
                (viewer_y, viewer_x), box_size, self.region_names[region_idx]
            )
            
            self.nav_box_layers[region_idx] = box_layer
            
            # Prevent box selection
            if self.viewer.layers.selection.active == box_layer:
                self.viewer.layers.selection.active = visible_nav
        
        except Exception as e:
            print(f"Error updating navigator box: {e}")
            import traceback
            traceback.print_exc()
    
    def _get_current_region(self) -> int:
        """Get current region index"""
        current_step = self.viewer.dims.current_step
        return current_step[1] if len(current_step) >= 2 else 0
    
    def _get_visible_navigator_for_region(self, region_idx: int):
        """Get the navigator layer for a specific region (visibility doesn't matter for box positioning)"""
        if region_idx >= len(self.region_names):
            return None
        
        # Look for any navigator layer for this region (visibility doesn't matter for box positioning)
        for (r_idx, c_idx), layer in self.nav_channel_layers.items():
            if r_idx == region_idx:  # Remove the layer.visible check
                return layer
        
        return None
    
    def _get_navigator_metadata_for_region(self, region_idx: int) -> Optional[dict]:
        """Get metadata for a specific region's navigator"""
        print(f"DEBUG: _get_navigator_metadata_for_region called for region_idx={region_idx}")
        
        if region_idx >= len(self.region_names):
            print(f"DEBUG: region_idx {region_idx} >= len(self.region_names) {len(self.region_names)}")
            return None
        
        print(f"DEBUG: self.region_channel_navigators keys = {list(self.region_channel_navigators.keys())}")
        
        # Get metadata from the first available channel for this region
        for nav_idx in range(len(self.navigator_channel_names or [])):
            key = (region_idx, nav_idx)
            print(f"DEBUG: Checking key {key}")
            if key in self.region_channel_navigators:
                _, _, metadata = self.region_channel_navigators[key]
                print(f"DEBUG: Found metadata for key {key}")
                return metadata
        
        print(f"DEBUG: No metadata found for region {region_idx}")
        return None
    
    def diagnose_visibility_sync(self):
        """Diagnostic method to check visibility synchronization"""
        print("=== VISIBILITY SYNC DIAGNOSTIC ===")
        current_region = self._get_current_region()
        visible_channels = set(self.fov_viewer.get_visible_channels()) if self.fov_viewer else set()
        
        print(f"Current region: {current_region}")
        print(f"FOV visible channels: {visible_channels}")
        print(f"Updating flag: {self.state.updating_navigator}")
        
        for (region_idx, channel_idx), nav_layer in self.nav_channel_layers.items():
            should_be_visible = (region_idx == current_region and channel_idx in visible_channels)
            status = "✓" if (should_be_visible == nav_layer.visible) else "✗"
            
            channel_name = self.channel_names[channel_idx]
            region_name = self.region_names[region_idx]
            
            print(f"{status} {region_name} {channel_name}: should={should_be_visible}, actual={nav_layer.visible}")
        
        print("=== END DIAGNOSTIC ===")


def add_navigator_to_viewer(viewer, metadata, directory, fov_viewer=None):
    """
    Add navigator overlay to an existing napari viewer.
    
    Parameters:
    -----------
    viewer : napari.Viewer
        The napari viewer instance
    metadata : dict
        Metadata dictionary
    directory : str
        Path to acquisition directory
    fov_viewer : FOVViewer, optional
        The FOV viewer instance for visible channels
        
    Returns:
    --------
    NavigatorOverlay or None
    """
    navigator = NavigatorOverlay(viewer, metadata, directory, fov_viewer)
    
    if navigator.create_navigators():
        return navigator
    else:
        return None
