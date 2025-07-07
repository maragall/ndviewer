#!/usr/bin/env python3
"""
FOV Viewer Module
Handles the core napari viewer setup for multi-dimensional FOV data.
"""

import os
import sys
import pickle
import napari
import dask.array as da
import dask
import tifffile
import numpy as np
from napari.layers import Image
from pathlib import Path
from typing import List, Optional

# Import our scaling manager
from scaling_manager import ScalingManager

class FOVViewer:
    """Main FOV viewer class that sets up napari for multi-dimensional data."""
    
    def __init__(self, temp_file_path, enable_navigator=True):
        """
        Initialize the FOV viewer from a temporary metadata file.
        
        Parameters:
        -----------
        temp_file_path : str
            Path to the temporary file containing metadata
        enable_navigator : bool, optional
            Whether to enable navigator functionality (default: True)
        """
        # Load the metadata from the temp file
        with open(temp_file_path, 'rb') as f:
            data = pickle.load(f)

        self.directory = data['directory']
        self.metadata = data['metadata']
        self.folder_name = data['folder_name']
        
        # Get dimensions and data
        self.dims = self.metadata['dimensions']
        self.file_map = self.metadata['file_map']
        self.acq_params = self.metadata['acquisition_parameters']
        
        # Extract names for layers
        self.channel_names = self.metadata['channels']
        self.region_names = self.metadata['regions']
        self.fov_names = self.metadata['fovs']
        
        # Initialize scaling manager
        self.scaling_manager = ScalingManager(self.acq_params)
        
        # Initialize viewer
        self.viewer = None
        self.channel_layers = {}  # Store individual channel layers
        self.dask_data_per_channel = {}  # Store dask arrays per channel
        
        # Flag to prevent infinite loops during visibility synchronization
        self._updating_visibility = False
        
        # Navigator preference
        self.enable_navigator = enable_navigator
        
        print(f"FOVViewer initialized for: {self.folder_name}")
        print(f"Navigator: {'Enabled' if self.enable_navigator else 'Disabled'}")
        print(f"Directory: {self.directory}")
        print(f"Dimensions: {self.dims}")
        print(f"Channel names: {self.channel_names}")
        print(f"Region names: {self.region_names}")
        print(f"FOV names: {self.fov_names}")
    
    def _create_dask_arrays_per_channel(self):
        """Create separate dask arrays for each channel for lazy loading of TIFF files."""
        
        # Create a function that loads TIFF files on demand
        @dask.delayed
        def load_tiff(t, r, f, z, c):
            key = (t, r, f, z, c)
            if key in self.file_map:
                return tifffile.imread(self.file_map[key])
            else:
                return np.zeros((self.dims['y'], self.dims['x']), dtype=np.uint16)

        print("Creating separate dask arrays for each channel...")
        
        # Create separate dask arrays for each channel
        for c in range(self.dims['channel']):
            lazy_arrays = []
            for t in range(self.dims['time']):
                region_arrays = []
                for r in range(self.dims['region']):
                    fov_arrays = []
                    for f in range(self.dims['fov']):
                        z_arrays = []
                        for z in range(self.dims['z']):
                            # Create a delayed reader for each position
                            delayed_reader = load_tiff(t, r, f, z, c)
                            # Convert to a dask array
                            sample_shape = (self.dims['y'], self.dims['x'])
                            lazy_array = da.from_delayed(delayed_reader, shape=sample_shape, dtype=np.uint16)
                            z_arrays.append(lazy_array)
                        fov_arrays.append(da.stack(z_arrays))
                    region_arrays.append(da.stack(fov_arrays))
                lazy_arrays.append(da.stack(region_arrays))

            # Stack everything into a 6D array for this channel: (t, r, f, z, y, x)
            channel_dask_data = da.stack(lazy_arrays)
            self.dask_data_per_channel[c] = channel_dask_data
            print(f"Channel {c} ({self.channel_names[c]}): dask array shape: {channel_dask_data.shape}")
    
    def create_viewer(self):
        """Create and configure the napari viewer."""
        
        # Create dask arrays per channel
        self._create_dask_arrays_per_channel()
        
        # Create napari viewer
        print(f"Opening napari viewer for: {self.folder_name}")
        self.viewer = napari.Viewer(title=f"Napari - {self.folder_name}")

        # Apply layer filter RIGHT AFTER creating viewer, BEFORE adding any layers
        # Only apply filter if navigator is enabled
        if self.enable_navigator:
            def _navigator_filter(row, parent):
                return "<hidden>" not in self.viewer.layers[row].name
            
            self.viewer.window.qt_viewer.layers.model().filterAcceptsRow = _navigator_filter
            print("✓ Layer filtering applied to viewer (navigator enabled)")

        # Get scaling information from scaling manager using the simple formula
        scale = self.scaling_manager.calculate_fov_scale(self.dims)

        # Add each channel as a separate layer
        print("Creating separate layers for each channel...")
        for c in range(self.dims['channel']):
            channel_name = self.channel_names[c] if c < len(self.channel_names) else f"Channel {c}"
            
            # Determine layer name based on navigator preference
            if self.enable_navigator:
                layer_name = f"Channel: {channel_name} <hidden>"
            else:
                layer_name = f"Channel: {channel_name}"
            
            # Add each channel as a separate 6D layer: (t, r, f, z, y, x)
            layer = self.viewer.add_image(
                self.dask_data_per_channel[c],
                name=layer_name,
                blending='additive',
                colormap=self._get_channel_colormap(c),
                visible=True,  # All channels visible by default
                multiscale=False,
                metadata={
                    'dimension_names': ['Time', 'Region', 'FOV', 'Z', 'Y', 'X'],
                    'channel_index': c,
                    'channel_name': channel_name
                }
            )
            
            # Apply scale after layer creation to ensure it takes effect
            layer.scale = scale
            
            # Store the layer reference
            self.channel_layers[c] = layer
            print(f"Added layer for {channel_name}")

        # Set dimension labels for the sliders (no channel dimension now)
        self.viewer.dims.axis_labels = ['Time', 'Region', 'FOV', 'Z', 'Y', 'X']

        # Add physical units to the dimension labels
        if self.scaling_manager.should_enable_scale_bar():
            self.viewer.scale_bar.unit = self.scaling_manager.get_scale_bar_unit()
            self.viewer.scale_bar.visible = True

        # Add info update callback
        self._setup_info_callback()
        
        # Setup layer visibility change callback for navigator updates
        self._setup_layer_visibility_callback()
        
        # Setup FOV position validation callback
        self._setup_fov_validation_callback()
        
        print("FOV viewer setup complete!")
        print(f"Created {len(self.channel_layers)} channel layers")
        print(f"Dimension labels: {self.viewer.dims.axis_labels}")
        print(f"Channel names: {self.channel_names}")
        return self.viewer
    
    def _get_channel_colormap(self, channel_idx):
        """Get appropriate colormap for each channel based on name/wavelength."""
        if channel_idx < len(self.channel_names):
            channel_name = self.channel_names[channel_idx].lower()
            
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
    
    def _setup_info_callback(self):
        """Setup callback to update window title with current region and FOV info."""
        
        @self.viewer.dims.events.current_step.connect
        def update_info(event):
            # This will update when sliders change
            current_dims = self.viewer.dims.current_step
            if len(current_dims) >= 3:  # Make sure we have enough dimensions (no channel dim now)
                region_idx = current_dims[1]
                fov_idx = current_dims[2]
                
                # Get names for display
                region_name = self.region_names[region_idx] if region_idx < len(self.region_names) else f"Region {region_idx}"
                fov_name = self.fov_names[fov_idx] if fov_idx < len(self.fov_names) else f"FOV {fov_idx}"
                
                # Count visible channels
                visible_channels = [name for c, layer in self.channel_layers.items() 
                                  if layer.visible for name in [self.channel_names[c] if c < len(self.channel_names) else f"Channel {c}"]]
                
                # Update window title with current region, FOV and visible channels
                if visible_channels:
                    channels_str = f"Channels: {', '.join(visible_channels[:3])}" + ("..." if len(visible_channels) > 3 else "")
                else:
                    channels_str = "No channels visible"
                    
                self.viewer.title = f"Napari - {self.folder_name} - {channels_str} - Region: {region_name}, FOV: {fov_name}"
    
    def _setup_layer_visibility_callback(self):
        """Setup callback to notify navigator when layer visibility changes."""
        
        # Connect to each layer's visibility events individually
        for c, layer in self.channel_layers.items():
            @layer.events.visible.connect
            def on_layer_visibility_changed(event):
                # Prevent infinite loops during bidirectional synchronization
                if self._updating_visibility:
                    return
                
                # This will be called when a layer's visibility changes
                # Notify the navigator to update its display
                if hasattr(self, '_notify_navigator_visibility_change'):
                    try:
                        self._updating_visibility = True
                        self._notify_navigator_visibility_change()
                    finally:
                        self._updating_visibility = False

    def _setup_fov_validation_callback(self):
        """Setup callback to validate FOV position when dimensions change."""
        
        @self.viewer.dims.events.current_step.connect
        def validate_fov_position(event):
            self.validate_fov_position(event)
    
    def set_navigator_visibility_callback(self, callback):
        """Set callback to notify navigator when channel visibility changes."""
        self._notify_navigator_visibility_change = callback
    
    def get_visible_channels(self):
        """Get list of currently visible channel indices."""
        visible_channels = []
        for c, layer in self.channel_layers.items():
            if layer.visible:
                visible_channels.append(c)
        return visible_channels
    
    def get_channel_layer(self, channel_idx):
        """Get the layer for a specific channel."""
        return self.channel_layers.get(channel_idx)
    
    def get_viewer(self):
        """Get the napari viewer instance."""
        return self.viewer
    
    def get_metadata(self):
        """Get the metadata dictionary."""
        return self.metadata
    
    def get_scaling_manager(self):
        """Get the scaling manager instance."""
        return self.scaling_manager
    
    def run(self):
        """Run the napari event loop."""
        napari.run()

    def _update_navigator_box(self, force_update=False):
        # Get current position
        current_step = self.viewer.dims.current_step
        region_idx = current_step[1]
        global_fov_idx = current_step[2]
        
        # Convert to region-specific FOV name
        region_fovs = self._get_fovs_for_region(region_idx)
        region_fov_idx = self._get_region_fov_index(global_fov_idx, region_idx)
        
        if region_fov_idx < len(region_fovs):
            current_fov = region_fovs[region_fov_idx]
            
            # Get metadata for box positioning
            metadata = self._get_navigator_metadata_for_region(region_idx)
            if metadata:
                fov_to_grid = metadata.get('fov_to_grid_pos', {})
                if current_fov in fov_to_grid:
                    # Create box at correct position
                    grid_row, grid_col = fov_to_grid[current_fov]
                    tile_size = metadata.get('tile_size', 75)
                    
                    # Calculate viewer position
                    nav_local_x = grid_col * tile_size + tile_size // 2
                    nav_local_y = grid_row * tile_size + tile_size // 2
                    
                    # Note: This is a placeholder - actual implementation would need navigator layer reference
                    print(f"Would create box at grid position ({grid_row}, {grid_col}) for FOV {current_fov}")

    def _jump_to_fov(self, fov_name: str, region_idx: int):
        """Jump viewer to specified FOV"""
        try:
            # Get region-specific FOV index
            region_fovs = self._get_fovs_for_region(region_idx)
            region_fov_idx = region_fovs.index(fov_name) if fov_name in region_fovs else 0
            
            # Convert to global FOV index for the slider
            global_fov_idx = self._get_global_fov_index(region_fov_idx, region_idx)
            
            current = list(self.viewer.dims.current_step)
            if len(current) >= 3:
                current[1] = region_idx
                current[2] = global_fov_idx  # Use global index for slider
                self.viewer.dims.current_step = current
                print(f"Jumped to Region {region_idx}, FOV {fov_name} (global index: {global_fov_idx})")
        except Exception as e:
            print(f"Error jumping to FOV: {e}")

    def _get_region_fov_index(self, global_fov_idx: int, region_idx: int) -> int:
        """Convert global FOV index to region-specific FOV index"""
        global_fov_name = self.fov_names[global_fov_idx]
        
        # Get FOVs for this region
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

    def _get_fovs_for_region(self, region_idx: int) -> List[str]:
        """Get list of FOV names that exist for a specific region"""
        if region_idx >= len(self.region_names):
            return []
        
        region_name = self.region_names[region_idx]
        region_fovs = []
        
        # Check which FOVs exist in the file map for this region
        for fov_name in self.fov_names:
            # Check if this FOV exists in the file map for this region
            key = (0, region_idx, self.fov_names.index(fov_name), 0, 0)  # Sample key
            if key in self.file_map:
                region_fovs.append(fov_name)
        
        return region_fovs
    
    def _get_navigator_metadata_for_region(self, region_idx: int) -> Optional[dict]:
        """Get metadata for a specific region's navigator"""
        if region_idx < len(self.region_names):
            region_name = self.region_names[region_idx]
            # This would need to be connected to the navigator's metadata
            # For now, return None as this is handled by the navigator
            return None
        return None

    def validate_fov_position(self, event):
        """Validate and correct FOV position when it's out of range for the current region."""
        current_step = self.viewer.dims.current_step
        if len(current_step) < 3:
            return
            
        region_idx = current_step[1]
        fov_idx = current_step[2]
        
        # Get valid FOVs for current region
        valid_fovs = self._get_fovs_for_region(region_idx)
        
        if fov_idx >= len(valid_fovs):
            # Jump to last valid FOV instead of first
            new_step = list(current_step)
            new_step[2] = len(valid_fovs) - 1 if valid_fovs else 0
            self.viewer.dims.current_step = new_step
            print(f"Jumped to last FOV of region {region_idx} (index: {new_step[2]})")

def create_fov_viewer(temp_file_path, enable_navigator=True):
    """
    Convenience function to create and return a configured FOV viewer.
    
    Parameters:
    -----------
    temp_file_path : str
        Path to the temporary metadata file
    enable_navigator : bool, optional
        Whether to enable navigator functionality (default: True)
        
    Returns:
    --------
    FOVViewer: Configured FOV viewer instance
    """
    viewer = FOVViewer(temp_file_path, enable_navigator=enable_navigator)
    viewer.create_viewer()
    return viewer 