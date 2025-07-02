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

class FOVViewer:
    """Main FOV viewer class that sets up napari for multi-dimensional data."""
    
    def __init__(self, temp_file_path):
        """
        Initialize the FOV viewer from a temporary metadata file.
        
        Parameters:
        -----------
        temp_file_path : str
            Path to the temporary file containing metadata
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
        
        # Calculate physical scales
        self.pixel_size_um = self._calculate_pixel_size()
        self.z_step_um = self._calculate_z_step()
        
        # Initialize viewer
        self.viewer = None
        self.channel_layers = {}  # Store individual channel layers
        self.dask_data_per_channel = {}  # Store dask arrays per channel
        
        print(f"FOVViewer initialized for: {self.folder_name}")
        print(f"Directory: {self.directory}")
        print(f"Dimensions: {self.dims}")
        print(f"Channel names: {self.channel_names}")
        print(f"Region names: {self.region_names}")
        print(f"FOV names: {self.fov_names}")
    
    def _calculate_pixel_size(self):
        """Calculate pixel size in microns."""
        pixel_size_um = None
        
        if self.acq_params:
            # Get pixel size from sensor pixel size and objective magnification
            if ('sensor_pixel_size_um' in self.acq_params and 
                'objective' in self.acq_params and 
                'magnification' in self.acq_params['objective']):
                sensor_pixel_size = self.acq_params['sensor_pixel_size_um']
                magnification = self.acq_params['objective']['magnification']
                pixel_size_um = sensor_pixel_size / magnification
                print(f"Calculated pixel size: {pixel_size_um:.3f} µm")
        
        # Default values if not found in parameters
        if pixel_size_um is None:
            pixel_size_um = 1.0
            print("Warning: Using default pixel size of 1.0 µm")
            
        return pixel_size_um
    
    def _calculate_z_step(self):
        """Calculate Z step size in microns."""
        z_step_um = None
        
        if self.acq_params:
            # Get Z step size
            if 'dz(um)' in self.acq_params:
                z_step_um = self.acq_params['dz(um)']
                print(f"Z step size: {z_step_um} µm")
        
        if z_step_um is None:
            z_step_um = 1.0
            print("Warning: Using default Z step size of 1.0 µm")
            
        return z_step_um
    
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

        # The dimensions are: (t, r, f, z, y, x) for each channel
        # Calculate scale for each dimension
        # [time, region, fov, z, y, x]
        # For time, region, and fov, we use 1.0 as they're indices
        # For z, y, x we use the physical dimensions
        scale = [1.0, 1.0, 1.0, self.z_step_um, self.pixel_size_um, self.pixel_size_um]

        # Store raw image dimensions for reference
        raw_fov_width = self.dims['x']
        raw_fov_height = self.dims['y']
        raw_fov_area = raw_fov_width * raw_fov_height

        print("="*60)
        print("RAW IMAGE DIMENSIONS (for reference)")
        print("="*60)
        print(f"Raw FOV image: {raw_fov_width} x {raw_fov_height} pixels ({raw_fov_area:,} pixels)")
        print(f"Raw navigator tile: 75 x 75 pixels (5,625 pixels)")
        print(f"Raw area ratio: {raw_fov_area/5625:.1f}:1")
        print("="*60)

        # Add each channel as a separate layer
        print("Creating separate layers for each channel...")
        for c in range(self.dims['channel']):
            channel_name = self.channel_names[c] if c < len(self.channel_names) else f"Channel {c}"
            
            # Add each channel as a separate 6D layer: (t, r, f, z, y, x)
            layer = self.viewer.add_image(
                self.dask_data_per_channel[c],
                name=f"Channel: {channel_name}",
                scale=scale,  # Apply physical scale
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
            
            # Store the layer reference
            self.channel_layers[c] = layer
            print(f"Added layer for {channel_name}")

        # Set dimension labels for the sliders (no channel dimension now)
        self.viewer.dims.axis_labels = ['Time', 'Region', 'FOV', 'Z', 'Y', 'X']

        # Add physical units to the dimension labels
        if self.z_step_um is not None and self.pixel_size_um is not None:
            self.viewer.scale_bar.unit = 'µm'
            self.viewer.scale_bar.visible = True

        # Add info update callback
        self._setup_info_callback()
        
        # Setup layer visibility change callback for navigator updates
        self._setup_layer_visibility_callback()
        
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
                # Default colors for additional channels
                colors = ['cyan', 'magenta', 'orange', 'pink']
                return colors[channel_idx % len(colors)]
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
                # This will be called when a layer's visibility changes
                # Notify the navigator to update its display
                if hasattr(self, '_notify_navigator_visibility_change'):
                    self._notify_navigator_visibility_change()
    
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
    
    def run(self):
        """Run the napari event loop."""
        napari.run()

def create_fov_viewer(temp_file_path):
    """
    Convenience function to create and return a configured FOV viewer.
    
    Parameters:
    -----------
    temp_file_path : str
        Path to the temporary metadata file
        
    Returns:
    --------
    FOVViewer: Configured FOV viewer instance
    """
    viewer = FOVViewer(temp_file_path)
    viewer.create_viewer()
    return viewer 