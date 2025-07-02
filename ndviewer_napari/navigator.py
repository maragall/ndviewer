#!/usr/bin/env python3
"""
Navigator Module
Handles navigator overlay functionality for napari viewer.
Provides region-specific multi-channel navigator overlays with click handling.
"""

import os
import sys
import numpy as np
from pathlib import Path

# Try to import downsampler from parent directory
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from downsampler import DownsampledNavigator
    NAVIGATOR_AVAILABLE = True
    print("Successfully imported downsampler module in navigator")
except ImportError as e:
    print(f"Warning: Navigator module not available: {e}")
    NAVIGATOR_AVAILABLE = False

class NavigatorOverlay:
    """Navigator overlay manager for napari viewer."""
    
    def __init__(self, viewer, metadata, directory, fov_viewer=None):
        """
        Initialize the navigator overlay.
        
        Parameters:
        -----------
        viewer : napari.Viewer
            The napari viewer instance to add navigators to
        metadata : dict
            Metadata dictionary containing dimensions, regions, channels, etc.
        directory : str
            Path to the acquisition directory
        fov_viewer : FOVViewer, optional
            The FOV viewer instance for getting visible channels
        """
        self.viewer = viewer
        self.metadata = metadata
        self.directory = directory
        self.fov_viewer = fov_viewer
        
        # Extract metadata
        self.dims = metadata['dimensions']
        self.region_names = metadata['regions']
        self.channel_names = metadata['channels']
        self.fov_names = metadata['fovs']
        
        # Storage for navigators per region-channel combination
        # EFFICIENCY: Use integer-based keys (region_idx, channel_idx) instead of strings
        self.region_channel_navigators = {}  # {(region_idx, channel_idx): (navigator, mosaic_array, metadata)}
        
        # Storage for individual channel navigator layers per region
        self.nav_channel_layers = {}         # {(region_idx, channel_idx): navigator_layer}
        self.nav_border_layers = {}          # {region_idx: border_layer}
        self.nav_box_layers = {}             # {region_idx: box_layer}
        
        # CRITICAL: Add flag to prevent recursive calls that cause infinite loops
        self._updating_navigator = False
        
        # Channel mappings for efficiency
        self.navigator_channel_names = None  # Will be populated after first region
        self.navigator_channel_to_idx = {}  # Map downsampler channel name -> integer index
        self.idx_to_navigator_channel = {}  # Map integer index -> downsampler channel name
        self.napari_idx_to_navigator_idx = {}  # Map napari channel index -> navigator channel index
        
        print("NavigatorOverlay initialized")
    
    def is_available(self):
        """Check if navigator functionality is available."""
        return NAVIGATOR_AVAILABLE and self.dims['region'] > 0 and self.dims['fov'] > 0
    
    def create_navigators(self):
        """Create region-specific multi-channel navigator overlays."""
        
        if not self.is_available():
            print("Navigator not available - skipping navigator creation")
            if not NAVIGATOR_AVAILABLE:
                print("Navigator not available - module not imported")
            else:
                print(f"Navigator conditions not met: regions={self.dims.get('region', 0)}, fovs={self.dims.get('fov', 0)}")
            return False
        
        print("Creating region-specific multi-channel navigator overlays...")
        
        try:
            # Create navigator with progress callback
            def progress_callback(percent, message):
                print(f"Navigator: {percent}% - {message}")
            
            # Get unique regions and channels from metadata
            unique_regions = self.region_names  # This is the list of region names
            unique_channels = self.channel_names  # This is the list of channel names
            
            print(f"Creating multi-channel navigators for regions: {unique_regions}")
            print(f"Channels: {unique_channels}")
            print(f"Total navigator images to create: {len(unique_regions)} regions × {len(unique_channels)} channels = {len(unique_regions) * len(unique_channels)}")
            print(f"Total FOVs in dataset: {len(self.fov_names)}")
            
            # EFFICIENCY OPTIMIZATION: Use integer-based channel processing
            # Convert all channel operations to integer indices for computational efficiency
            num_channels = len(unique_channels)
            num_regions = len(unique_regions)
            
            # Create integer-based mappings for fast lookups
            napari_channel_to_idx = {ch: idx for idx, ch in enumerate(unique_channels)}
            idx_to_napari_channel = {idx: ch for idx, ch in enumerate(unique_channels)}
            
            print(f"\nOptimized for integer-based processing:")
            print(f"Napari channels ({num_channels}): {[f'{i}:{ch}' for i, ch in enumerate(unique_channels)]}")
            print(f"Regions ({num_regions}): {[f'{i}:{reg}' for i, reg in enumerate(unique_regions)]}")
            
            # Create navigator instance once
            navigator = DownsampledNavigator(
                Path(self.directory), 
                tile_size=75,  # Good balance of quality and speed
                cache_enabled=True,
                progress_callback=progress_callback
            )
            
            # Create multi-channel mosaics for each region
            current_time = self.viewer.dims.current_step[0] if self.viewer.dims.current_step else 0
            
            for region_idx, region_name in enumerate(unique_regions):
                print(f"Creating multi-channel navigator mosaics for region {region_idx}:'{region_name}' at timepoint {current_time}")
                
                try:
                    # Get all channel mosaics for this region
                    channel_mosaics, nav_metadata = navigator.create_channel_mosaics_for_region(region_name, current_time)
                    
                    print(f"Multi-channel navigator mosaics created for {region_name}: {len(channel_mosaics)} channels")
                    
                    # Build channel mapping on first region (all regions should have same channels)
                    if self.navigator_channel_names is None:
                        self.navigator_channel_names = list(channel_mosaics.keys())
                        
                        # Create integer-based mappings for navigator channels
                        self.navigator_channel_to_idx = {ch: idx for idx, ch in enumerate(self.navigator_channel_names)}
                        self.idx_to_navigator_channel = {idx: ch for idx, ch in enumerate(self.navigator_channel_names)}
                        
                        print(f"Downsampler channels: {[f'{i}:{ch}' for i, ch in enumerate(self.navigator_channel_names)]}")
                        
                        # Build efficient mapping between napari and downsampler channel indices
                        for napari_idx, napari_ch in enumerate(unique_channels):
                            for nav_idx, nav_ch in enumerate(self.navigator_channel_names):
                                # Match by common wavelength patterns
                                if any(wavelength in napari_ch and wavelength in nav_ch 
                                      for wavelength in ['405', '488', '561', '638']):
                                    self.napari_idx_to_navigator_idx[napari_idx] = nav_idx
                                    break
                        
                        print(f"Efficient channel index mapping:")
                        for napari_idx, nav_idx in self.napari_idx_to_navigator_idx.items():
                            napari_ch = idx_to_napari_channel[napari_idx]
                            nav_ch = self.idx_to_navigator_channel[nav_idx]
                            print(f"  {napari_idx}:{napari_ch} -> {nav_idx}:{nav_ch}")
                    
                    # Store navigator data for each channel in this region using integer keys
                    for channel_name, mosaic_array in channel_mosaics.items():
                        nav_channel_idx = self.navigator_channel_to_idx[channel_name]
                        print(f"Region {region_idx}:{region_name}, Channel {nav_channel_idx}:{channel_name}: shape={mosaic_array.shape}, range={mosaic_array.min()}-{mosaic_array.max()}")
                        print(f"Non-zero pixels: {np.count_nonzero(mosaic_array)}")
                        
                        # Ensure the mosaic has data
                        if np.count_nonzero(mosaic_array) == 0:
                            print(f"WARNING: Navigator mosaic for {region_name}, {channel_name} is empty! Skipping this navigator.")
                            continue
                        
                        # EFFICIENCY: Store with integer-based key for fast lookup
                        region_channel_key = (region_idx, nav_channel_idx)
                        self.region_channel_navigators[region_channel_key] = (navigator, mosaic_array, nav_metadata)
                        
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to create multi-channel navigator for region {region_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Create combined navigator layers and setup interactivity
            self._create_combined_navigator_layers()
            self._setup_navigator_interactivity()
            
            # Set up visibility change callback with FOV viewer
            if self.fov_viewer:
                self.fov_viewer.set_navigator_visibility_callback(self._on_channel_visibility_changed)
            
            # CRITICAL: Initialize navigator state to match the current viewer state
            # This ensures the navigator works immediately without requiring slider movement
            print("Initializing navigator state...")
            self._update_combined_navigator()
            self._update_navigator_box()
            
            print("Navigator state initialized successfully")
            
            print(f"Region-specific multi-channel navigator overlays created successfully!")
            print(f"Total navigators created: {len(self.region_channel_navigators)} ({len(unique_regions)} regions × {len(unique_channels)} channels)")
            
            return True
            
        except Exception as e:
            print(f"Failed to create multi-channel navigator: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_channel_colormap(self, channel_idx):
        """Get appropriate colormap for each channel based on name/wavelength - matches FOV viewer logic."""
        if channel_idx < len(self.channel_names):
            channel_name = self.channel_names[channel_idx].lower()
            
            # Map common wavelengths to colors - same as FOV viewer
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
    
    def _get_channel_number(self, channel_name):
        """Extract wavelength number from channel name."""
        try:
            # Extract wavelength patterns like 405, 488, 561, 638
            import re
            match = re.search(r'(\d{3})', channel_name)
            if match:
                return match.group(1)
            else:
                return "000"  # Default fallback
        except:
            return "000"

    def _create_combined_navigator_layers(self):
        """Create napari layers for individual channel navigator overlays with proper colors."""
        
        # Position calculation - find main FOV image layer bounds
        print("=== NAVIGATOR POSITIONING ===")
        
        # Find the main FOV image layer to get its actual bounds
        main_image_layer = None
        for layer in self.viewer.layers:
            if hasattr(layer, 'data') and 'Channel:' in layer.name:
                main_image_layer = layer
                break
        
        def get_fov_positioning_bounds():
            if main_image_layer:
                # Get the actual world coordinate bounds using napari's extent property
                extent = main_image_layer.extent
                print(f"Main layer extent: {extent}")
                
                # extent.world gives us the actual world coordinate bounds: 
                # [[min_coords], [max_coords]] for all dimensions
                world_bounds = extent.world
                print(f"World bounds: {world_bounds}")
                
                # For positioning, we need the last two dimensions (Y, X)
                # world_bounds shape is (2, ndim) where [0] is min coords, [1] is max coords
                fov_right = world_bounds[1, -1]  # Max X coordinate (right edge)
                fov_top = world_bounds[0, -2]    # Min Y coordinate (top edge)
                
                print(f"FOV right edge (X): {fov_right}")
                print(f"FOV top edge (Y): {fov_top}")
                
                return fov_right, fov_top
            else:
                print("No main image layer found, using fallback")
                # Fallback to metadata approach
                return self.dims.get('x', 5000), 0
        
        # Get initial positioning bounds (will be updated after scaling)
        fov_right, fov_top = get_fov_positioning_bounds()
        
        # Store individual channel navigator layers: {(region_idx, channel_idx): layer}
        self.nav_channel_layers = {}
        
        # Create individual channel navigator layers for each region
        for region_idx, region_name in enumerate(self.region_names):
            print(f"Creating individual channel navigator layers for region {region_idx}:{region_name}")
            
            # Get available navigator channels for this region
            available_channels = []
            for napari_channel_idx in range(len(self.channel_names)):
                nav_channel_idx = self.napari_idx_to_navigator_idx.get(napari_channel_idx, napari_channel_idx)
                region_channel_key = (region_idx, nav_channel_idx)
                
                if region_channel_key in self.region_channel_navigators:
                    _, mosaic_array, nav_metadata = self.region_channel_navigators[region_channel_key]
                    available_channels.append((napari_channel_idx, nav_channel_idx, mosaic_array))
            
            if not available_channels:
                print(f"No data available for region {region_idx}:{region_name}, skipping")
                continue
            
            # Determine initial visibility: visible only if it's the first region
            is_initial_visible = (region_idx == 0)
            
            # Create separate navigator layer for each channel
            for napari_channel_idx, nav_channel_idx, mosaic_array in available_channels:
                channel_name = self.channel_names[napari_channel_idx]
                
                print(f"Creating navigator layer for region {region_idx}:{region_name}, channel {napari_channel_idx}:{channel_name}")
                
                # Get the appropriate colormap for this channel
                colormap = self._get_channel_colormap(napari_channel_idx)
                
                # Add navigator as an overlay layer with proper color
                nav_layer = self.viewer.add_image(
                    mosaic_array,
                    name=f'_{region_name} - Fluorescence_{self._get_channel_number(channel_name)}_nm_EX',
                    opacity=1.0,  # Fully opaque
                    colormap=colormap,  # Use channel-specific colormap
                    blending='additive',  # Use additive blending like FOV channels
                    visible=is_initial_visible,  # Only first region visible initially
                    scale=(1, 1),  # Navigator has its own coordinate system
                    translate=(0, 0),  # Start at origin, we'll position it after
                    metadata={
                        'region_idx': region_idx, 
                        'region_name': region_name,
                        'channel_idx': napari_channel_idx,
                        'channel_name': channel_name,
                        'navigator_channel_idx': nav_channel_idx
                    },
                    # Set appropriate contrast limits based on actual data range
                    contrast_limits=[mosaic_array.min(), mosaic_array.max()]
                    # NO gamma correction - preserve original intensity relationships to match FOV
                )
                
                # Store the layer reference with both region and channel indices
                self.nav_channel_layers[(region_idx, napari_channel_idx)] = nav_layer
                print(f"Individual navigator layer added: {nav_layer.name}")
                
                # Position navigator just outside the right edge (same for all channels, they'll stack)
                nav_height, nav_width = mosaic_array.shape
                gap = 1
                
                nav_x_position = fov_right + gap  # RIGHT EDGE + gap
                nav_y_position = fov_top + gap    # TOP EDGE + gap
                
                nav_layer.translate = (nav_y_position, nav_x_position)
                
                # Make navigator interactive for clicking
                nav_layer.interactive = True
            
            # Add a single border around the navigator area for this region (using first channel's dimensions)
            if available_channels:
                _, _, first_mosaic = available_channels[0]
                nav_height, nav_width = first_mosaic.shape
                gap = 1
                nav_x_position = fov_right + gap
                nav_y_position = fov_top + gap
                
                nav_border_coords = [
                    [nav_y_position, nav_x_position],  # Top-left
                    [nav_y_position, nav_x_position + nav_width],  # Top-right
                    [nav_y_position + nav_height, nav_x_position + nav_width],  # Bottom-right
                    [nav_y_position + nav_height, nav_x_position]  # Bottom-left
                ]
                
                nav_border_layer = self.viewer.add_shapes(
                    [nav_border_coords],
                    shape_type='rectangle',
                    edge_color='white',
                    face_color='transparent',
                    edge_width=2,
                    name=f'_Navigator Border {region_name}',
                    opacity=1.0,
                    visible=is_initial_visible  # Only first region visible initially
                )
                
                self.nav_border_layers[region_idx] = nav_border_layer
                
                # Make the border non-interactive
                nav_border_layer.interactive = False
                nav_border_layer.editable = False
                nav_border_layer.mouse_pan = False
                nav_border_layer.mouse_zoom = False
        
        print("=== INDIVIDUAL CHANNEL NAVIGATOR LAYER CREATION COMPLETE ===")
    
    def _create_combined_mosaic_for_region(self, region_idx):
        """This method is no longer used - keeping for compatibility."""
        return None
    
    def _on_channel_visibility_changed(self):
        """Callback when channel visibility changes - update the navigator channel visibility."""
        if self._updating_navigator:
            return
        
        print("Channel visibility changed, updating navigator channel visibility...")
        self._update_navigator_visibility()
    
    def _update_combined_navigator(self):
        """Update the navigator based on current region and visible channels."""
        if self._updating_navigator:
            return
        
        self._updating_navigator = True
        
        try:
            # Get current region from viewer
            current_dims = self.viewer.dims.current_step
            if len(current_dims) >= 2:  # No channel dimension now: (t, r, f, z, y, x)
                current_region = current_dims[1]
            else:
                current_region = 0
            
            print(f"Updating navigator for region {current_region}")
            
            # Update visibility of individual channel navigator layers and borders
            for (region_idx, channel_idx), nav_layer in self.nav_channel_layers.items():
                # Layer should be visible if:
                # 1. It's the current region AND
                # 2. The corresponding FOV channel is visible
                is_current_region = (region_idx == current_region)
                is_channel_visible = False
                
                if self.fov_viewer:
                    visible_channels = self.fov_viewer.get_visible_channels()
                    is_channel_visible = channel_idx in visible_channels
                else:
                    is_channel_visible = True  # Default to visible if no FOV viewer
                
                should_be_visible = is_current_region and is_channel_visible
                nav_layer.visible = should_be_visible
            
            # Update border visibility
            for region_idx, border_layer in self.nav_border_layers.items():
                should_be_visible = (region_idx == current_region)
                border_layer.visible = should_be_visible
            
            print(f"Updated navigator visibility for region {current_region}")
        
        finally:
            self._updating_navigator = False
    
    def _update_navigator_visibility(self):
        """Update navigator visibility to match FOV channel visibility."""
        self._update_combined_navigator()  # Reuse the same logic
    
    def _setup_navigator_interactivity(self):
        """Setup navigator click handlers and dimension change callbacks."""
        
        # Create click handlers for all individual channel navigator layers
        for (region_idx, channel_idx), nav_layer in self.nav_channel_layers.items():
            # Get metadata from this specific channel
            nav_channel_idx = self.napari_idx_to_navigator_idx.get(channel_idx, channel_idx)
            region_channel_key = (region_idx, nav_channel_idx)
            
            if region_channel_key in self.region_channel_navigators:
                _, _, nav_metadata = self.region_channel_navigators[region_channel_key]
                self._create_channel_navigator_click_handler(region_idx, channel_idx, nav_layer, nav_metadata)
        
        # Setup dimension change handlers
        self._setup_dimension_change_handlers()
        
        # Setup contrast synchronization between FOV and navigator channels
        self._setup_channel_contrast_sync()
        
        print("Navigator interactivity setup complete")
    
    def _create_channel_navigator_click_handler(self, region_idx, channel_idx, nav_layer, nav_metadata):
        """Create click handler for an individual channel navigator."""
        
        # EFFICIENCY: Get names only for display purposes
        region_name = self.region_names[region_idx]
        channel_name = self.channel_names[channel_idx]
        
        @nav_layer.mouse_drag_callbacks.append
        def on_navigator_click(layer, event):
            if event.button == 1:  # Left click
                # Only respond to clicks if this navigator is currently visible
                if not nav_layer.visible:
                    return
                    
                # Get click position - this is in the viewer's coordinate system
                pos = event.position
                if len(pos) >= 2:
                    viewer_click_y, viewer_click_x = pos[-2], pos[-1]
                    
                    # Convert to navigator-local coordinates by subtracting navigator position
                    nav_y_pos, nav_x_pos = nav_layer.translate
                    nav_local_x = viewer_click_x - nav_x_pos
                    nav_local_y = viewer_click_y - nav_y_pos
                    
                    # Get navigator dimensions
                    nav_height, nav_width = nav_layer.data.shape
                    
                    # Check if click is within navigator bounds
                    if (0 <= nav_local_x < nav_width and 0 <= nav_local_y < nav_height):
                        # Convert to grid coordinates
                        tile_size = nav_metadata['tile_size']
                        grid_col = int(nav_local_x / tile_size)
                        grid_row = int(nav_local_y / tile_size)
                        
                        print(f"Navigator click in region {region_idx}:{region_name}, channel {channel_idx}:{channel_name}: viewer({viewer_click_x}, {viewer_click_y}) -> nav_local({nav_local_x}, {nav_local_y}) -> grid({grid_row}, {grid_col})")
                        print(f"Tile size: {tile_size}, Navigator dimensions: {nav_width}x{nav_height}")
                        
                        # Find the corresponding FOV in THIS region's grid
                        fov_grid = nav_metadata.get('fov_grid', {})
                        print(f"FOV grid has {len(fov_grid)} entries: {list(fov_grid.keys())[:10]}...")  # Show first 10 grid positions
                        
                        if (grid_row, grid_col) in fov_grid:
                            clicked_fov = fov_grid[(grid_row, grid_col)]
                            
                            print(f"Clicked FOV: {clicked_fov} in region {region_idx}:{region_name}")
                            
                            # EFFICIENCY: Use integer indices directly for viewer updates
                            try:
                                # Find FOV index in the main viewer
                                fov_idx = self.fov_names.index(clicked_fov) if clicked_fov in self.fov_names else None
                                
                                if fov_idx is not None:
                                    # Update viewer dimensions to jump to this FOV (no channel change)
                                    current = list(self.viewer.dims.current_step)
                                    if len(current) >= 3:  # Make sure we have enough dimensions (t, r, f, z, y, x)
                                        current[1] = region_idx  # Region dimension (position 1)
                                        current[2] = fov_idx     # FOV dimension (position 2)
                                        self.viewer.dims.current_step = current
                                        
                                        print(f"Jumped to Region: {region_idx}:{region_name}, FOV: {clicked_fov} (idx={fov_idx})")
                                else:
                                    print(f"Could not find FOV index for: {clicked_fov}")
                                    
                            except Exception as e:
                                print(f"Error in navigator click handling: {e}")
                        else:
                            print(f"No FOV found at grid position ({grid_row}, {grid_col}) in region {region_idx}:{region_name}")
                            print(f"Available grid positions: {sorted(list(fov_grid.keys()))}")
                    else:
                        print(f"Click outside navigator bounds: nav_local({nav_local_x}, {nav_local_y})")
    
    def _setup_dimension_change_handlers(self):
        """Setup handlers for dimension changes and layer selection."""
        
        # Combined function that handles both region/channel switching and box updates
        def on_dimension_change():
            if not self._updating_navigator:
                self._update_combined_navigator()
                self._update_navigator_box()
        
        # Connect the update function to dimension changes
        self.viewer.dims.events.current_step.connect(on_dimension_change)
        
        # Add specialized layer selection handler
        @self.viewer.layers.selection.events.active.connect
        def on_layer_selection_changed(event):
            # CRITICAL: Prevent recursive calls
            if self._updating_navigator:
                return
            
            selected_layer = event.value if hasattr(event, 'value') else None
            if selected_layer is None or not hasattr(selected_layer, 'name'):
                return
                
            layer_name = selected_layer.name
            
            # Handle FOV Channel layer selection - ensure proper navigator linking
            if 'Channel:' in layer_name and hasattr(selected_layer, 'metadata'):
                if 'channel_index' in selected_layer.metadata:
                    self._handle_fov_channel_selection(selected_layer)
                    return
            
            # If navigator box is selected, immediately switch back to the main data layer
            if '_Navigator Box' in layer_name:
                try:
                    self._updating_navigator = True
                    
                    # Find the main data layer (Multi-channel FOV Data) and select it instead
                    for layer in self.viewer.layers:
                        if hasattr(layer, 'name') and 'Multi-channel FOV Data' in layer.name:
                            self.viewer.layers.selection.active = layer
                            print(f"Prevented Navigator Box selection, switched back to main data layer")
                            break
                except Exception as e:
                    print(f"Error switching back to main data layer: {e}")
                finally:
                    self._updating_navigator = False
                return
            
            # For any other layer selection, do standard update
            on_dimension_change()
    
    def _handle_fov_channel_selection(self, selected_fov_layer):
        """Handle FOV channel layer selection to ensure proper navigator linking."""
        try:
            self._updating_navigator = True
            
            # Get channel index from the selected FOV layer
            channel_idx = selected_fov_layer.metadata['channel_index']
            channel_name = self.channel_names[channel_idx] if channel_idx < len(self.channel_names) else f"Channel {channel_idx}"
            
            # Get current region
            current_step = self.viewer.dims.current_step
            current_region_idx = current_step[1] if len(current_step) >= 2 else 0
            current_region_name = self.region_names[current_region_idx] if current_region_idx < len(self.region_names) else f"Region {current_region_idx}"
            
            print(f"FOV Channel layer selected: {selected_fov_layer.name} (channel {channel_idx}:{channel_name})")
            print(f"Current region: {current_region_idx}:{current_region_name}")
            
            # Find the corresponding navigator layer for this region-channel combination
            target_nav_layer = None
            nav_key = (current_region_idx, channel_idx)
            
            if nav_key in self.nav_channel_layers:
                target_nav_layer = self.nav_channel_layers[nav_key]
                print(f"Found corresponding navigator layer: {target_nav_layer.name}")
            else:
                print(f"No navigator layer found for region {current_region_idx}, channel {channel_idx}")
                print(f"Available navigator keys: {list(self.nav_channel_layers.keys())}")
                
                # Try to find navigator layer with different channel mapping
                nav_channel_idx = self.napari_idx_to_navigator_idx.get(channel_idx, channel_idx)
                alt_nav_key = (current_region_idx, nav_channel_idx)
                if alt_nav_key in self.nav_channel_layers:
                    target_nav_layer = self.nav_channel_layers[alt_nav_key]
                    print(f"Found navigator layer with mapped channel: {target_nav_layer.name} (nav_channel_idx={nav_channel_idx})")
            
            # Ensure the target navigator layer is visible and properly synced
            if target_nav_layer:
                # First, update all navigator visibility to match current state
                self._update_combined_navigator()
                
                # Then ensure the specific channel's navigator is visible if the FOV channel is visible
                if selected_fov_layer.visible:
                    target_nav_layer.visible = True
                    print(f"Ensured navigator visibility: {target_nav_layer.name} -> visible={target_nav_layer.visible}")
                    
                    # Sync contrast between FOV and navigator layers
                    self._sync_channel_contrast(channel_idx, selected_fov_layer, target_nav_layer)
                else:
                    print(f"FOV channel layer is not visible, keeping navigator hidden")
            
            # Update navigator box position
            self._update_navigator_box()
            
            print(f"Completed FOV-Navigator linking for channel {channel_idx}:{channel_name} in region {current_region_idx}:{current_region_name}")
            
        except Exception as e:
            print(f"Error in FOV channel selection handler: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._updating_navigator = False
    
    def _sync_channel_contrast(self, channel_idx, fov_layer, nav_layer):
        """Sync contrast between a specific FOV and navigator layer pair."""
        try:
            if not hasattr(self, '_updating_contrast'):
                self._updating_contrast = False
                
            if self._updating_contrast:
                return
                
            self._updating_contrast = True
            
            # Sync FOV contrast to navigator
            fov_contrast = fov_layer.contrast_limits
            fov_data_range = self._get_layer_data_range(fov_layer)
            nav_data_range = self._get_layer_data_range(nav_layer)
            
            synced_contrast = self._convert_contrast_proportionally(
                fov_contrast, fov_data_range, nav_data_range
            )
            
            nav_layer.contrast_limits = synced_contrast
            
            channel_name = self.channel_names[channel_idx] if channel_idx < len(self.channel_names) else f"Channel {channel_idx}"
            print(f"Synced contrast for {channel_name}: FOV {fov_contrast} -> Navigator {synced_contrast}")
            
        except Exception as e:
            print(f"Error syncing channel contrast: {e}")
        finally:
            self._updating_contrast = False

    def _setup_channel_contrast_sync(self):
        """Setup bidirectional contrast synchronization between FOV and navigator channel pairs."""
        
        print("Setting up channel contrast synchronization...")
        
        # Flag to prevent recursive contrast updates
        self._updating_contrast = False
        
        # Get all FOV channel layers
        fov_channel_layers = {}  # {channel_idx: layer}
        for layer in self.viewer.layers:
            if hasattr(layer, 'name') and 'Channel:' in layer.name:
                # Extract channel index from metadata
                if hasattr(layer, 'metadata') and 'channel_index' in layer.metadata:
                    channel_idx = layer.metadata['channel_index']
                    fov_channel_layers[channel_idx] = layer
                    print(f"Found FOV channel layer: {layer.name} (channel {channel_idx})")
        
        if not fov_channel_layers:
            print("Warning: No FOV channel layers found for contrast sync")
            return
        
        # Set up bidirectional sync for each channel
        sync_pairs = 0
        for channel_idx, fov_layer in fov_channel_layers.items():
            # Find all navigator layers for this channel (across all regions)
            nav_layers_for_channel = []
            for (region_idx, nav_channel_idx), nav_layer in self.nav_channel_layers.items():
                if nav_channel_idx == channel_idx:
                    nav_layers_for_channel.append((region_idx, nav_layer))
            
            if nav_layers_for_channel:
                # Set up FOV -> Navigator sync
                self._setup_fov_to_navigator_sync(channel_idx, fov_layer, nav_layers_for_channel)
                
                # Set up Navigator -> FOV sync for each navigator layer
                for region_idx, nav_layer in nav_layers_for_channel:
                    self._setup_navigator_to_fov_sync(channel_idx, region_idx, nav_layer, fov_layer)
                
                sync_pairs += 1
                channel_name = self.channel_names[channel_idx] if channel_idx < len(self.channel_names) else f"Channel {channel_idx}"
                print(f"Set up contrast sync for {channel_name}: 1 FOV layer ↔ {len(nav_layers_for_channel)} navigator layers")
        
        print(f"Channel contrast synchronization setup complete! ({sync_pairs} channels)")
    
    def _setup_fov_to_navigator_sync(self, channel_idx, fov_layer, nav_layers_for_channel):
        """Setup sync from FOV layer to all navigator layers for this channel."""
        
        @fov_layer.events.contrast_limits.connect
        def on_fov_contrast_change(event):
            if self._updating_contrast:
                return
                
            try:
                self._updating_contrast = True
                new_fov_contrast = fov_layer.contrast_limits
                
                # Get current region to determine which navigator should be visible
                current_step = self.viewer.dims.current_step
                current_region = current_step[1] if len(current_step) >= 2 else 0
                
                # Sync to all navigator layers for this channel, but only update visible ones
                for region_idx, nav_layer in nav_layers_for_channel:
                    if nav_layer.visible and region_idx == current_region:
                        # Calculate proportional contrast based on data ranges
                        fov_data_range = self._get_layer_data_range(fov_layer)
                        nav_data_range = self._get_layer_data_range(nav_layer)
                        
                        synced_contrast = self._convert_contrast_proportionally(
                            new_fov_contrast, fov_data_range, nav_data_range
                        )
                        
                        nav_layer.contrast_limits = synced_contrast
                        
                        channel_name = self.channel_names[channel_idx] if channel_idx < len(self.channel_names) else f"Channel {channel_idx}"
                        region_name = self.region_names[region_idx] if region_idx < len(self.region_names) else f"Region {region_idx}"
                        print(f"Synced FOV→Navigator: {channel_name} in {region_name}: {new_fov_contrast} → {synced_contrast}")
                
            except Exception as e:
                print(f"Error in FOV→Navigator sync: {e}")
            finally:
                self._updating_contrast = False
    
    def _setup_navigator_to_fov_sync(self, channel_idx, region_idx, nav_layer, fov_layer):
        """Setup sync from navigator layer to FOV layer."""
        
        @nav_layer.events.contrast_limits.connect
        def on_navigator_contrast_change(event):
            if self._updating_contrast:
                return
                
            # Only sync if this navigator is currently visible
            if not nav_layer.visible:
                return
                
            try:
                self._updating_contrast = True
                new_nav_contrast = nav_layer.contrast_limits
                
                # Calculate proportional contrast based on data ranges
                nav_data_range = self._get_layer_data_range(nav_layer)
                fov_data_range = self._get_layer_data_range(fov_layer)
                
                synced_contrast = self._convert_contrast_proportionally(
                    new_nav_contrast, nav_data_range, fov_data_range
                )
                
                fov_layer.contrast_limits = synced_contrast
                
                channel_name = self.channel_names[channel_idx] if channel_idx < len(self.channel_names) else f"Channel {channel_idx}"
                region_name = self.region_names[region_idx] if region_idx < len(self.region_names) else f"Region {region_idx}"
                print(f"Synced Navigator→FOV: {channel_name} in {region_name}: {new_nav_contrast} → {synced_contrast}")
                
            except Exception as e:
                print(f"Error in Navigator→FOV sync: {e}")
            finally:
                self._updating_contrast = False
    
    def _get_layer_data_range(self, layer):
        """Get the full data range (min, max) for a layer."""
        try:
            if hasattr(layer, 'data'):
                data = layer.data
                # For dask arrays, compute min/max efficiently
                if hasattr(data, 'compute'):
                    # Sample a subset for performance
                    sample = data[..., ::10, ::10] if data.ndim >= 2 else data
                    sample_computed = sample.compute()
                    return (float(sample_computed.min()), float(sample_computed.max()))
                else:
                    return (float(data.min()), float(data.max()))
        except Exception as e:
            print(f"Error getting data range for layer {layer.name}: {e}")
            # Fallback to current contrast limits
            return layer.contrast_limits
    
    def _convert_contrast_proportionally(self, source_contrast, source_range, target_range):
        """Convert contrast limits proportionally between different data ranges."""
        
        source_min, source_max = source_range
        target_min, target_max = target_range
        contrast_min, contrast_max = source_contrast
        
        # Calculate proportions within source range
        source_full_range = source_max - source_min
        if source_full_range == 0:
            return target_range  # Avoid division by zero
            
        min_proportion = (contrast_min - source_min) / source_full_range
        max_proportion = (contrast_max - source_min) / source_full_range
        
        # Apply proportions to target range
        target_full_range = target_max - target_min
        new_min = target_min + (min_proportion * target_full_range)
        new_max = target_min + (max_proportion * target_full_range)
        
        return [new_min, new_max]

    def _update_navigator_box(self):
        """Update navigator box to show current FOV position."""
        
        # CRITICAL: Prevent recursive calls
        if self._updating_navigator:
            return
            
        try:
            self._updating_navigator = True
            
            # Get current FOV, region from sliders (no channel dimension now)
            current_step = self.viewer.dims.current_step
            if len(current_step) >= 3:  # Updated for new dimension order: (t, r, f, z, y, x)
                region_idx = current_step[1]   # Region is now at position 1  
                fov_idx = current_step[2]      # FOV is now at position 2
                
                # Get the actual region and fov names (only for display)
                if region_idx < len(self.region_names) and fov_idx < len(self.fov_names):
                    current_region = self.region_names[region_idx]
                    current_fov = self.fov_names[fov_idx]
                    
                    print(f"Updating navigator box for region {region_idx}:{current_region}, FOV {current_fov}")
                    
                    # Remove ALL existing navigator boxes (clean slate)
                    boxes_to_remove = []
                    for layer in self.viewer.layers:
                        if hasattr(layer, 'name') and '_Navigator Box' in layer.name:
                            boxes_to_remove.append(layer)
                    
                    for box_layer in boxes_to_remove:
                        try:
                            self.viewer.layers.remove(box_layer)
                            print(f"Removed old navigator box: {box_layer.name}")
                        except:
                            pass
                    
                    # Clear the tracking dictionary
                    self.nav_box_layers.clear()
                    
                    # Only show box if we have navigator channels for the current region AND at least one is visible
                    has_visible_navigator = False
                    for (r_idx, c_idx), nav_layer in self.nav_channel_layers.items():
                        if r_idx == region_idx and nav_layer.visible:
                            has_visible_navigator = True
                            break
                    
                    if has_visible_navigator:
                        
                        # Get metadata from any available channel for this region
                        nav_metadata = None
                        for nav_channel_idx in range(len(self.navigator_channel_names or [])):
                            region_channel_key = (region_idx, nav_channel_idx)
                            if region_channel_key in self.region_channel_navigators:
                                _, _, nav_metadata = self.region_channel_navigators[region_channel_key]
                                break
                        
                        if nav_metadata:
                            # Get any visible navigator layer for this region for positioning
                            nav_layer = None
                            for (r_idx, c_idx), layer in self.nav_channel_layers.items():
                                if r_idx == region_idx and layer.visible:
                                    nav_layer = layer
                                    break
                            
                            if not nav_layer:
                                print(f"No visible navigator layer found for region {region_idx}:{current_region}")
                                return
                            
                            # Check if the current FOV belongs to this region
                            region_fovs = list(nav_metadata.get('coordinates', {}).keys())
                            if current_fov not in region_fovs:
                                print(f"FOV {current_fov} does not belong to region {current_region} (has {len(region_fovs)} FOVs), skipping navigator box")
                                return
                            
                            # Find this FOV in the current region's navigator grid
                            fov_to_grid = nav_metadata.get('fov_to_grid_pos', {})
                            
                            if current_fov in fov_to_grid:
                                grid_row, grid_col = fov_to_grid[current_fov]
                                tile_size = nav_metadata['tile_size']
                                
                                print(f"FOV {current_fov} found at grid position ({grid_row}, {grid_col}) in region {region_idx}:{current_region}")
                                
                                # Calculate position in viewer coordinates
                                nav_local_x = grid_col * tile_size + tile_size // 2
                                nav_local_y = grid_row * tile_size + tile_size // 2
                                
                                # Convert to viewer coordinates by adding navigator position
                                nav_y_pos, nav_x_pos = nav_layer.translate
                                viewer_x = nav_local_x + nav_x_pos
                                viewer_y = nav_local_y + nav_y_pos
                                
                                # Create a rectangle shape for the box
                                box_size = tile_size * 0.8
                                box_coords = [
                                    [viewer_y - box_size//2, viewer_x - box_size//2],
                                    [viewer_y - box_size//2, viewer_x + box_size//2],
                                    [viewer_y + box_size//2, viewer_x + box_size//2],
                                    [viewer_y + box_size//2, viewer_x - box_size//2]
                                ]
                                
                                # CRITICAL: Add new box WITHOUT automatic selection to prevent recursion
                                nav_box_layer = self.viewer.add_shapes(
                                    [box_coords],
                                    shape_type='rectangle',
                                    edge_color='red',
                                    face_color='transparent',
                                    edge_width=3,
                                    name=f'_Navigator Box {current_region}',
                                    opacity=1.0
                                )
                                
                                self.nav_box_layers[region_idx] = nav_box_layer
                                
                                # Make the red box non-interactive to prevent selection
                                nav_box_layer.interactive = False
                                nav_box_layer.visible = True
                                nav_box_layer.editable = False
                                nav_box_layer.mouse_pan = False
                                nav_box_layer.mouse_zoom = False
                                
                                # CRITICAL: Immediately deselect the box to prevent selection events
                                if self.viewer.layers.selection.active == nav_box_layer:
                                    try:
                                        # Find and select any visible navigator layer for this region instead
                                        for (r_idx, c_idx), layer in self.nav_channel_layers.items():
                                            if r_idx == region_idx and layer.visible:
                                                self.viewer.layers.selection.active = layer
                                                break
                                    except:
                                        pass
                                
                                print(f"Created navigator box for region {region_idx}:{current_region}, FOV {current_fov} at grid({grid_row}, {grid_col}) -> viewer({viewer_x}, {viewer_y})")
                            else:
                                print(f"FOV {current_fov} not found in region {region_idx}:{current_region} navigator grid mapping")
                        else:
                            print(f"No metadata available for region {region_idx}:{current_region}")
                    else:
                        print(f"No visible navigator for region {region_idx}:{current_region}, skipping navigator box")
                            
        except Exception as e:
            print(f"Error updating navigator box: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._updating_navigator = False

def add_navigator_to_viewer(viewer, metadata, directory, fov_viewer=None):
    """
    Convenience function to add navigator overlay to an existing napari viewer.
    
    Parameters:
    -----------
    viewer : napari.Viewer
        The napari viewer instance
    metadata : dict
        Metadata dictionary
    directory : str
        Path to acquisition directory
    fov_viewer : FOVViewer, optional
        The FOV viewer instance for getting visible channels
        
    Returns:
    --------
    NavigatorOverlay: The navigator overlay instance, or None if not available
    """
    navigator = NavigatorOverlay(viewer, metadata, directory, fov_viewer)
    
    if navigator.create_navigators():
        return navigator
    else:
        return None 