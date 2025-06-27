import os
import re
import json
import pandas as pd
import numpy as np
import zarr
from glob import glob
import xml.etree.ElementTree as ET
import napari
import sys
import subprocess
import tempfile
import pickle
from PyQt6.QtWidgets import QApplication, QFileDialog, QDialog, QVBoxLayout, QPushButton, QLabel
from PyQt6.QtCore import QObject, pyqtSignal, QThread, Qt

# Import the downsampler module
try:
    from downsampler import DownsampledNavigator
    NAVIGATOR_AVAILABLE = True
    print("Successfully imported downsampler module")
except ImportError as e:
    print(f"Warning: downsampler module not found: {e}")
    NAVIGATOR_AVAILABLE = False

def create_tiff_zarr_map(input_dir, coordinate_csv=None, acquisition_params_json=None, configurations_xml=None):
    """
    Create a virtual mapping of TIFF files to a Zarr store structure without loading data into RAM.
    Returns a dictionary with metadata and a file map.
    
    Parameters:
    -----------
    input_dir : str
        Path to the directory containing TIFF files and time point folders
    coordinate_csv : str, optional
        Path to the CSV file containing region, fov, and coordinates information
        If None, will look for coordinates.csv in the input directory
    acquisition_params_json : str, optional
        Path to the JSON file containing acquisition parameters
        If None, will look for acquisition parameters.json in the input directory
    configurations_xml : str, optional
        Path to the XML file containing channel configurations
        If None, will look for configurations.xml in the input directory
        
    Returns:
    --------
    dict: Metadata and file mapping information
    """
    # Find and load files if not specified
    coordinate_csv = os.path.join(input_dir,"0", "coordinates.csv")
    acquisition_params_json = os.path.join(input_dir, "acquisition parameters.json")
    configurations_xml = os.path.join(input_dir, "configurations.xml")
    
    # Load coordinates from CSV
    if os.path.exists(coordinate_csv):
        coordinates = pd.read_csv(coordinate_csv)
        print(f"Loaded coordinates from {coordinate_csv}")
    else:
        coordinates = None
        print("Warning: Coordinates CSV not found")
    
    # Load acquisition parameters
    if os.path.exists(acquisition_params_json):
        with open(acquisition_params_json, 'r') as f:
            acq_params = json.load(f)
        num_timepoints = acq_params.get('Nt', 1)
        num_z = acq_params.get('Nz', 1)
        print(f"Using Nt={num_timepoints}, Nz={num_z} from acquisition parameters")
    else:
        acq_params = {}
        num_timepoints = None
        num_z = None
        print("Warning: Acquisition parameters JSON not found")
    
    # Load selected channels from configurations XML
    selected_channels = []
    channel_info = {}
    
    if os.path.exists(configurations_xml):
        try:
            tree = ET.parse(configurations_xml)
            root = tree.getroot()
            
            for mode in root.findall('.//mode'):
                name = mode.get('Name', '')
                selected = mode.get('Selected', 'false').lower() == 'true'
                
                if 'Fluorescence' in name and 'nm Ex' in name and selected:
                    # Extract wavelength from name (e.g., "Fluorescence 488 nm Ex")
                    wavelength_match = re.search(r'(\d+)\s*nm', name)
                    if wavelength_match:
                        wavelength = wavelength_match.group(1)
                        channel_name = f"Fluorescence_{wavelength}_nm_Ex"
                        selected_channels.append(channel_name)
                        
                        channel_info[channel_name] = {
                            'id': mode.get('ID'),
                            'name': name,
                            'wavelength': wavelength,
                            'exposure': float(mode.get('ExposureTime', 0)),
                            'intensity': float(mode.get('IlluminationIntensity', 0))
                        }
            
            print(f"Selected channels from XML: {selected_channels}")
        except Exception as e:
            print(f"Warning: Error parsing configurations XML: {e}")
    else:
        print("Warning: Configurations XML not found")
    
    # Get all time point directories or use the input directory directly
    if os.path.isdir(os.path.join(input_dir, '0')):  # Check if time point directories exist
        timepoint_dirs = sorted([d for d in os.listdir(input_dir) 
                                if os.path.isdir(os.path.join(input_dir, d)) and d.isdigit()],
                                key=lambda x: int(x))
        timepoint_dirs = [os.path.join(input_dir, d) for d in timepoint_dirs]
    else:
        # If no timepoint directories, assume the input directory is a single timepoint
        timepoint_dirs = [input_dir]
        if num_timepoints is None:
            num_timepoints = 1
    
    # Get the actual number of timepoints based on available directories
    if num_timepoints is None:
        num_timepoints = len(timepoint_dirs)
    else:
        num_timepoints = min(num_timepoints, len(timepoint_dirs))
    
    # Process first timepoint to discover dimensions
    first_tp_files = glob(os.path.join(timepoint_dirs[0], "*.tif*"))
    
    if not first_tp_files:
        raise ValueError(f"No TIFF files found in {timepoint_dirs[0]}")
    
    # Extract pattern from filenames
    # Example: C5_0_0_Fluorescence_488_nm_Ex.tiff
    pattern = r'([^_]+)_(\d+)_(\d+)_(.+)\.tiff?'
    
    # Get unique regions, FOVs, and validate z levels from filenames
    unique_regions = set()
    unique_fovs = set()
    z_levels = set()
    found_channels = set()
    
    for file_path in first_tp_files:
        filename = os.path.basename(file_path)
        match = re.match(pattern, filename)
        if match:
            region, fov, z_level, channel_name = match.groups()
            unique_regions.add(region)
            unique_fovs.add(int(fov))
            z_levels.add(int(z_level))
            found_channels.add(channel_name)
    
    # Convert sets to sorted lists
    unique_regions = sorted(list(unique_regions))
    unique_fovs = sorted(list(unique_fovs))
    z_levels = sorted(list(z_levels))
    
    # If Nz not provided in parameters, infer from z levels in files
    if num_z is None:
        num_z = max(z_levels) + 1
        print(f"Inferring Nz={num_z} from file z levels")
    
    # Use selected channels from XML if available, otherwise use all found channels
    if selected_channels:
        # Filter to only include channels that actually exist in the files
        channels_to_use = [ch for ch in selected_channels if any(ch in fc for fc in found_channels)]
        if not channels_to_use:
            print("Warning: None of the selected channels from XML match the files. Using all found channels.")
            channels_to_use = sorted(list(found_channels))
    else:
        channels_to_use = sorted(list(found_channels))
    
    print(f"Using channels: {channels_to_use}")
    
    # Map channel names to indices
    channel_map = {name: idx for idx, name in enumerate(channels_to_use)}
    
    # Create a file lookup dictionary to map coordinates to files
    file_map = {}
    
    # Collect all TIFF files and organize them in the map
    for t_idx, tp_dir in enumerate(timepoint_dirs[:num_timepoints]):
        tiff_files = glob(os.path.join(tp_dir, "*.tif*"))
        for tiff_file in tiff_files:
            filename = os.path.basename(tiff_file)
            match = re.match(pattern, filename)
            if match:
                region, fov, z_level, full_channel_name = match.groups()
                z_level = int(z_level)
                fov = int(fov)
                
                # Find the channel from the ones we're using
                channel_name = None
                for ch in channels_to_use:
                    if ch in full_channel_name:
                        channel_name = ch
                        break
                
                # Skip if channel not in our list or z level out of range
                if channel_name is None or z_level >= num_z:
                    continue
                
                # Find indices in the array
                region_idx = unique_regions.index(region) if region in unique_regions else None
                fov_idx = unique_fovs.index(fov) if fov in unique_fovs else None
                channel_idx = channel_map.get(channel_name)
                
                # Skip if any index not found
                if region_idx is None or fov_idx is None or channel_idx is None:
                    continue
                
                # Store file path in map
                key = (t_idx, region_idx, fov_idx, z_level, channel_idx)
                file_map[key] = tiff_file
    
    # Need to determine image dimensions to complete metadata
    if file_map:
        # Sample a file to get image dimensions
        sample_file = next(iter(file_map.values()))
        try:
            import tifffile
            sample_img = tifffile.imread(sample_file)
            y_size, x_size = sample_img.shape
        except Exception as e:
            print(f"Warning: Could not read sample file to determine dimensions: {e}")
            y_size, x_size = None, None
    else:
        y_size, x_size = None, None
    
    # Create coordinate arrays for metadata
    time_array = list(range(num_timepoints))
    region_array = unique_regions
    fov_array = unique_fovs
    z_array = list(range(num_z))
    channel_array = channels_to_use
    
    # Create dictionary with all the dimension information
    dimensions = {
        'time': num_timepoints,
        'region': len(unique_regions),
        'fov': len(unique_fovs),
        'z': num_z,
        'channel': len(channels_to_use),
        'y': y_size,
        'x': x_size
    }
    
    # Create coordinates information
    coords_info = {}
    if coordinates is not None:
        for col in coordinates.columns:
            coords_info[col] = coordinates[col].tolist()
    
    # Build the final metadata package
    metadata = {
        'file_map': file_map,
        'dimensions': dimensions,
        'regions': unique_regions,
        'fovs': unique_fovs,
        'channels': channels_to_use,
        'channel_info': channel_info,
        'acquisition_parameters': acq_params,
        'coordinates': coords_info,
        'dimension_arrays': {
            'time': time_array,
            'region': region_array,
            'fov': fov_array,
            'z': z_array,
            'channel': channel_array
        }
    }
    
    print(f"Created mapping with dimensions: {dimensions}")
    print(f"Mapped {len(file_map)} files")
    
    return metadata

def get_zarr_store_with_lazy_tiff_mapping(input_dir):
    """
    Get metadata and file mapping for TIFF files.
    
    Parameters:
    -----------
    Same as create_tiff_zarr_map
    
    Returns:
    --------
    dict: Comprehensive metadata and file mapping information
    """
    # Create the mapping
    metadata = create_tiff_zarr_map(input_dir)
    
    # For compatibility with previous versions, return a tuple
    return metadata

class Worker(QThread):
    finished = pyqtSignal(str, str)
    error = pyqtSignal(str, str)
    
    def __init__(self, directory):
        super().__init__()
        self.directory = directory
        
    def run(self):
        try:
            folder_name = os.path.basename(os.path.normpath(self.directory))
            
            # Get metadata for the selected directory
            metadata = get_zarr_store_with_lazy_tiff_mapping(self.directory)
            
            # Save the metadata to a temporary file
            temp_dir = tempfile.gettempdir()
            temp_file = os.path.join(temp_dir, f"ndv_metadata_{os.getpid()}_{hash(folder_name)}.pkl")
            
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'directory': self.directory,
                    'metadata': metadata,
                    'folder_name': folder_name
                }, f)
            
            # Emit signal with the temp file path
            self.finished.emit(temp_file, folder_name)
            
        except Exception as e:
            import traceback
            print(f"Error processing acquisition: {str(e)}")
            print(traceback.format_exc())
            self.error.emit(str(e), folder_name)

class DirectorySelector(QDialog):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Select Acquisition Directory')
        self.setGeometry(300, 300, 400, 150)
        
        layout = QVBoxLayout()
        
        self.label = QLabel('Drag and drop your acquisition directory here')
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                border: 2px dashed #aaa;
                border-radius: 5px;
                padding: 20px;
                background-color: #f0f0f0;
            }
        """)
        layout.addWidget(self.label)
        
        self.setLayout(layout)
        
        # Enable drag and drop
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            directory = urls[0].toLocalFile()
            if os.path.isdir(directory):
                folder_name = os.path.basename(os.path.normpath(directory))
                self.label.setText(f'Loading: {folder_name}')
                
                # Create worker thread
                self.worker = Worker(directory)
                self.worker.finished.connect(self.launch_ndv)
                self.worker.error.connect(self.handle_error)
                self.worker.start()
    
    def launch_ndv(self, temp_file, folder_name):
        self.label.setText(f'Opened: {folder_name}')
        
        # Create a separate Python script to launch napari
        launcher_script = os.path.join(tempfile.gettempdir(), f"napari_launcher_{os.getpid()}_{hash(folder_name)}.py")
        
        with open(launcher_script, 'w') as f:
            f.write(f"""
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

# Add the directory containing downsampler to Python path
sys.path.insert(0, r'{os.path.dirname(os.path.abspath(__file__))}')

# Try to import downsampler
try:
    from downsampler import DownsampledNavigator
    NAVIGATOR_AVAILABLE = True
    print("Successfully imported downsampler module in launcher")
except ImportError as e:
    print(f"Warning: Navigator module not available in launcher: {{e}}")
    NAVIGATOR_AVAILABLE = False

# Load the metadata from the temp file
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

directory = data['directory']
metadata = data['metadata']
folder_name = data['folder_name']

print(f"Launching napari for: {{folder_name}}")
print(f"Directory: {{directory}}")
print(f"Navigator available: {{NAVIGATOR_AVAILABLE}}")

# Get dimensions
dims = metadata['dimensions']
file_map = metadata['file_map']
acq_params = metadata['acquisition_parameters']

print(f"Dimensions: {{dims}}")

# Calculate pixel size in microns
pixel_size_um = None
z_step_um = None

if acq_params:
    # Get pixel size from sensor pixel size and objective magnification
    if 'sensor_pixel_size_um' in acq_params and 'objective' in acq_params and 'magnification' in acq_params['objective']:
        sensor_pixel_size = acq_params['sensor_pixel_size_um']
        magnification = acq_params['objective']['magnification']
        pixel_size_um = sensor_pixel_size / magnification
        print(f"Calculated pixel size: {{pixel_size_um:.3f}} µm")
    
    # Get Z step size
    if 'dz(um)' in acq_params:
        z_step_um = acq_params['dz(um)']
        print(f"Z step size: {{z_step_um}} µm")

# Default values if not found in parameters
if pixel_size_um is None:
    pixel_size_um = 1.0
    print("Warning: Using default pixel size of 1.0 µm")
    
if z_step_um is None:
    z_step_um = 1.0
    print("Warning: Using default Z step size of 1.0 µm")

# Create a function that loads TIFF files on demand
@dask.delayed
def load_tiff(t, r, f, z, c):
    key = (t, r, f, z, c)
    if key in file_map:
        return tifffile.imread(file_map[key])
    else:
        return np.zeros((dims['y'], dims['x']), dtype=np.uint16)

# Create a dask array with a delayed loader function
lazy_arrays = []
for t in range(dims['time']):
    channel_arrays = []
    for c in range(dims['channel']):
        region_arrays = []
        for r in range(dims['region']):
            fov_arrays = []
            for f in range(dims['fov']):
                z_arrays = []
                for z in range(dims['z']):
                    # Create a delayed reader for each position
                    delayed_reader = load_tiff(t, r, f, z, c)
                    # Convert to a dask array
                    sample_shape = (dims['y'], dims['x'])
                    lazy_array = da.from_delayed(delayed_reader, shape=sample_shape, dtype=np.uint16)
                    z_arrays.append(lazy_array)
                fov_arrays.append(da.stack(z_arrays))
            region_arrays.append(da.stack(fov_arrays))
        channel_arrays.append(da.stack(region_arrays))
    lazy_arrays.append(da.stack(channel_arrays))

# Stack everything into a single dask array
dask_data = da.stack(lazy_arrays)

# Display the data using napari
print(f"Opening napari viewer for: {{folder_name}}")
viewer = napari.Viewer(title=f"Napari - {{folder_name}}")

# The dimensions are: (t, c, r, f, z, y, x)
channel_names = metadata['channels']
region_names = metadata['regions']
fov_names = metadata['fovs']

print(f"Channel names: {{channel_names}}")
print(f"Region names: {{region_names}}")
print(f"FOV names: {{fov_names}}")

# Calculate scale for each dimension
# [time, region, fov, z, y, x]
# For time, region, and fov, we use 1.0 as they're indices
# For z, y, x we use the physical dimensions
scale = [1.0, 1.0, 1.0, z_step_um, pixel_size_um, pixel_size_um]

# Store raw image dimensions for later napari coordinate comparison
raw_fov_width = dims['x']
raw_fov_height = dims['y']
raw_fov_area = raw_fov_width * raw_fov_height

print("="*60)
print("RAW IMAGE DIMENSIONS (for reference)")
print("="*60)
print(f"Raw FOV image: {{raw_fov_width}} x {{raw_fov_height}} pixels ({{raw_fov_area:,}} pixels)")
print(f"Raw navigator tile: 75 x 75 pixels (5,625 pixels)")
print(f"Raw area ratio: {{raw_fov_area/5625:.1f}}:1")
print("="*60)

# Add each channel as a separate layer
for c in range(dims['channel']):
    # For each channel, we keep all other dimensions
    # This creates a 6D array: (t, r, f, z, y, x)
    channel_data = dask_data[:, c, :, :, :, :, :]
    
    # Create a meaningful name for the layer
    layer_name = f"Channel: {{channel_names[c]}}"
    
    # Add to viewer with appropriate dimension labels and scale
    viewer.add_image(
        channel_data,
        name=layer_name,
        scale=scale,  # Apply physical scale
        blending='additive',
        colormap='gray',
        # Define dimension names for sliders
        # The order should match the dimensions in the array
        multiscale=False,
        metadata={{
            'dimension_names': ['Time', 'Region', 'FOV', 'Z', 'Y', 'X']
        }}
    )

# Set dimension labels for the sliders
viewer.dims.axis_labels = ['Time', 'Region', 'FOV', 'Z', 'Y', 'X']

# Add physical units to the dimension labels
if z_step_um is not None and pixel_size_um is not None:
    viewer.scale_bar.unit = 'µm'
    viewer.scale_bar.visible = True

# ===== ADD REGION-SPECIFIC NAVIGATOR FUNCTIONALITY =====
if NAVIGATOR_AVAILABLE and dims['region'] > 0 and dims['fov'] > 0:
    print("Creating region-specific navigator overlays...")
    
    # Storage for navigators and layers per region
    region_navigators = {{}}  # {{region_name: (navigator, mosaic_array, metadata)}}
    nav_layers = {{}}         # {{region_name: layer}}
    nav_border_layers = {{}}  # {{region_name: border_layer}}
    nav_box_layers = {{}}     # {{region_name: box_layer}}
    
    try:
        # Create navigator with progress callback
        def progress_callback(percent, message):
            print(f"Navigator: {{percent}}% - {{message}}")
        
        # Get unique regions from metadata
        unique_regions = region_names  # This is the list of region names
        print(f"Creating navigators for regions: {{unique_regions}}")
        print(f"Total FOVs in dataset: {{len(fov_names)}}")
        print(f"Sample FOV names: {{fov_names[:10]}}")  # Show first 10 FOVs
        
        # Create navigator instance once
        navigator = DownsampledNavigator(
            Path(directory), 
            tile_size=75,  # Good balance of quality and speed
            cache_enabled=True,
            progress_callback=progress_callback
        )
        
        # Create mosaics for each region
        current_time = viewer.dims.current_step[0] if viewer.dims.current_step else 0
        
        for region_name in unique_regions:
            print(f"Creating navigator mosaic for region '{{region_name}}' at timepoint {{current_time}}")
            
            try:
                mosaic_array, nav_metadata = navigator.create_mosaic_for_region(region_name, current_time)
                
                print(f"Navigator mosaic created for {{region_name}}: shape={{mosaic_array.shape}}, range={{mosaic_array.min()}}-{{mosaic_array.max()}}")
                print(f"Non-zero pixels: {{np.count_nonzero(mosaic_array)}}")
                
                # Debug: Show which FOVs belong to this region
                region_fovs = list(nav_metadata.get('coordinates', {{}}).keys())
                print(f"Region {{region_name}} has {{len(region_fovs)}} FOVs: {{region_fovs[:5]}}..." if len(region_fovs) > 5 else f"Region {{region_name}} has FOVs: {{region_fovs}}")
                
                # Store the navigator data
                region_navigators[region_name] = (navigator, mosaic_array, nav_metadata)
                
                # Ensure the mosaic has data
                if np.count_nonzero(mosaic_array) == 0:
                    print(f"WARNING: Navigator mosaic for {{region_name}} is empty! Skipping this region's navigator.")
                    continue
                    
            except Exception as e:
                print(f"Failed to create navigator for region {{region_name}}: {{e}}")
                continue
        
        # Position calculation - find main FOV image layer bounds
        print("=== NAVIGATOR POSITIONING ===")
        
        # Find the main FOV image layer to get its actual bounds
        main_image_layer = None
        for layer in viewer.layers:
            if hasattr(layer, 'data') and 'Channel:' in layer.name:
                main_image_layer = layer
                break
        
        def get_fov_positioning_bounds():
            if main_image_layer:
                # Get the actual world coordinate bounds using napari's extent property
                extent = main_image_layer.extent
                print(f"Main layer extent: {{extent}}")
                
                # extent.world gives us the actual world coordinate bounds: 
                # [[min_coords], [max_coords]] for all dimensions
                world_bounds = extent.world
                print(f"World bounds: {{world_bounds}}")
                
                # For positioning, we need the last two dimensions (Y, X)
                # world_bounds shape is (2, ndim) where [0] is min coords, [1] is max coords
                fov_right = world_bounds[1, -1]  # Max X coordinate (right edge)
                fov_top = world_bounds[0, -2]    # Min Y coordinate (top edge)
                
                print(f"FOV right edge (X): {{fov_right}}")
                print(f"FOV top edge (Y): {{fov_top}}")
                
                return fov_right, fov_top
            else:
                print("No main image layer found, using fallback")
                # Fallback to metadata approach
                return dims.get('x', 5000), 0
        
        # Get initial positioning bounds (will be updated after scaling)
        fov_right, fov_top = get_fov_positioning_bounds()
        
        # Create layers for each region navigator (initially all hidden except first)
        for i, (region_name, (nav_obj, mosaic_array, nav_metadata)) in enumerate(region_navigators.items()):
            print(f"Creating layer for region {{region_name}}")
            
            # Add navigator as an overlay layer
            nav_layer = viewer.add_image(
                mosaic_array,
                name=f'Navigator {{region_name}}',
                opacity=1.0,  # Fully opaque
                colormap='gray',
                visible=(i == 0),  # Only first region visible initially
                scale=(1, 1),  # Navigator has its own coordinate system
                translate=(0, 0),  # Start at origin, we'll position it after
                metadata=nav_metadata,
                contrast_limits=(0, 255),  # Full contrast range
                gamma=0.8  # Lower gamma for brighter appearance
            )
            
            nav_layers[region_name] = nav_layer
            print(f"Navigator layer added for {{region_name}}: {{nav_layer.name}}")
            
            # Position navigator just outside the right edge
            nav_height, nav_width = mosaic_array.shape
            gap = 1
            
            nav_x_position = fov_right + gap  # RIGHT EDGE + gap
            nav_y_position = fov_top + gap    # TOP EDGE + gap
            
            print(f"Navigator position for {{region_name}}: x={{nav_x_position}}, y={{nav_y_position}}")
            print(f"Navigator raw dimensions: {{nav_width}} x {{nav_height}} pixels")
            
            nav_layer.translate = (nav_y_position, nav_x_position)
            
            # Add a simple bounding box around the navigator
            nav_border_coords = [
                [nav_y_position, nav_x_position],  # Top-left
                [nav_y_position, nav_x_position + nav_width],  # Top-right
                [nav_y_position + nav_height, nav_x_position + nav_width],  # Bottom-right
                [nav_y_position + nav_height, nav_x_position]  # Bottom-left
            ]
            
            nav_border_layer = viewer.add_shapes(
                [nav_border_coords],
                shape_type='rectangle',
                edge_color='white',
                face_color='transparent',
                edge_width=2,
                name=f'Navigator Border {{region_name}}',
                opacity=1.0,
                visible=(i == 0)  # Only first region visible initially
            )
            
            nav_border_layers[region_name] = nav_border_layer
            
            # Make the border non-interactive
            nav_border_layer.interactive = False
            nav_border_layer.editable = False
            nav_border_layer.mouse_pan = False
            nav_border_layer.mouse_zoom = False
            
            # Make navigator interactive for clicking
            nav_layer.interactive = True
            
        # Function to handle clicks on navigator - region-aware coordinate handling
        def create_navigator_click_handler(region_name, nav_layer, nav_metadata):
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
                            
                            print(f"Navigator click in {{region_name}}: viewer({{viewer_click_x}}, {{viewer_click_y}}) -> nav_local({{nav_local_x}}, {{nav_local_y}}) -> grid({{grid_row}}, {{grid_col}})")
                            print(f"Tile size: {{tile_size}}, Navigator dimensions: {{nav_width}}x{{nav_height}}")
                            
                            # Find the corresponding FOV in THIS region's grid
                            fov_grid = nav_metadata.get('fov_grid', {{}})
                            print(f"FOV grid has {{len(fov_grid)}} entries: {{list(fov_grid.keys())[:10]}}...")  # Show first 10 grid positions
                            
                            if (grid_row, grid_col) in fov_grid:
                                clicked_fov = fov_grid[(grid_row, grid_col)]
                                
                                print(f"Clicked FOV: {{clicked_fov}} in region {{region_name}}")
                                
                                # Find indices in the main viewer - but constrain to current region
                                try:
                                    # First, ensure we're in the correct region
                                    region_idx = region_names.index(region_name) if region_name in region_names else None
                                    fov_idx = fov_names.index(clicked_fov) if clicked_fov in fov_names else None
                                    
                                    if region_idx is not None and fov_idx is not None:
                                        # Update viewer dimensions to jump to this FOV
                                        current = list(viewer.dims.current_step)
                                        if len(current) >= 3:
                                            current[1] = region_idx  # Region dimension - force to this region
                                            current[2] = fov_idx     # FOV dimension
                                            viewer.dims.current_step = current
                                            print(f"Jumped to Region: {{region_name}} (idx={{region_idx}}), FOV: {{clicked_fov}} (idx={{fov_idx}})")
                                    else:
                                        print(f"Could not find indices for region/FOV: {{region_name}}/{{clicked_fov}}")
                                        
                                except Exception as e:
                                    print(f"Error in navigator click handling: {{e}}")
                            else:
                                print(f"No FOV found at grid position ({{grid_row}}, {{grid_col}}) in region {{region_name}}")
                                print(f"Available grid positions: {{sorted(list(fov_grid.keys()))}}")
                                
                                # Try to find nearby grid positions to help debug
                                nearby_positions = []
                                for dr in [-1, 0, 1]:
                                    for dc in [-1, 0, 1]:
                                        test_pos = (grid_row + dr, grid_col + dc)
                                        if test_pos in fov_grid:
                                            nearby_positions.append(f"{{test_pos}}->FOV{{fov_grid[test_pos]}}")
                                
                                if nearby_positions:
                                    print(f"Nearby FOVs: {{nearby_positions}}")
                                else:
                                    print("No nearby FOVs found")
                        else:
                            print(f"Click outside navigator bounds: nav_local({{nav_local_x}}, {{nav_local_y}})")
        
        # Create click handlers for all navigator layers
        for region_name, nav_layer in nav_layers.items():
            if region_name in region_navigators:
                _, _, nav_metadata = region_navigators[region_name]
                create_navigator_click_handler(region_name, nav_layer, nav_metadata)
        
        print("=== END POSITIONING ===")
        
        # Function to switch navigator visibility based on current region
        def switch_navigator_region():
            try:
                # Get current region from sliders
                current_step = viewer.dims.current_step
                if len(current_step) >= 3:
                    region_idx = current_step[1]
                    current_fov_idx = current_step[2]
                    
                    if region_idx < len(region_names):
                        current_region = region_names[region_idx]
                        print(f"Switching to navigator for region: {{current_region}}")
                        
                        # First, remove all navigator boxes before switching
                        boxes_to_remove = []
                        for layer in viewer.layers:
                            if hasattr(layer, 'name') and 'Navigator Box' in layer.name:
                                boxes_to_remove.append(layer)
                        
                        for box_layer in boxes_to_remove:
                            try:
                                viewer.layers.remove(box_layer)
                            except:
                                pass
                        nav_box_layers.clear()
                        
                        # Hide all navigator layers and borders
                        for region_name in nav_layers:
                            if region_name in nav_layers:
                                nav_layers[region_name].visible = False
                            if region_name in nav_border_layers:
                                nav_border_layers[region_name].visible = False
                        
                        # Show current region's navigator and border
                        if current_region in nav_layers:
                            nav_layers[current_region].visible = True
                            print(f"Made navigator visible for region: {{current_region}}")
                        if current_region in nav_border_layers:
                            nav_border_layers[current_region].visible = True
                            print(f"Made navigator border visible for region: {{current_region}}")
                        
                        # CRITICAL FIX: Check if current FOV belongs to this region
                        # If not, jump to the first FOV that belongs to this region
                        if current_region in region_navigators:
                            _, _, nav_metadata = region_navigators[current_region]
                            region_fovs = list(nav_metadata.get('coordinates', {{}}).keys())
                            
                            current_fov_name = fov_names[current_fov_idx] if current_fov_idx < len(fov_names) else None
                            
                            print(f"Current FOV: {{current_fov_name}}, Region FOVs: {{region_fovs[:5]}}...") # Show first 5 for brevity
                            
                            if current_fov_name not in region_fovs:
                                # Current FOV doesn't belong to this region, jump to first FOV in this region
                                if region_fovs:
                                    first_region_fov = region_fovs[0]
                                    try:
                                        first_fov_idx = fov_names.index(first_region_fov)
                                        print(f"FOV {{current_fov_name}} doesn't belong to region {{current_region}}, jumping to {{first_region_fov}} (idx={{first_fov_idx}})")
                                        
                                        # Update the viewer to jump to the correct FOV
                                        new_step = list(viewer.dims.current_step)
                                        new_step[2] = first_fov_idx  # Update FOV dimension
                                        viewer.dims.current_step = new_step
                                        
                                    except ValueError:
                                        print(f"Could not find index for FOV {{first_region_fov}}")
                                else:
                                    print(f"No FOVs found for region {{current_region}}")
                            else:
                                print(f"FOV {{current_fov_name}} belongs to region {{current_region}}, keeping current position")
                        
                        print(f"Navigator switched to region: {{current_region}}")
                        
            except Exception as e:
                print(f"Error switching navigator region: {{e}}")
                import traceback
                traceback.print_exc()
        
        # Function to update navigator box - region-aware positioning
        def update_navigator_box():
            try:
                # Get current FOV and region from sliders
                current_step = viewer.dims.current_step
                if len(current_step) >= 3:
                    region_idx = current_step[1]
                    fov_idx = current_step[2]
                    
                    # Get the actual region and fov names
                    if region_idx < len(region_names) and fov_idx < len(fov_names):
                        current_region = region_names[region_idx]
                        current_fov = fov_names[fov_idx]
                        
                        print(f"Updating navigator box for region {{current_region}}, FOV {{current_fov}}")
                        
                        # Remove ALL existing navigator boxes (clean slate)
                        boxes_to_remove = []
                        for layer in viewer.layers:
                            if hasattr(layer, 'name') and 'Navigator Box' in layer.name:
                                boxes_to_remove.append(layer)
                        
                        for box_layer in boxes_to_remove:
                            try:
                                viewer.layers.remove(box_layer)
                                print(f"Removed old navigator box: {{box_layer.name}}")
                            except:
                                pass
                        
                        # Clear the tracking dictionary
                        nav_box_layers.clear()
                        
                        # Only show box if we have a navigator for the current region AND it's visible
                        if (current_region in region_navigators and 
                            current_region in nav_layers and 
                            nav_layers[current_region].visible):
                            
                            _, _, nav_metadata = region_navigators[current_region]
                            nav_layer = nav_layers[current_region]
                            
                            # Check if the current FOV belongs to this region
                            region_fovs = list(nav_metadata.get('coordinates', {{}}).keys())
                            if current_fov not in region_fovs:
                                print(f"FOV {{current_fov}} does not belong to region {{current_region}} (has {{len(region_fovs)}} FOVs), skipping navigator box")
                                # Don't return here - let the region switching handle jumping to correct FOV
                                return
                            
                            # Find this FOV in the current region's navigator grid
                            fov_to_grid = nav_metadata.get('fov_to_grid_pos', {{}})
                            
                            if current_fov in fov_to_grid:
                                grid_row, grid_col = fov_to_grid[current_fov]
                                tile_size = nav_metadata['tile_size']
                                
                                print(f"FOV {{current_fov}} found at grid position ({{grid_row}}, {{grid_col}}) in region {{current_region}}")
                                
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
                                
                                # Add new box as a shapes layer
                                nav_box_layer = viewer.add_shapes(
                                    [box_coords],
                                    shape_type='rectangle',
                                    edge_color='red',
                                    face_color='transparent',
                                    edge_width=3,
                                    name=f'Navigator Box {{current_region}}',
                                    opacity=1.0
                                )
                                
                                nav_box_layers[current_region] = nav_box_layer
                                
                                # Make the red box non-interactive
                                nav_box_layer.interactive = False
                                nav_box_layer.visible = True
                                nav_box_layer.editable = False
                                nav_box_layer.mouse_pan = False
                                nav_box_layer.mouse_zoom = False
                                
                                print(f"Created navigator box for region {{current_region}}, FOV {{current_fov}} at grid({{grid_row}}, {{grid_col}}) -> viewer({{viewer_x}}, {{viewer_y}})")
                            else:
                                print(f"FOV {{current_fov}} not found in region {{current_region}} navigator grid mapping")
                        else:
                            print(f"No visible navigator for region {{current_region}}, skipping navigator box")
                                
            except Exception as e:
                print(f"Error updating navigator box: {{e}}")
                import traceback
                traceback.print_exc()
        
        # Combined function that handles both region switching and box updates
        def on_dimension_change():
            switch_navigator_region()
            update_navigator_box()
        
        # Connect the update function to dimension changes
        viewer.dims.events.current_step.connect(on_dimension_change)
        
        # Add layer selection event handler to prevent navigator box selection
        @viewer.layers.selection.events.active.connect
        def on_layer_selection_changed(event):
            # If navigator box is selected, immediately switch back to navigator
            if (hasattr(event, 'value') and event.value is not None and 
                hasattr(event.value, 'name') and 'Navigator Box' in event.value.name):
                try:
                    # Find current region navigator to switch back to
                    current_step = viewer.dims.current_step
                    if len(current_step) >= 2:
                        region_idx = current_step[1]
                        if region_idx < len(region_names):
                            current_region = region_names[region_idx]
                            if current_region in nav_layers:
                                viewer.layers.selection.active = nav_layers[current_region]
                                print(f"Prevented Navigator Box selection, switched back to Navigator {{current_region}}")
                except Exception as e:
                    print(f"Error switching back to navigator layer: {{e}}")
        
        # ===== CALCULATE REAL NAPARI COORDINATE SYSTEM AREAS =====
        def calculate_napari_area_ratios():
            try:
                print("\\n" + "="*70)
                print("NAPARI VIEWER COORDINATE SYSTEM COMPARISON")
                print("="*70)
                
                # Get FOV layer extent (main image)
                main_image_layer = None
                for layer in viewer.layers:
                    if hasattr(layer, 'data') and 'Channel:' in layer.name:
                        main_image_layer = layer
                        break
                
                if main_image_layer:
                    fov_extent = main_image_layer.extent
                    fov_world_bounds = fov_extent.world
                    
                    # Calculate FOV dimensions in napari viewer coordinates
                    fov_width_napari = fov_world_bounds[1, -1] - fov_world_bounds[0, -1]  # Max X - Min X
                    fov_height_napari = fov_world_bounds[1, -2] - fov_world_bounds[0, -2]  # Max Y - Min Y
                    fov_area_napari = fov_width_napari * fov_height_napari
                    
                    print(f"FOV layer in napari coordinates: {{fov_width_napari:.1f}} x {{fov_height_napari:.1f}} units")
                    print(f"FOV area in napari coordinates: {{fov_area_napari:,.0f}} square units")
                    
                    # Get navigator layer extent (first visible navigator)
                    active_nav_layer = None
                    for region_name, nav_layer in nav_layers.items():
                        if nav_layer.visible:
                            active_nav_layer = nav_layer
                            break
                    
                    if active_nav_layer:
                        nav_extent = active_nav_layer.extent
                        nav_world_bounds = nav_extent.world
                        
                        # Calculate navigator dimensions in napari viewer coordinates  
                        nav_width_napari = nav_world_bounds[1, -1] - nav_world_bounds[0, -1]  # Max X - Min X
                        nav_height_napari = nav_world_bounds[1, -2] - nav_world_bounds[0, -2]  # Max Y - Min Y
                        nav_area_napari = nav_width_napari * nav_height_napari
                        
                        print(f"Navigator layer in napari coordinates: {{nav_width_napari:.1f}} x {{nav_height_napari:.1f}} units")
                        print(f"Navigator area in napari coordinates: {{nav_area_napari:,.0f}} square units")
                        
                        # Calculate the REAL area ratio
                        real_area_ratio = nav_area_napari / fov_area_napari
                        
                        print(f"\\nREAL NAPARI AREA RATIO:")
                        print(f"Navigator/FOV area ratio: {{real_area_ratio:.2f}}:1")
                        print(f"This means the navigator covers {{real_area_ratio:.2f}}x the area of one FOV")
                        print(f"Or: One FOV covers {{1/real_area_ratio:.3f}} of the navigator area")
                        
                        # Show the coordinate scaling
                        coord_scale_x = nav_width_napari / active_nav_layer.data.shape[1]
                        coord_scale_y = nav_height_napari / active_nav_layer.data.shape[0]
                        
                        print(f"\\nNAPARI COORDINATE SCALING:")
                        print(f"Navigator: {{active_nav_layer.data.shape[1]}} x {{active_nav_layer.data.shape[0]}} pixels -> {{nav_width_napari:.1f}} x {{nav_height_napari:.1f}} napari units")
                        print(f"Coordinate scale: {{coord_scale_x:.3f}} napari units per pixel (X), {{coord_scale_y:.3f}} napari units per pixel (Y)")
                        
                    else:
                        print("No visible navigator layer found")
                else:
                    print("No main image layer found")
                    
                print("="*70)
                
            except Exception as e:
                print(f"Error calculating napari area ratios: {{e}}")
                import traceback
                traceback.print_exc()
        
        # ===== ENSURE PROPER FOV TO NAVIGATOR SCALING =====
        def ensure_proper_fov_scaling():
            try:
                print("\\n" + "="*60)
                print("ADJUSTING FOV SCALING FOR PROPER PROPORTIONS")
                print("="*60)
                
                # Get FOV layer extent (main image)
                main_image_layer = None
                for layer in viewer.layers:
                    if hasattr(layer, 'data') and 'Channel:' in layer.name:
                        main_image_layer = layer
                        break
                
                # Get navigator layer extent (first available navigator)
                active_nav_layer = None
                for region_name, nav_layer in nav_layers.items():
                    active_nav_layer = nav_layer
                    break
                
                if main_image_layer and active_nav_layer:
                    # Calculate current areas
                    fov_extent = main_image_layer.extent
                    fov_world_bounds = fov_extent.world
                    fov_width_napari = fov_world_bounds[1, -1] - fov_world_bounds[0, -1]
                    fov_height_napari = fov_world_bounds[1, -2] - fov_world_bounds[0, -2]
                    fov_area_napari = fov_width_napari * fov_height_napari
                    
                    nav_extent = active_nav_layer.extent
                    nav_world_bounds = nav_extent.world
                    nav_width_napari = nav_world_bounds[1, -1] - nav_world_bounds[0, -1]
                    nav_height_napari = nav_world_bounds[1, -2] - nav_world_bounds[0, -2]
                    nav_area_napari = nav_width_napari * nav_height_napari
                    
                    # Calculate current ratio (FOV area / Navigator area)
                    current_ratio = fov_area_napari / nav_area_napari
                    target_ratio = 8.0  # FOV should be 8x bigger than navigator
                    
                    print(f"Current FOV/Navigator area ratio: {{current_ratio:.3f}}:1")
                    print(f"Target FOV/Navigator area ratio: {{target_ratio:.1f}}:1")
                    
                    if current_ratio < target_ratio:
                        # Calculate required scaling factor
                        # Since area scales with the square of linear scaling, we need sqrt of the area ratio
                        required_scale_factor = (target_ratio / current_ratio) ** 0.5
                        
                        print(f"FOV is too small, applying scale factor: {{required_scale_factor:.2f}}")
                        
                        # Apply scaling to all FOV image layers
                        for layer in viewer.layers:
                            if hasattr(layer, 'data') and 'Channel:' in layer.name:
                                # Get current scale
                                current_scale = list(layer.scale)
                                
                                # Apply the scale factor to the spatial dimensions (Y and X)
                                # Scale array format: [time, region, fov, z, y, x]
                                current_scale[-1] *= required_scale_factor  # X dimension
                                current_scale[-2] *= required_scale_factor  # Y dimension
                                
                                # Update the layer scale
                                layer.scale = current_scale
                                
                                print(f"Updated scale for layer {{layer.name}}: {{current_scale}}")
                        
                        # Recalculate and display the new ratio
                        print("\\nRechecking after scaling adjustment...")
                        # Give napari a moment to update
                        import time
                        time.sleep(0.1)
                        
                        # Recalculate the areas after scaling
                        fov_extent_new = main_image_layer.extent
                        fov_world_bounds_new = fov_extent_new.world
                        fov_width_new = fov_world_bounds_new[1, -1] - fov_world_bounds_new[0, -1]
                        fov_height_new = fov_world_bounds_new[1, -2] - fov_world_bounds_new[0, -2]
                        fov_area_new = fov_width_new * fov_height_new
                        
                        new_ratio = fov_area_new / nav_area_napari
                        
                        print(f"New FOV dimensions: {{fov_width_new:.1f}} x {{fov_height_new:.1f}} napari units")
                        print(f"New FOV area: {{fov_area_new:,.0f}} square units")
                        print(f"New FOV/Navigator area ratio: {{new_ratio:.2f}}:1")
                        
                        if new_ratio >= target_ratio * 0.95:  # Allow 5% tolerance
                            print("✓ FOV scaling successfully adjusted!")
                        else:
                            print("⚠ FOV scaling may need further adjustment")
                            
                    else:
                        print(f"✓ FOV is already {{current_ratio:.2f}}x bigger than navigator (target: {{target_ratio}}x)")
                        print("No scaling adjustment needed")
                    
                else:
                    print("Could not find FOV or Navigator layers for scaling adjustment")
                    
                print("="*60)
                
            except Exception as e:
                print(f"Error in FOV scaling adjustment: {{e}}")
                import traceback
                traceback.print_exc()
        
        # Apply the scaling adjustment
        ensure_proper_fov_scaling()
        
        # Update navigator positioning after scaling
        print("\\n=== UPDATING NAVIGATOR POSITIONING AFTER SCALING ===")
        fov_right_after_scaling, fov_top_after_scaling = get_fov_positioning_bounds()
        
        # Reposition all navigator layers with updated FOV bounds
        for i, (region_name, nav_layer) in enumerate(nav_layers.items()):
            nav_height, nav_width = nav_layer.data.shape
            gap = 50  # Increased gap for better visibility
            
            nav_x_position = fov_right_after_scaling + gap
            nav_y_position = fov_top_after_scaling + gap
            
            print(f"Repositioning navigator {{region_name}} to: x={{nav_x_position}}, y={{nav_y_position}}")
            nav_layer.translate = (nav_y_position, nav_x_position)
            
            # Update border position too
            if region_name in nav_border_layers:
                nav_border_coords = [
                    [nav_y_position, nav_x_position],  # Top-left
                    [nav_y_position, nav_x_position + nav_width],  # Top-right
                    [nav_y_position + nav_height, nav_x_position + nav_width],  # Bottom-right
                    [nav_y_position + nav_height, nav_x_position]  # Bottom-left
                ]
                
                # Update the border layer data
                nav_border_layers[region_name].data = [nav_border_coords]
        
        print("=== NAVIGATOR REPOSITIONING COMPLETE ===")
        
        # Calculate and display real napari coordinate areas (after scaling)
        calculate_napari_area_ratios()
        
        # Auto-zoom to fit all visible layers
        def auto_zoom_to_fit():
            try:
                print("\\n=== AUTO-ZOOM TO FIT ALL LAYERS ===")
                
                # Reset the camera to show all layers
                viewer.reset_view()
                
                print("✓ Auto-zoom completed - all layers should now be visible")
                
            except Exception as e:
                print(f"Error in auto-zoom: {{e}}")
                # Fallback method
                try:
                    # Get the camera and set zoom level manually
                    camera = viewer.camera
                    camera.zoom = 0.1  # Very zoomed out
                    print("✓ Fallback zoom applied")
                except Exception as e2:
                    print(f"Fallback zoom also failed: {{e2}}")
        
        # Apply auto-zoom after a brief delay to ensure all layers are loaded
        import time
        time.sleep(0.2)
        auto_zoom_to_fit()
        
        # Initial update
        on_dimension_change()
        
        print("Region-specific navigator overlays created successfully!")
            
    except Exception as e:
        print(f"Failed to create navigator: {{e}}")
        import traceback
        traceback.print_exc()

else:
    if not NAVIGATOR_AVAILABLE:
        print("Navigator not available - module not imported")
    else:
        print(f"Navigator conditions not met: regions={{dims.get('region', 0)}}, fovs={{dims.get('fov', 0)}}")

# Optional: Add text overlays for region and FOV names
@viewer.dims.events.current_step.connect
def update_info(event):
    # This will update when sliders change
    current_dims = viewer.dims.current_step
    if len(current_dims) >= 3:  # Make sure we have enough dimensions
        region_idx = current_dims[1]
        fov_idx = current_dims[2]
        if region_idx < len(region_names) and fov_idx < len(fov_names):
            region_name = region_names[region_idx]
            fov_name = fov_names[fov_idx]
            # Update window title with current region and FOV
            viewer.title = f"Napari - {{folder_name}} - Region: {{region_name}}, FOV: {{fov_name}}"

napari.run()
""")
        
        # Launch the script in a separate process
        subprocess.Popen([sys.executable, launcher_script, temp_file])
    
    def handle_error(self, error_msg, folder_name):
        self.label.setText(f'Error loading {{folder_name}}: {{error_msg}}')


if __name__ == "__main__":
    # Create PyQt application
    app = QApplication(sys.argv)
    
    # Create and show the directory selector
    selector = DirectorySelector()
    selector.show()
    
    # Run the application
    sys.exit(app.exec())
