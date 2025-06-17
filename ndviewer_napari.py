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
from PyQt6.QtCore import QObject, pyqtSignal, QThread

# Import the downsampler module
try:
    from downsampler import DownsampledNavigator
except ImportError:
    print("Warning: downsampler module not found. Navigator will be disabled.")
    DownsampledNavigator = None

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
        
        self.label = QLabel('Please select the directory containing your acquisition data')
        layout.addWidget(self.label)
        
        self.browse_button = QPushButton('Browse...')
        self.browse_button.clicked.connect(self.browse_directory)
        layout.addWidget(self.browse_button)
        
        self.setLayout(layout)
        
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, 'Select Acquisition Directory')
        if directory:
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
            f.write("""
import os
import sys
import pickle
import napari
import dask.array as da
import dask
import tifffile
import numpy as np
from napari.layers import Image
from vispy.scene.visuals import Rectangle
from vispy.scene import PanZoomCamera

# Try to import downsampler
try:
    sys.path.insert(0, os.path.dirname('""" + os.path.abspath(__file__) + """'))
    from downsampler import DownsampledNavigator
    NAVIGATOR_AVAILABLE = True
except ImportError:
    print("Warning: Navigator module not available")
    NAVIGATOR_AVAILABLE = False

# Load the metadata from the temp file
with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)

directory = data['directory']
metadata = data['metadata']
folder_name = data['folder_name']

# Get dimensions
dims = metadata['dimensions']
file_map = metadata['file_map']
acq_params = metadata['acquisition_parameters']

# Calculate pixel size in microns
pixel_size_um = None
z_step_um = None

if acq_params:
    # Get pixel size from sensor pixel size and objective magnification
    if 'sensor_pixel_size_um' in acq_params and 'objective' in acq_params and 'magnification' in acq_params['objective']:
        sensor_pixel_size = acq_params['sensor_pixel_size_um']
        magnification = acq_params['objective']['magnification']
        pixel_size_um = sensor_pixel_size / magnification
        print(f"Calculated pixel size: {pixel_size_um:.3f} µm")
    
    # Get Z step size
    if 'dz(um)' in acq_params:
        z_step_um = acq_params['dz(um)']
        print(f"Z step size: {z_step_um} µm")

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
print(f"Opening napari viewer for: {folder_name}")
viewer = napari.Viewer(title=f"Napari - {folder_name}")

# The dimensions are: (t, c, r, f, z, y, x)
channel_names = metadata['channels']
region_names = metadata['regions']
fov_names = metadata['fovs']

# Calculate scale for each dimension
# [time, region, fov, z, y, x]
# For time, region, and fov, we use 1.0 as they're indices
# For z, y, x we use the physical dimensions
scale = [1.0, 1.0, 1.0, z_step_um, pixel_size_um, pixel_size_um]

# Add each channel as a separate layer
for c in range(dims['channel']):
    # For each channel, we keep all other dimensions
    # This creates a 6D array: (t, r, f, z, y, x)
    channel_data = dask_data[:, c, :, :, :, :, :]
    
    # Create a meaningful name for the layer
    layer_name = f"Channel: {channel_names[c]}"
    
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
        metadata={
            'dimension_names': ['Time', 'Region', 'FOV', 'Z', 'Y', 'X']
        }
    )

# Set dimension labels for the sliders
viewer.dims.axis_labels = ['Time', 'Region', 'FOV', 'Z', 'Y', 'X']

# Add physical units to the dimension labels
if z_step_um is not None and pixel_size_um is not None:
    viewer.scale_bar.unit = 'µm'
    viewer.scale_bar.visible = True

# ===== ADD NAVIGATOR FUNCTIONALITY =====
if NAVIGATOR_AVAILABLE and dims['region'] > 0 and dims['fov'] > 0:
    print("Creating navigator overlay...")
    
    try:
        # Create navigator with progress callback
        def progress_callback(percent, message):
            print(f"Navigator: {percent}% - {message}")
        
        navigator = DownsampledNavigator(
            directory, 
            tile_size=50,
            cache_enabled=True,
            progress_callback=progress_callback
        )
        
        # Create mosaic for current timepoint
        current_time = viewer.dims.current_step[0] if viewer.dims.current_step else 0
        mosaic_array, nav_metadata = navigator.create_mosaic(current_time)
        
        # Add navigator as an overlay layer
        nav_layer = viewer.add_image(
            mosaic_array,
            name='Navigator',
            opacity=0.8,
            colormap='gray',
            visible=True,
            scale=(1, 1),  # Navigator has its own coordinate system
            translate=(0, 0),
            metadata=nav_metadata
        )
        
        # Position navigator in top-right corner
        # Use the navigator layer's shape to determine positioning
        nav_height, nav_width = mosaic_array.shape
        # Position it in the top-right area (coordinates will be in data space)
        nav_layer.translate = (0, 0)  # Keep at origin for now
        
        # Create a red box overlay for current view
        # Instead of using vispy directly, use napari shapes layer for the overlay
        nav_box_layer = None
        
        # Function to update navigator box based on current view
        def update_navigator_box():
            global nav_box_layer
            try:
                # Get current FOV and region from sliders
                current_step = viewer.dims.current_step
                if len(current_step) >= 3:
                    region_idx = current_step[1]
                    fov_idx = current_step[2]
                    
                    # Get the actual region and fov names
                    if region_idx < len(region_names) and fov_idx < len(fov_names):
                        region_name = region_names[region_idx]
                        fov_name = fov_names[fov_idx]
                        
                        # Find this FOV in the navigator metadata
                        fov_grid = nav_metadata['fov_grid']
                        tile_size = nav_metadata['tile_size']
                        
                        # Find the grid position of this FOV
                        for (row, col), (nav_region, nav_fov) in fov_grid.items():
                            if nav_region == region_name and nav_fov == fov_name:
                                # Calculate pixel position in navigator
                                nav_x = col * tile_size + tile_size // 2
                                nav_y = row * tile_size + tile_size // 2
                                
                                # Create a rectangle shape for the box
                                box_size = tile_size * 0.8
                                box_coords = [
                                    [nav_y - box_size//2, nav_x - box_size//2],
                                    [nav_y - box_size//2, nav_x + box_size//2],
                                    [nav_y + box_size//2, nav_x + box_size//2],
                                    [nav_y + box_size//2, nav_x - box_size//2]
                                ]
                                
                                # Remove existing box layer if it exists
                                if nav_box_layer is not None and nav_box_layer in viewer.layers:
                                    viewer.layers.remove(nav_box_layer)
                                
                                # Add new box as a shapes layer
                                nav_box_layer = viewer.add_shapes(
                                    [box_coords],
                                    shape_type='rectangle',
                                    edge_color='red',
                                    face_color='transparent',
                                    edge_width=2,
                                    name='Navigator Box',
                                    opacity=0.8
                                )
                                
                                # Make it non-interactive
                                nav_box_layer.interactive = False
                                break
            except Exception as e:
                print(f"Error updating navigator box: {e}")
        
        # Connect the update function to dimension changes
        viewer.dims.events.current_step.connect(update_navigator_box)
        
        # Also update on camera changes (pan/zoom)
        viewer.camera.events.zoom.connect(update_navigator_box)
        viewer.camera.events.center.connect(update_navigator_box)
        
        # Initial update
        update_navigator_box()
        
        # Make navigator non-interactive to prevent accidental moves
        nav_layer.interactive = False
        
        # Function to handle clicks on navigator
        @nav_layer.mouse_drag_callbacks.append
        def on_navigator_click(layer, event):
            if event.button == 1:  # Left click
                # Get click position relative to navigator
                pos = event.position[:2]  # Get 2D position
                nav_pos = pos - nav_layer.translate
                
                # Convert to grid coordinates
                tile_size = nav_metadata['tile_size']
                col = int(nav_pos[0] / tile_size)
                row = int(nav_pos[1] / tile_size)
                
                # Find the corresponding FOV
                fov_grid = nav_metadata['fov_grid']
                if (row, col) in fov_grid:
                    region_name, fov_num = fov_grid[(row, col)]
                    
                    # Find indices
                    if region_name in region_names and fov_num in fov_names:
                        region_idx = region_names.index(region_name)
                        fov_idx = fov_names.index(fov_num)
                        
                        # Update viewer dimensions
                        current = list(viewer.dims.current_step)
                        current[1] = region_idx
                        current[2] = fov_idx
                        viewer.dims.current_step = current
        
        print("Navigator overlay created successfully!")
        
    except Exception as e:
        print(f"Failed to create navigator: {e}")
        import traceback
        traceback.print_exc()

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
            viewer.title = f"Napari - {folder_name} - Region: {region_name}, FOV: {fov_name}"

napari.run()
""")
        
        # Launch the script in a separate process
        subprocess.Popen([sys.executable, launcher_script, temp_file])
    
    def handle_error(self, error_msg, folder_name):
        self.label.setText(f'Error loading {folder_name}: {error_msg}')


if __name__ == "__main__":
    # Create PyQt application
    app = QApplication(sys.argv)
    
    # Create and show the directory selector
    selector = DirectorySelector()
    selector.show()
    
    # Run the application
    sys.exit(app.exec())