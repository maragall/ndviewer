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

# Import the downsampler module from parent directory
try:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        
        # Get the path to the viewer script
        viewer_script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'napari_viewer_script.py')
        
        # Launch the viewer script in a separate process
        subprocess.Popen([sys.executable, viewer_script_path, temp_file])
    
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