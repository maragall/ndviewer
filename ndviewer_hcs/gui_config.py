"""Configuration GUI for preprocessing settings"""

import sys
from pathlib import Path
from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSpinBox, QFileDialog, 
                             QGroupBox, QRadioButton, QButtonGroup, QMessageBox,
                             QProgressBar, QApplication, QFrame)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QDragEnterEvent, QDropEvent
import json

from .preprocessing import FlatfieldManager, PlateAssembler
from .viewer import ViewerMainWindow


class DropBox(QFrame):
    """Drag-and-drop box for folder selection"""
    folder_dropped = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setup_ui()
    
    def setup_ui(self):
        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)
        self.setStyleSheet("""
            DropBox {
                background-color: #f8f8f8;
                border: 2px dashed #999;
                border-radius: 5px;
            }
            DropBox:hover {
                background-color: #e8f4f8;
                border-color: #4CAF50;
            }
        """)
        
        layout = QVBoxLayout()
        self.label = QLabel("")  # Empty label
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("border: none;")
        layout.addWidget(self.label)
        self.setLayout(layout)
    
    def set_folder_path(self, path):
        """Update display with folder path"""
        # Show folder name and change border color
        folder_name = Path(path).name
        self.label.setText(f"{folder_name}")
        self.label.setStyleSheet("color: #2e7d32; font-weight: bold; font-size: 11px; border: none;")
        self.setStyleSheet("""
            DropBox {
                background-color: #e8f5e9;
                border: 2px solid #4CAF50;
                border-radius: 5px;
            }
        """)
    
    def set_status(self, message):
        """Update status message in dropbox"""
        self.label.setText(message)
        self.label.setStyleSheet("color: #666; font-size: 11px; border: none;")
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
    
    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            folder_path = urls[0].toLocalFile()
            if Path(folder_path).is_dir():
                self.folder_dropped.emit(folder_path)


class PreprocessingThread(QThread):
    """Background thread for preprocessing to avoid blocking UI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, base_path, downsample_factor, flatfield_option, flatfield_path=None):
        super().__init__()
        self.base_path = base_path
        self.downsample_factor = downsample_factor
        self.flatfield_option = flatfield_option
        self.flatfield_path = flatfield_path
    
    def run(self):
        try:
            self.progress.emit("Starting preprocessing...")
            
            # # Handle flatfield option - DISABLED (Less is more)
            # flatfields = None
            # if self.flatfield_option == "compute":
            #     self.progress.emit("Computing flatfields...")
            #     ff_manager = FlatfieldManager(self.base_path)
            #     flatfields = ff_manager.compute_flatfields()
            # elif self.flatfield_option == "load" and self.flatfield_path:
            #     self.progress.emit("Loading flatfields...")
            #     ff_manager = FlatfieldManager(self.base_path)
            #     flatfields = ff_manager.load_flatfields(self.flatfield_path)
            
            # Assemble plate (no flatfield correction)
            self.progress.emit("Assembling plate...")
            assembler = PlateAssembler(self.base_path, timepoint=0)
            # Convert downsample_factor to target_pixel_size for assembler
            original_px = assembler._get_original_pixel_size()
            target_px = int(original_px * self.downsample_factor)
            assembler.assemble_plate(target_px, flatfields=None, 
                                    output_dir=str(Path(self.base_path) / "assembled_tiles_cache"))
            
            self.finished.emit(True, "Preprocessing completed successfully!")
        except Exception as e:
            self.finished.emit(False, f"Error during preprocessing: {str(e)}")


class ConfigurationGUI(QMainWindow):
    """Central GUI for preprocessing configuration"""
    
    TARGET_PIXEL_SIZE_UM = 10.0  # Target pixel size on sample plane in micrometers
    MAGNIFICATION = 10  # Objective magnification
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NDViewer HCS")
        self.setFixedSize(200, 200)  # Square dropbox
        
        self.acquisition_dir = None
        self.sensor_pixel_size_um = None
        self.downsample_factor = None
        self.flatfield_path = None
        
        self.setup_ui()
    
    def setup_ui(self):
        central_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)  # Small margins around dropbox
        layout.setSpacing(0)
        
        # Just the dropbox - nothing else
        self.dropbox = DropBox()
        self.dropbox.folder_dropped.connect(self.on_folder_selected)
        layout.addWidget(self.dropbox)
        
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)
    
    def on_folder_selected(self, folder):
        """Handle folder selection - automatically launch viewer"""
        self.acquisition_dir = folder
        self.dropbox.set_folder_path(folder)
        self.dropbox.set_status("Reading acquisition parameters...")
        
        # Read sensor pixel size from acquisition_parameters.json
        self.sensor_pixel_size_um = self.read_sensor_pixel_size(folder)
        if self.sensor_pixel_size_um:
            # Calculate actual pixel size on sample plane
            sample_pixel_size_um = self.sensor_pixel_size_um / self.MAGNIFICATION
            
            # Downsample factor: sample_pixel_size / target_pixel_size
            # If target > sample: downsample (factor < 1, make image smaller)
            # If target < sample: upsample (factor > 1, make image larger)
            self.downsample_factor = sample_pixel_size_um / self.TARGET_PIXEL_SIZE_UM
            
            # Print scaling info to debug
            print(f"Sensor pixel size: {self.sensor_pixel_size_um} µm")
            print(f"Magnification: {self.MAGNIFICATION}x")
            print(f"Sample pixel size: {sample_pixel_size_um} µm")
            print(f"Target pixel size: {self.TARGET_PIXEL_SIZE_UM} µm")
            print(f"Downsample factor: {self.downsample_factor:.4f}x")
            
            # Auto-launch viewer
            self.launch_viewer_with_preprocessing()
        else:
            self.dropbox.set_status("Error: Could not read parameters")
            print("Error: Could not read sensor_pixel_size_um from acquisition_parameters.json")
    
    # def browse_flatfield(self):  # DISABLED (Less is more)
    #     file_path, _ = QFileDialog.getOpenFileName(self, "Select Flatfield File", "", "Flatfield Files (*.pkl *.npy)")
    #     if file_path:
    #         self.flatfield_path = file_path
    #         QMessageBox.information(self, "Flatfield Selected", f"Selected: {Path(file_path).name}")
    
    def read_sensor_pixel_size(self, acquisition_dir: str) -> float:
        """Read sensor_pixel_size_um from acquisition_parameters.json"""
        try:
            path = Path(acquisition_dir)
            json_file = path / "acquisition parameters.json"
            
            if not json_file.exists():
                print(f"acquisition parameters.json not found in {acquisition_dir}")
                return None
            
            with open(json_file, 'r') as f:
                params = json.load(f)
            
            sensor_pixel_size = params.get('sensor_pixel_size_um')
            if sensor_pixel_size:
                return float(sensor_pixel_size)
            else:
                print("sensor_pixel_size_um not found in acquisition parameters")
                return None
                
        except Exception as e:
            print(f"Error reading acquisition parameters: {e}")
            return None
    
    def run_preprocessing(self):
        """Execute preprocessing in background thread"""
        if not self.acquisition_dir:
            return
        
        # Start preprocessing thread (no flatfield, simplified)
        self.preprocessing_thread = PreprocessingThread(
            self.acquisition_dir,
            self.downsample_factor,
            flatfield_option="none",
            flatfield_path=None
        )
        self.preprocessing_thread.progress.connect(self.on_progress)
        self.preprocessing_thread.finished.connect(self.on_preprocessing_finished)
        self.preprocessing_thread.start()
    
    def on_progress(self, message):
        """Update dropbox with progress"""
        # Show abstracted user-friendly message
        if "Starting" in message:
            self.dropbox.set_status("Preparing data...")
        elif "Assembling" in message:
            self.dropbox.set_status("Building plate...")
        else:
            self.dropbox.set_status(message)
        print(message)
    
    def on_preprocessing_finished(self, success, message):
        """Handle preprocessing completion and launch viewer"""
        if success:
            self.dropbox.set_status("Opening viewer...")
            # Auto-launch viewer after successful preprocessing
            self.launch_viewer()
        else:
            self.dropbox.set_status("Error occurred")
            print(f"Error: {message}")
    
    def launch_viewer_with_preprocessing(self):
        """Check if preprocessing is needed, run it, then launch viewer"""
        if not self.acquisition_dir:
            return
        
        self.dropbox.set_status("Checking for cached data...")
        
        # Check if preprocessing is needed
        from .preprocessing import PlateAssembler
        assembler = PlateAssembler(self.acquisition_dir, timepoint=0)
        cache_key = assembler._get_cache_key(self.downsample_factor, self.downsample_factor >= 0.98)
        
        if assembler._cache_exists(cache_key):
            # Cache exists, launch viewer directly
            self.dropbox.set_status("Loading cached data...")
            self.launch_viewer()
        else:
            # Need to preprocess first
            self.run_preprocessing()
    
    def launch_viewer(self):
        """Open viewer with preprocessed data"""
        try:
            self.dropbox.set_status("Starting viewer...")
            self.viewer_window = ViewerMainWindow(self.acquisition_dir, timepoint=0, 
                                                 downsample_factor=self.downsample_factor)
            self.viewer_window.show()
            self.dropbox.set_status("Viewer ready")
        except Exception as e:
            self.dropbox.set_status("Error launching viewer")
            print(f"Error: Failed to launch viewer: {str(e)}")


def main():
    """Entry point for configuration GUI"""
    app = QApplication(sys.argv)
    window = ConfigurationGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

