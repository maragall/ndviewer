"""Multi-dimensional plate stack manager for synchronized Z×T viewing"""

from pathlib import Path
from typing import Dict, Tuple, Optional, List
import numpy as np
import pickle
from PyQt5.QtCore import QObject, pyqtSignal, QThread


try:
    import tifffile as tf
    TIFFFILE_AVAILABLE = True
except ImportError:
    TIFFFILE_AVAILABLE = False


class PlateStackManager:
    """
    Manages pre-computed Z×T plate assemblies stored as multi-page TIFF.
    
    Structure:
        - Multi-page TIFF with shape: (t, z, channels, height, width)
        - Each page is a full assembled plate for one (t, z) combination
        - Memory-mapped for efficient random access
    """
    
    def __init__(self, base_path: str, cache_dir: Path):
        self.base_path = Path(base_path)
        self.cache_dir = cache_dir
        self.stack_file = self.cache_dir / "plate_stack_zt.tiff"
        self._memmap = None  # Memory-mapped TIFF for efficient access
        self._metadata = None
        
    def get_cache_key(self, downsample_factor: float) -> str:
        """Generate cache key for this downsample factor"""
        return f"plate_stack_ds{downsample_factor:.4f}"
    
    def exists(self, downsample_factor: float) -> bool:
        """Check if pre-built stack exists for this downsample factor"""
        cache_key = self.get_cache_key(downsample_factor)
        stack_file = self.cache_dir / f"{cache_key}.tiff"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.pkl"
        return stack_file.exists() and metadata_file.exists()
    
    def load_stack(self, downsample_factor: float) -> bool:
        """Load stack as memory-mapped array for efficient access"""
        if not TIFFFILE_AVAILABLE:
            print("Error: tifffile not available")
            return False
        
        cache_key = self.get_cache_key(downsample_factor)
        self.stack_file = self.cache_dir / f"{cache_key}.tiff"
        metadata_file = self.cache_dir / f"{cache_key}_metadata.pkl"
        
        if not self.stack_file.exists():
            print(f"Stack file not found: {self.stack_file}")
            return False
        
        try:
            # Try memory-mapping first
            try:
                self._memmap = tf.memmap(str(self.stack_file), mode='r')
                print(f"✓ Loaded plate stack (memory-mapped): {self._memmap.shape}")
            except Exception as memmap_err:
                # Fallback: load entire stack into memory
                print(f"Memory-mapping failed ({memmap_err}), loading into RAM...")
                self._memmap = tf.imread(str(self.stack_file))
                print(f"✓ Loaded plate stack (RAM): {self._memmap.shape}")
            
            # Load metadata
            if metadata_file.exists():
                with open(metadata_file, 'rb') as f:
                    self._metadata = pickle.load(f)
            
            return True
            
        except Exception as e:
            print(f"Error loading stack: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_page(self, t_idx: int, z_idx: int) -> Optional[np.ndarray]:
        """
        Get assembled plate for specific (t, z) indices.
        
        Returns:
            Multichannel plate image: (channels, height, width)
        """
        if self._memmap is None:
            print("Stack not loaded!")
            return None
        
        try:
            # Index into memmap: [t, z, :, :, :]
            page = self._memmap[t_idx, z_idx]
            # Ensure it's loaded into memory (not lazy)
            return np.array(page)
        except IndexError as e:
            print(f"Invalid indices: t={t_idx}, z={z_idx} - {e}")
            return None
        except Exception as e:
            print(f"Error getting page: {e}")
            return None
    
    def get_metadata(self) -> Optional[Dict]:
        """Get stack metadata (wavelengths, colormaps, dimensions, etc.)"""
        return self._metadata
    
    def get_shape(self) -> Optional[Tuple]:
        """Get stack shape (t, z, c, y, x)"""
        if self._memmap is not None:
            return self._memmap.shape
        return None


class StackBuilderThread(QThread):
    """Background thread for building Z×T plate stack"""
    
    progress = pyqtSignal(str)  # Progress message
    finished = pyqtSignal(bool, str)  # (success, message)
    
    def __init__(self, base_path: str, cache_dir: Path, downsample_factor: float):
        super().__init__()
        self.base_path = Path(base_path)
        self.cache_dir = cache_dir
        self.downsample_factor = downsample_factor
        self._stop_requested = False
    
    def request_stop(self):
        """Request thread to stop gracefully"""
        self._stop_requested = True
    
    def run(self):
        """Build the plate stack"""
        try:
            from .preprocessing import PlateAssembler
            
            # Detect available timepoints and z-levels
            self.progress.emit("Scanning data...")
            timepoints = self._detect_timepoints()
            z_levels = self._detect_z_levels()
            
            if not timepoints:
                self.finished.emit(False, "No timepoints found")
                return
            
            if not z_levels:
                self.finished.emit(False, "No z-levels found")
                return
            
            self.progress.emit(f"Building {len(timepoints)}T × {len(z_levels)}Z stack...")
            
            # Build stack
            pages = []
            metadata = {
                'timepoints': timepoints,
                'z_levels': z_levels,
                'downsample_factor': self.downsample_factor,
                'shape_info': None
            }
            
            total_pages = len(timepoints) * len(z_levels)
            page_count = 0
            
            # Build each (t, z) combination
            for t_idx, t in enumerate(timepoints):
                if self._stop_requested:
                    self.finished.emit(False, "Cancelled by user")
                    return
                
                for z_idx, z in enumerate(z_levels):
                    page_count += 1
                    self.progress.emit(f"Processing {page_count}/{total_pages}...")
                    
                    try:
                        # Assemble plate for this specific (t, z)
                        assembled = self._assemble_plate_for_tz(t, z)
                        
                        if assembled and 'multichannel' in assembled:
                            pages.append(assembled['multichannel'])
                            if metadata['shape_info'] is None:
                                metadata['shape_info'] = {
                                    'wavelengths': assembled['wavelengths'],
                                    'colormaps': assembled['colormaps']
                                }
                        else:
                            # Empty page if assembly failed
                            if pages:
                                pages.append(np.zeros_like(pages[0]))
                            else:
                                self.progress.emit(f"Warning: Failed to assemble t={t}, z={z}")
                                
                    except Exception as e:
                        self.progress.emit(f"Error at t={t}, z={z}: {e}")
                        if pages:
                            pages.append(np.zeros_like(pages[0]))
            
            if not pages:
                self.finished.emit(False, "No plates could be assembled")
                return
            
            # Stack into 5D array: (t, z, c, y, x)
            self.progress.emit("Finalizing...")
            stack = np.array(pages).reshape(
                len(timepoints), len(z_levels), *pages[0].shape
            )
            
            # Save as multi-page TIFF
            cache_key = f"plate_stack_ds{self.downsample_factor:.4f}"
            stack_file = self.cache_dir / f"{cache_key}.tiff"
            
            # Ensure cache directory exists
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            if TIFFFILE_AVAILABLE:
                tf.imwrite(str(stack_file), stack, compression='lzw', photometric='minisblack')
            else:
                from skimage import io
                # Fallback: reshape for skimage
                io.imsave(str(stack_file), stack)
            
            # Save metadata separately
            metadata_file = self.cache_dir / f"{cache_key}_metadata.pkl"
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            self.finished.emit(True, f"Stack ready: {len(timepoints)}T × {len(z_levels)}Z")
            
        except Exception as e:
            import traceback
            error_msg = f"Error building stack: {e}\n{traceback.format_exc()}"
            print(error_msg)
            self.finished.emit(False, error_msg)
    
    def _detect_timepoints(self) -> List[int]:
        """Detect available timepoint directories"""
        timepoints = []
        for item in self.base_path.iterdir():
            if item.is_dir() and item.name.isdigit():
                timepoints.append(int(item.name))
        return sorted(timepoints)
    
    def _detect_z_levels(self) -> List[int]:
        """Detect available z-levels from first timepoint"""
        from .common import fpattern, fpattern_ome, detect_acquisition_format
        
        # Look in first timepoint directory
        timepoints = self._detect_timepoints()
        if not timepoints:
            return []
        
        first_tp = self.base_path / str(timepoints[0])
        z_levels = set()
        
        # Check format
        format_type = detect_acquisition_format(self.base_path)
        
        if format_type == 'ome_tiff':
            # Get z-levels from OME-TIFF metadata using bioio
            from bioio import BioImage
            ome_dir = self.base_path / "ome_tiff" if (self.base_path / "ome_tiff").exists() else self.base_path / "0"
            for ome_file in ome_dir.glob("*.ome.tif*"):
                try:
                    bio_img = BioImage(str(ome_file))
                    n_z = bio_img.dims.Z
                    z_levels.update(range(n_z))
                    break  # Only need to check one file
                except:
                    pass
        else:
            # Single-TIFF: parse from filenames
            for tiff_file in first_tp.glob("*.tiff"):
                if m := fpattern.search(tiff_file.name):
                    z = int(m.group("z"))
                    z_levels.add(z)
        
        return sorted(list(z_levels))
    
    def _assemble_plate_for_tz(self, timepoint: int, z_level: int) -> Optional[Dict]:
        """Assemble plate for specific timepoint and z-level"""
        from .preprocessing import PlateAssembler
        from .common import detect_acquisition_format
        
        # Use PlateAssembler with z_level parameter
        assembler = PlateAssembler(str(self.base_path), timepoint=timepoint)
        
        # Get target pixel size
        original_px = assembler._get_original_pixel_size()
        target_px = int(original_px * self.downsample_factor)
        skip_downsampling = self.downsample_factor >= 0.98
        
        # Load tiles for this specific z-level
        # _load_tiles now handles both single-TIFF and OME-TIFF
        tiles = assembler._load_tiles(
            flatfields=None,
            downsample_factor=self.downsample_factor,
            skip_downsampling=skip_downsampling,
            z_level=z_level
        )
        
        if not tiles:
            return None
        
        # Assemble into canvas
        assembled_images, _ = assembler._assemble_images(tiles)
        return assembled_images


class NDVSyncController(QObject):
    """
    Synchronizes NDV viewer state (z, t indices) with plate stack display.
    
    Signals:
        indices_changed: Emitted when (t_idx, z_idx) changes
    """
    
    indices_changed = pyqtSignal(int, int)  # (t_idx, z_idx)
    
    def __init__(self, ndv_viewer, parent=None):
        super().__init__(parent)
        self.ndv_viewer = ndv_viewer
        self._current_t = 0
        self._current_z = 0
        self._connected = False
        self._timer = None
    
    def connect_to_viewer(self):
        """Connect to NDV viewer's dimension slider changes"""
        if self._connected:
            return
        
        from PyQt5.QtCore import QTimer
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_ndv_state)
        self._timer.start(100)  # Poll every 100ms
        self._connected = True
    
    def _poll_ndv_state(self):
        """Poll NDV viewer for index changes"""
        try:
            if hasattr(self.ndv_viewer, 'display_model'):
                display = self.ndv_viewer.display_model
                current_index = display.current_index
                
                # Get indices - handle both integer keys and string keys
                t_idx = current_index.get('time', current_index.get(0, 0))
                z_idx = current_index.get('z_level', current_index.get(1, 0))
                
                # Convert to int if needed
                if isinstance(t_idx, (slice, tuple, list)):
                    t_idx = 0
                if isinstance(z_idx, (slice, tuple, list)):
                    z_idx = 0
                t_idx = int(t_idx) if t_idx is not None else 0
                z_idx = int(z_idx) if z_idx is not None else 0
                
                # Emit signal if indices changed
                if t_idx != self._current_t or z_idx != self._current_z:
                    self._current_t = t_idx
                    self._current_z = z_idx
                    self.indices_changed.emit(t_idx, z_idx)
        except Exception as e:
            pass  # Silently ignore
    
    def disconnect(self):
        """Stop polling"""
        if self._timer:
            self._timer.stop()
        self._connected = False


class NDVContrastSyncController(QObject):
    """
    Synchronizes NDV channel contrast adjustments with plate display.
    
    Signals:
        contrast_changed: (channel_idx, vmin, vmax)
    """
    
    contrast_changed = pyqtSignal(int, float, float)  # (channel_idx, vmin, vmax)
    
    def __init__(self, ndv_viewer, parent=None):
        super().__init__(parent)
        self.ndv_viewer = ndv_viewer
        self._connected = False
        self._timer = None
        self._last_contrast = {}
    
    def connect_to_viewer(self):
        """Start polling for contrast changes"""
        if self._connected:
            return
        
        from PyQt5.QtCore import QTimer
        self._timer = QTimer()
        self._timer.timeout.connect(self._poll_contrast_state)
        self._timer.start(100)
        self._connected = True
    
    def _poll_contrast_state(self):
        """Poll NDV viewer for contrast limit changes"""
        try:
            if hasattr(self.ndv_viewer, 'display_model'):
                display = self.ndv_viewer.display_model
                
                if hasattr(display, 'luts'):
                    luts = display.luts
                    
                    for channel_idx, lut_model in luts.items():
                        if channel_idx is None:
                            continue  # Skip fallback LUT
                        
                        # Get current contrast limits from clims
                        if hasattr(lut_model, 'clims'):
                            clims = lut_model.clims
                            
                            # Check if it's manual clims (user adjusted)
                            if hasattr(clims, 'clim_type'):
                                if clims.clim_type == 'manual' and hasattr(clims, 'min') and hasattr(clims, 'max'):
                                    vmin, vmax = float(clims.min), float(clims.max)
                                    
                                    # Check if changed
                                    last = self._last_contrast.get(channel_idx)
                                    if last is None or last != (vmin, vmax):
                                        self._last_contrast[channel_idx] = (vmin, vmax)
                                        self.contrast_changed.emit(channel_idx, vmin, vmax)
                                else:
                                    # Auto mode
                                    last = self._last_contrast.get(channel_idx)
                                    if last is not None:
                                        self._last_contrast[channel_idx] = None
                                        self.contrast_changed.emit(channel_idx, -1.0, -1.0)
        except Exception as e:
            if not hasattr(self, '_error_printed'):
                print(f"[Contrast Sync Error] {e}")
                import traceback
                traceback.print_exc()
                self._error_printed = True
    
    def disconnect(self):
        """Stop polling"""
        if self._timer:
            self._timer.stop()
        self._connected = False



