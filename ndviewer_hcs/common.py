"""Shared constants, utilities, and data structures for ndviewer_hcs"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence
import numpy as np

# Constants
WELL_FORMATS = {4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12), 384: (16, 24), 1536: (32, 48)}
COLOR_MAPS = {'405': 'blue', '488': 'green', '561': 'yellow', '638': 'red', '640': 'red'}
COLOR_WEIGHTS = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1], 'yellow': [1, 1, 0], 'darkred': [0.5, 0, 0], 'gray': [0.5, 0.5, 0.5]}

# Filename patterns
fpattern = re.compile(
    r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.tiff?", re.IGNORECASE
)
fpattern_ome = re.compile(
    r"(?P<r>[^_]+)_(?P<f>\d+)\.ome\.tiff?", re.IGNORECASE
)

@dataclass
class TileData:
    image: np.ndarray
    x_mm: float
    y_mm: float
    file_path: str
    region: str
    fov: int
    wavelength: str

def parse_filenames(filenames: Sequence[str]) -> tuple:
    """Parse a sequence of TIFF file paths to extract multi-dimensional metadata."""
    parsed: List[Tuple[str, Tuple[int, str, int, int, str]]] = []

    for fname in filenames:
        path = Path(fname)
        try:
            t = int(path.parent.name)
        except ValueError as err:
            raise ValueError(f"Cannot parse time from parent folder of {fname}") from err

        if m := fpattern.search(path.name):
            region = m.group("r")
            fov = int(m.group("f"))
            z_level = int(m.group("z"))
            channel = m.group("c")
            parsed.append((fname, (t, region, fov, z_level, channel)))

    parsed.sort(key=lambda x: x[1])
    sorted_files = [p[0] for p in parsed]
    metadata = [p[1] for p in parsed]

    times = sorted({md[0] for md in metadata})
    regions = sorted({md[1] for md in metadata})
    fovs = sorted({md[2] for md in metadata})
    z_levels = sorted({md[3] for md in metadata})
    channels = sorted({md[4] for md in metadata})

    axes = ("time", "region", "fov", "z_level", "channel")
    shape = (len(times), len(regions), len(fovs), len(z_levels), len(channels))

    indices = []
    for md in metadata:
        t_idx = times.index(md[0])
        r_idx = regions.index(md[1])
        f_idx = fovs.index(md[2])
        z_idx = z_levels.index(md[3])
        c_idx = channels.index(md[4])
        indices.append((t_idx, r_idx, f_idx, z_idx, c_idx))

    return axes, shape, indices, sorted_files

def detect_acquisition_format(base_path: Path) -> str:
    """Detect if acquisition uses single-TIFF or OME-TIFF format"""
    # Check for ome_tiff/ directory first
    ome_dir = base_path / "ome_tiff"
    if ome_dir.exists() and ome_dir.is_dir():
        has_ome = any(f.suffix in ['.tif', '.tiff'] and '.ome' in f.name for f in ome_dir.glob('*.tif*'))
        if has_ome:
            return 'ome_tiff'
    
    # Fallback: check in timepoint directories
    first_tp = next((d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()), None)
    if first_tp:
        has_ome = any(f.suffix in ['.tif', '.tiff'] and '.ome' in f.name for f in first_tp.glob('*.tif*'))
        return 'ome_tiff' if has_ome else 'single_tiff'
    return 'single_tiff'

def extract_region_name_from_path(file_path: Path) -> str:
    """
    Extract region name from file/directory path using common patterns.
    
    Args:
        file_path: Path to file or directory
        
    Returns:
        Extracted region name (e.g., 'A1', 'B12', etc.)
    """
    if isinstance(file_path, str):
        file_path = Path(file_path)
    
    name = file_path.stem  # Get filename without extension
    
    # Remove common suffixes to get clean region name
    suffixes_to_remove = ['_stitched', '_processed', '_merged', '.ome']
    for suffix in suffixes_to_remove:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
    
    # Handle common patterns:
    # Pattern 1: "A1_0.ome.tif" -> "A1" (region_fov format)
    match = re.match(r'^([A-Z]\d+)_\d+', name)
    if match:
        return match.group(1)
    
    # Pattern 2: Direct region name "A1", "B12", etc.
    match = re.match(r'^([A-Z]+\d+)$', name)
    if match:
        return match.group(1)
    
    # Pattern 3: Longer names with region "RegionA1_something" -> "A1"
    match = re.search(r'([A-Z]+\d+)', name)
    if match:
        return match.group(1)
    
    # Return the cleaned name if no specific pattern matches
    return name

def detect_hcs_vs_normal_tissue(dataset_directory_path: Path) -> bool:
    """
    Detect whether a dataset is HCS (wellplate) or normal tissue based on:
    1. Timepoint directory structure patterns
    2. Region naming conventions (wellplate patterns)
    3. Multiple timepoint/region analysis
    
    Works with any file format (OME-TIFF, TIFF, etc.)
    
    Args:
        dataset_directory_path: Path to the dataset directory
        
    Returns:
        True if HCS/wellplate dataset, False if normal tissue
    """
    if isinstance(dataset_directory_path, str):
        dataset_directory_path = Path(dataset_directory_path)
    
    # STEP 1: Extract dataset directory from output path
    dataset_dir = dataset_directory_path
    
    # Navigate to dataset root if currently in a subdirectory
    if dataset_dir.name.endswith("_stitched") or dataset_dir.name.isdigit():
        dataset_dir = dataset_dir.parent
    
    if not dataset_dir.exists():
        print(f"Warning: Dataset directory does not exist: {dataset_dir}")
        return False
    
    # STEP 2: Find timepoint directories with strict pattern matching
    timepoint_dirs = []
    for item in dataset_dir.iterdir():
        if (item.is_dir() and 
            re.match(r'^\d+(_stitched)?$', item.name)):  # Match "0", "1" or "0_stitched", "1_stitched"
            timepoint_dirs.append(item)
    
    if not timepoint_dirs:
        print(f"No timepoint directories found in {dataset_dir}")
        return False  # No timepoint directories = not HCS
    
    # STEP 3: Analyze files and check for wellplate naming patterns
    wellplate_regions = set()
    
    for timepoint_dir in timepoint_dirs:
        # Look for image files (OME-TIFF, TIFF, etc.)
        for item in timepoint_dir.iterdir():
            if item.is_file() and item.suffix.lower() in ['.tiff', '.tif', '.ome.tiff', '.ome.tif']:
                region_name = extract_region_name_from_path(item)
                
                if not region_name or region_name.startswith("._"):
                    continue
                
                # KEY PATTERN: Check for wellplate naming (letter(s) + number(s))
                # Examples: A1, B2, C12, AA1, AB24, etc.
                if re.match(r'^([A-Z]+)(\d+)$', region_name):
                    wellplate_regions.add(region_name)
    
    # STEP 4: Final HCS determination logic
    # HCS dataset ONLY if we found wellplate naming patterns (e.g., A1, B2, C12)
    # Wellplate naming is the ONLY reliable indicator of an HCS dataset
    is_hcs = len(wellplate_regions) > 0
    
    # Log detection results
    print(f"[HCS Detection] Dataset: {dataset_dir.name}")
    print(f"  - Wellplate regions detected: {len(wellplate_regions)} {list(wellplate_regions)[:5]}")
    print(f"  - Result: {'HCS/Wellplate' if is_hcs else 'Normal Tissue'}")
    
    return is_hcs

