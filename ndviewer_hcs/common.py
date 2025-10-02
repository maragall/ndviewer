"""Shared constants, utilities, and data structures for ndviewer_hcs"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Sequence
import numpy as np

# Constants
WELL_FORMATS = {4: (2, 2), 6: (2, 3), 12: (3, 4), 24: (4, 6), 96: (8, 12), 384: (16, 24), 1536: (32, 48)}
COLOR_MAPS = {'405': 'blue', '488': 'green', '561': 'yellow', '638': 'red', '640': 'red'}
COLOR_WEIGHTS = {'red': [1, 0, 0], 'green': [0, 1, 0], 'blue': [0, 0, 1], 'yellow': [1, 1, 0], 'gray': [0.5, 0.5, 0.5]}

# Filename pattern
fpattern = re.compile(
    r"(?P<r>[^_]+)_(?P<f>\d+)_(?P<z>\d+)_(?P<c>.+)\.tiff?", re.IGNORECASE
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

class ImageProcessor:
    @staticmethod
    def downsample_fast(img: np.ndarray, target_size: int) -> np.ndarray:
        h, w = img.shape[:2]
        if h <= target_size and w <= target_size:
            result = np.zeros((target_size, target_size) + img.shape[2:], dtype=img.dtype)
            result[:min(h, target_size), :min(w, target_size)] = img[:min(h, target_size), :min(w, target_size)]
            return result
        
        factor = max(h, w) / target_size
        if factor > 4:
            kernel = int(factor)
            crop_h, crop_w = (h // kernel) * kernel, (w // kernel) * kernel
            cropped = img[:crop_h, :crop_w]
            if len(img.shape) == 3:
                reshaped = cropped.reshape(crop_h // kernel, kernel, crop_w // kernel, kernel, img.shape[2])
                return reshaped.mean(axis=(1, 3)).astype(img.dtype)
            else:
                reshaped = cropped.reshape(crop_h // kernel, kernel, crop_w // kernel, kernel)
                return reshaped.mean(axis=(1, 3)).astype(img.dtype)
        
        step = max(1, int(max(h, w) / target_size))
        return img[::step, ::step]

