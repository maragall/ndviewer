"""Preprocessing: plate assembly and tile stitching"""

import pickle
import random
from pathlib import Path
from typing import Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import io

from .common import TileData, COLOR_MAPS, detect_acquisition_format, fpattern_ome, extract_wavelength
import tifffile as tf


class ImageProcessor:
    """Fast image downsampling utilities for preprocessing"""
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


class PlateAssembler:
    """Assembles full plate from tiles with optional preprocessing"""
    
    def __init__(self, base_path: str, timepoint: int = 0):
        self.base_path = Path(base_path)
        self.timepoint = timepoint
        
        # Create downsampled_image folder for outputs
        self.output_dir = self.base_path / "downsampled_image"
        self.output_dir.mkdir(exist_ok=True)
        
        # Cache directory inside downsampled_image
        self.cache_dir = self.output_dir / "assembled_tiles_cache"
        self.cache_dir.mkdir(exist_ok=True)
    
    def assemble_plate(self, 
                      target_pixel_size: int,
                      output_dir: str = None) -> Tuple[Dict, Dict]:  # output_dir kept for compatibility but ignored
        """Assemble plate with downsampling
        
        Returns:
            assembled_images: dict with 'multichannel', 'wavelengths', 'colormaps'
            tile_map: lightweight dict with tile_dimensions, grid_to_tile, fov_to_files
        """
        downsample_factor = target_pixel_size / self._get_original_pixel_size() if target_pixel_size else 0.85
        skip_downsampling = downsample_factor >= 0.98
        
        # Check cache
        cache_key = self._get_cache_key(downsample_factor, skip_downsampling)
        if self._cache_exists(cache_key):
            print(f"Loading from cache: {cache_key}")
            return self._load_from_cache(cache_key)
        
        # Load and process tiles
        tiles = self._load_tiles(downsample_factor, skip_downsampling)
        assembled_images, tile_map = self._assemble_images(tiles)
        
        if assembled_images:
            self._save_to_cache(cache_key, assembled_images, tile_map)
            
            # Save target pixel size to acquisition folder
            self._save_metadata(target_pixel_size)
        
        return assembled_images, tile_map
    
    def _get_original_pixel_size(self) -> int:
        """Get original image pixel size from first image"""
        timepoint_path = self.base_path / str(self.timepoint)
        for filepath in timepoint_path.glob("*.tiff"):
            if filepath.stem != "coordinates":
                img = io.imread(filepath)
                return min(img.shape[:2])
        return 2048  # fallback
    
    def _get_cache_key(self, downsample_factor: float, skip_downsampling: bool) -> str:
        suffix = "original" if skip_downsampling else f"ds{downsample_factor:.4f}"
        return f"t{self.timepoint}_{suffix}"
    
    def _cache_exists(self, cache_key: str) -> bool:
        return all((self.cache_dir / f"{cache_key}_{name}").exists()
                  for name in ["tile_map.pkl", "multichannel.tiff", "metadata.pkl"])
    
    def _load_from_cache(self, cache_key: str) -> Tuple[Dict, Dict]:
        with open(self.cache_dir / f"{cache_key}_tile_map.pkl", 'rb') as f:
            tile_map = pickle.load(f)
        # Use tifffile to preserve (C, H, W) axis order
        multichannel_image = tf.imread(self.cache_dir / f"{cache_key}_multichannel.tiff")
        
        with open(self.cache_dir / f"{cache_key}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        # Ensure wavelengths are integers (for backward compatibility with old caches)
        wavelengths = metadata['wavelengths']
        if wavelengths:
            if isinstance(wavelengths[0], str):
                wavelengths = [extract_wavelength(wl) for wl in wavelengths]
            elif not isinstance(wavelengths[0], int):
                # Handle any other type by converting to int
                wavelengths = [int(wl) if isinstance(wl, (int, float)) else extract_wavelength(str(wl)) for wl in wavelengths]
        
        return {
            'multichannel': multichannel_image,
            'wavelengths': wavelengths,
            'colormaps': metadata['colormaps']
        }, tile_map
    
    def _save_to_cache(self, cache_key: str, assembled_images: Dict, tile_map: Dict):
        # Save to cache directory for fast loading
        with open(self.cache_dir / f"{cache_key}_tile_map.pkl", 'wb') as f:
            pickle.dump(tile_map, f)
        
        multichannel_image = assembled_images['multichannel']
        
        # Use tifffile.imwrite to preserve (C, H, W) axis order
        compression = 'lzw' if multichannel_image.dtype == np.uint16 else None
        tf.imwrite(self.cache_dir / f"{cache_key}_multichannel.tiff", multichannel_image, compression=compression, photometric='minisblack')
        
        metadata = {k: assembled_images[k] for k in ['wavelengths', 'colormaps']}
        with open(self.cache_dir / f"{cache_key}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        # ALSO save downsampled plate to downsampled_image folder
        plate_tiff_path = self.output_dir / "assembled_plate_downsampled.tiff"
        tf.imwrite(plate_tiff_path, multichannel_image, compression=compression, photometric='minisblack')
        print(f"Saved downsampled plate to: {plate_tiff_path}")
    
    def _save_metadata(self, target_pixel_size: int):
        """Save target pixel size to text file in downsampled_image folder"""
        output_path = self.output_dir / "downsampled_pixel_size.txt"
        with open(output_path, 'w') as f:
            f.write(f"{target_pixel_size}\n")
        print(f"Saved target pixel size ({target_pixel_size}) to: {output_path}")
    
    def _parse_filename(self, filepath: Path) -> Optional[Tuple]:
        import re
        match = re.match(r"([A-Z]+\d+)_(\d+)_(\d+)_.*_(\d+)_nm_Ex\.tiff", filepath.name)
        return match.groups() if match else None
    
    def _get_colormap(self, wavelength: int) -> str:
        """Get colormap for channel based on wavelength (nm).
        
        Mapping:
        - 405nm -> blue
        - 488nm -> green
        - 561nm -> yellow
        - 638/640nm -> red
        - 730nm -> darkred
        """
        if wavelength <= 420:
            return 'blue'
        elif 470 <= wavelength <= 510:
            return 'green'
        elif 540 <= wavelength <= 590:
            return 'yellow'
        elif 620 <= wavelength <= 660:
            return 'red'
        elif wavelength >= 700:
            return 'darkred'
        else:
            return 'gray'
    
    def _load_tiles(self, downsample_factor: float, skip_downsampling: bool, z_level: int = 0) -> Dict:
        """Load tiles with downsampling"""
        format_type = detect_acquisition_format(self.base_path)
        
        if format_type == 'ome_tiff':
            return self._load_tiles_from_ome(downsample_factor, skip_downsampling, z_level)
        else:
            return self._load_tiles_from_single_tiff(downsample_factor, skip_downsampling)
    
    def _load_tiles_from_single_tiff(self, downsample_factor: float, skip_downsampling: bool) -> Dict:
        """Load tiles from single-TIFF files"""
        timepoint_path = self.base_path / str(self.timepoint)
        coords_df = pd.read_csv(timepoint_path / "coordinates.csv")
        tiles = {}
        
        for filepath in timepoint_path.glob("*.tiff"):
            parsed = self._parse_filename(filepath)
            if not parsed:
                continue
            
            region, fov, z, wavelength_str = parsed
            wavelength = int(wavelength_str)  # Convert to int
            coord_row = coords_df[(coords_df['region'] == region) & (coords_df['fov'] == int(fov))]
            if coord_row.empty:
                continue
            
            img = io.imread(filepath)
            
            # Downsample if requested
            if not skip_downsampling and downsample_factor < 0.98:
                target_size = int(min(img.shape[:2]) * downsample_factor)
                if target_size < min(img.shape[:2]) - 1:
                    img = ImageProcessor.downsample_fast(img, target_size)
            
            tiles[(region, int(fov), wavelength)] = TileData(
                image=img,
                x_mm=coord_row['x (mm)'].iloc[0],
                y_mm=coord_row['y (mm)'].iloc[0],
                file_path=str(filepath),
                region=region,
                fov=int(fov),
                wavelength=wavelength
            )
        
        return tiles
    
    def _load_tiles_from_ome(self, downsample_factor: float, skip_downsampling: bool, z_level: int = 0) -> Dict:
        """Load tiles from OME-TIFF files for specific z-level using bioio"""
        from bioio import BioImage
        
        # For OME-TIFF, files are in ome_tiff/ directory or fallback to 0/
        ome_dir = self.base_path / "ome_tiff" if (self.base_path / "ome_tiff").exists() else self.base_path / "0"
        coords_path = self.base_path / "0" / "coordinates.csv"
        coords_df = pd.read_csv(coords_path)
        tiles = {}
        
        for ome_file in ome_dir.glob("*.ome.tif*"):
            if not (m := fpattern_ome.search(ome_file.name)):
                continue
            
            region, fov = m.group("r"), int(m.group("f"))
            
            bio_img = None
            try:
                # Load with bioio - no need to worry about dimension order!
                bio_img = BioImage(str(ome_file))
                
                # Get dimensions
                n_channels = bio_img.dims.C
                n_z = bio_img.dims.Z
                n_t = bio_img.dims.T
                
                # Extract actual channel names from bioio
                channel_names = []
                try:
                    if hasattr(bio_img, 'channel_names') and bio_img.channel_names:
                        channel_names = list(bio_img.channel_names)
                except:
                    pass
                
                # Check if requested indices are valid
                if z_level >= n_z or self.timepoint >= n_t:
                    del bio_img
                    continue
                
                # Find z_level from coordinates
                coord_row = coords_df[(coords_df['region'] == region) & 
                                     (coords_df['fov'] == fov) & 
                                     (coords_df['z_level'] == z_level)]
                if coord_row.empty:
                    del bio_img
                    continue
                
                # Extract each channel at the specific z and t
                for c_idx in range(n_channels):
                    # bioio handles all dimension ordering automatically!
                    img = bio_img.get_image_data("YX", T=self.timepoint, C=c_idx, Z=z_level)
                    
                    # Use actual channel names from OME metadata if available
                    channel_name = channel_names[c_idx] if c_idx < len(channel_names) else f'Channel_{c_idx}'
                    wavelength = extract_wavelength(channel_name)
                    
                    # Downsample if requested
                    if not skip_downsampling and downsample_factor < 0.98:
                        target_size = int(min(img.shape[:2]) * downsample_factor)
                        if target_size < min(img.shape[:2]) - 1:
                            img = ImageProcessor.downsample_fast(img, target_size)
                    
                    tiles[(region, fov, wavelength)] = TileData(
                        image=img,
                        x_mm=coord_row['x (mm)'].iloc[0],
                        y_mm=coord_row['y (mm)'].iloc[0],
                        file_path=str(ome_file),
                        region=region,
                        fov=fov,
                        wavelength=wavelength
                    )
                
                # Release reference after extracting all channels
                del bio_img
                
            except Exception as e:
                print(f"Error processing {ome_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return tiles
    
    def _assemble_images(self, tiles: Dict) -> Tuple[Dict, Dict]:
        """Assemble tiles into multichannel canvas with lightweight tile mapping
        
        Returns:
            assembled_images: dict with multichannel canvas, wavelengths, colormaps
            tile_map: lightweight dict with tile_dimensions, grid_to_tile, fov_to_files
        """
        if not tiles:
            return {}, {}
        
        sample_img = next(iter(tiles.values())).image
        tile_h, tile_w = sample_img.shape[:2]
        
        unique_coords = {
            'x': sorted(set(t.x_mm for t in tiles.values())),
            'y': sorted(set(t.y_mm for t in tiles.values()))
        }
        coord_to_grid = {
            'x': {x: i for i, x in enumerate(unique_coords['x'])},
            'y': {y: i for i, y in enumerate(unique_coords['y'])}
        }
        
        canvas_w = len(unique_coords['x']) * tile_w
        canvas_h = len(unique_coords['y']) * tile_h
        wavelengths = sorted(set(key[2] for key in tiles.keys()))
        canvas = np.zeros((len(wavelengths), canvas_h, canvas_w), dtype=sample_img.dtype)
        
        # Build lightweight tile map instead of per-pixel map
        tile_map = {
            'tile_dimensions': (tile_h, tile_w),
            'grid_to_tile': {},      # (grid_x, grid_y) -> tile metadata dict
            'fov_to_files': {}       # (region, fov) -> {wavelength: file_path}
        }
        
        for channel_idx, wavelength in enumerate(wavelengths):
            for (region, fov, wl), tile in tiles.items():
                if wl != wavelength:
                    continue
                
                x_pixel = coord_to_grid['x'][tile.x_mm] * tile_w
                y_pixel = coord_to_grid['y'][tile.y_mm] * tile_h
                canvas[channel_idx, y_pixel:y_pixel+tile_h, x_pixel:x_pixel+tile_w] = tile.image
                
                # Store tile metadata once per grid position (not per pixel!)
                if channel_idx == 0:
                    grid_x = coord_to_grid['x'][tile.x_mm]
                    grid_y = coord_to_grid['y'][tile.y_mm]
                    tile_map['grid_to_tile'][(grid_x, grid_y)] = {
                        'region': tile.region,
                        'fov': tile.fov,
                        'x_mm': tile.x_mm,
                        'y_mm': tile.y_mm,
                        'x_pixel': x_pixel,
                        'y_pixel': y_pixel
                    }
                
                # Build FOV file index for O(1) lookup
                key = (region, fov)
                if key not in tile_map['fov_to_files']:
                    tile_map['fov_to_files'][key] = {}
                tile_map['fov_to_files'][key][wl] = tile.file_path
        
        # Assign colors by ORDER: 1st=blue, 2nd=green, 3rd=yellow, 4th=red, 5th=darkred, 6th+=gray
        default_colors = ['blue', 'green', 'yellow', 'red', 'darkred']
        colormaps = []
        for i, wl in enumerate(wavelengths):
            # Try wavelength-based first
            color = self._get_colormap(wl)
            # If no wavelength match (gray), use order-based
            if color == 'gray':
                color = default_colors[i] if i < len(default_colors) else 'gray'
            colormaps.append(color)
        
        return {
            'multichannel': canvas,
            'wavelengths': wavelengths,
            'colormaps': colormaps
        }, tile_map

