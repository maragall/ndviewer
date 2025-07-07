#!/usr/bin/env python3
"""
Simple Scaling Manager
Uses the straightforward formula to calculate FOV layer size.
"""
import math
from typing import Dict, List, Tuple, Optional


class ScalingManager:
    """Simple scaling manager using the FOV count formula."""
    
    def __init__(self, acquisition_parameters: Optional[Dict] = None):
        """Initialize with acquisition parameters."""
        self.acquisition_parameters = acquisition_parameters or {}
        self.z_step_um = self._calculate_z_step()
        
        # Constants from the formula
        self.downsampler_tile_size = 75  # 75x75
        self.desired_ratio = 3  # 5:1 ratio
        
        print(f"Simple ScalingManager initialized:")
        print(f"  Z step size: {self.z_step_um} µm")
        print(f"  Downsampler tile size: {self.downsampler_tile_size}")
        print(f"  Desired ratio: {self.desired_ratio}:1")
    
    def _calculate_z_step(self) -> float:
        """Calculate Z step size in microns from acquisition parameters."""
        z_step_um = None
        
        if self.acquisition_parameters:
            if 'dz(um)' in self.acquisition_parameters:
                z_step_um = self.acquisition_parameters['dz(um)']
                print(f"Z step size: {z_step_um} µm")
        
        if z_step_um is None:
            z_step_um = 1.0
            print("Warning: Using default Z step size of 1.0 µm")
        
        return z_step_um
    
    def calculate_fov_scale(self, fov_dims: Optional[Dict[str, int]] = None) -> List[float]:
        """
        Calculate FOV scale using the simple formula.
        
        Formula: FOV_layer_size = floor(sqrt(((number_of_fovs/number_of_regions)*downsampler_tile_size)*desired_ratio))
        
        Parameters:
        -----------
        fov_dims : dict, optional
            Dictionary containing 'x' and 'y' dimensions of FOV image
            
        Returns:
        --------
        List[float]: Scale array [time, region, fov, z, y, x]
        """
        # Import the global variable from downsampler
        try:
            from utils.downsampler import FIRST_FOV_COUNT
        except ImportError:
            print("Warning: Could not import FIRST_FOV_COUNT, using default")
            FIRST_FOV_COUNT = 610  # Default fallback
        
        if FIRST_FOV_COUNT is None:
            print("Warning: FIRST_FOV_COUNT is None, using default")
            FIRST_FOV_COUNT = 610  # Default fallback
        
        # Get number of regions (default to 1 if not available)
        num_regions = 1  # Default to single region
        
        # Calculate using the corrected formula
        # Navigator area = 75 * (75 * number_of_fovs) = 75 * 75 * number_of_fovs
        navigator_area = self.downsampler_tile_size * self.downsampler_tile_size * FIRST_FOV_COUNT
        fov_layer_size = math.floor(math.sqrt(navigator_area * self.desired_ratio))
        
        # Calculate scale to achieve this size
        if fov_dims and 'x' in fov_dims and 'y' in fov_dims:
            fov_width = fov_dims['x']
            fov_height = fov_dims['y']
        else:
            fov_width = 2048  # Default
            fov_height = 2048
        
        # Calculate scale to make FOV appear at the calculated size
        scale_x = fov_layer_size / fov_width
        scale_y = fov_layer_size / fov_height
        
        # Use uniform scaling
        uniform_scale = min(scale_x, scale_y)
        
        print(f"FOV Scale Calculation:")
        print(f"  First FOV count: {FIRST_FOV_COUNT}")
        print(f"  Navigator area: {navigator_area:,} pixels (75×75×{FIRST_FOV_COUNT})")
        print(f"  Formula result: {fov_layer_size} pixels")
        print(f"  FOV actual size: {fov_width}x{fov_height} pixels")
        print(f"  Required scale: {uniform_scale:.4f}")
        print(f"  FOV display size: {fov_width*uniform_scale:.1f}x{fov_height*uniform_scale:.1f} pixels")
        
        # Complete scale array: [time, region, fov, z, y, x]
        scale = [1.0, 1.0, 1.0, self.z_step_um, uniform_scale, uniform_scale]
        
        return scale
    
    def get_fov_xy_scales(self, fov_dims: Optional[Dict[str, int]] = None) -> Tuple[float, float]:
        """Get just the X and Y scales for FOV layers."""
        scale = self.calculate_fov_scale(fov_dims)
        return (scale[4], scale[5])  # y, x scales
    
    def get_navigator_scale(self) -> Tuple[float, float]:
        """Navigator scale - always 1:1."""
        return (1.0, 1.0)
    
    def get_navigator_tile_size(self) -> int:
        """Get the navigator tile size."""
        return self.downsampler_tile_size
    
    def should_enable_scale_bar(self) -> bool:
        """Determine if scale bar should be enabled."""
        return self.z_step_um is not None
    
    def get_scale_bar_unit(self) -> str:
        """Get the appropriate unit for the scale bar."""
        return 'µm'


# Convenience function
def create_scaling_manager(acquisition_parameters: Optional[Dict] = None) -> ScalingManager:
    """Create a scaling manager instance."""
    return ScalingManager(acquisition_parameters)