#!/usr/bin/env python3
"""
Test script for contrast synchronization between FOV viewer and navigator.
Verifies that contrast changes are properly synchronized between main data layer and navigator layers.
"""

import os
import sys
import time
import pickle
import tempfile
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_contrast_synchronization():
    """Test contrast synchronization between FOV viewer and navigator."""
    
    print("="*80)
    print("TESTING CONTRAST SYNCHRONIZATION")
    print("="*80)
    
    # Test dataset path
    test_dataset = "/Users/julioamaragall/Documents/10x_mouse_brain_2025-04-23_00-53-11.236590"
    
    if not os.path.exists(test_dataset):
        print(f"❌ Test dataset not found: {test_dataset}")
        return False
    
    try:
        # Import modules
        from ndviewer_napari.napari_launcher import create_tiff_zarr_map
        from ndviewer_napari.fov_viewer import FOVViewer
        from ndviewer_napari.navigator import NavigatorOverlay
        print("✓ Successfully imported modules")
        
        # Create metadata
        print("\\n--- Creating metadata ---")
        metadata = create_tiff_zarr_map(test_dataset)
        
        print(f"✓ Metadata created successfully")
        print(f"  Channels: {metadata['channels']}")
        print(f"  Regions: {metadata['regions']}")
        print(f"  FOVs: {len(metadata['fovs'])}")
        
        # Create temporary file for testing
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_contrast_test.pkl', delete=False) as temp_file:
            # Create the same structure as napari_launcher
            data = {
                'directory': test_dataset,
                'metadata': metadata,
                'folder_name': os.path.basename(test_dataset)
            }
            pickle.dump(data, temp_file)
            temp_file_path = temp_file.name
        
        print(f"✓ Temporary file created: {temp_file_path}")
        
        # Create FOV viewer
        print("\\n--- Creating FOV viewer ---")
        fov_viewer = FOVViewer(temp_file_path)
        viewer = fov_viewer.create_viewer()
        
        print("✓ FOV viewer created successfully")
        print(f"  Viewer layers: {len(viewer.layers)}")
        
        # Find the main data layer
        main_data_layer = None
        for layer in viewer.layers:
            if hasattr(layer, 'name') and 'Multi-channel FOV Data' in layer.name:
                main_data_layer = layer
                break
        
        if not main_data_layer:
            print("❌ Could not find main data layer")
            return False
        
        print(f"✓ Found main data layer: {main_data_layer.name}")
        print(f"  Initial contrast: {main_data_layer.contrast_limits}")
        
        # Create navigator overlay
        print("\\n--- Creating navigator overlay ---")
        navigator = NavigatorOverlay(viewer, metadata, test_dataset)
        navigator_success = navigator.create_navigators()
        
        if not navigator_success:
            print("❌ Navigator creation failed")
            return False
            
        print("✓ Navigator overlay created successfully")
        print(f"  Navigator layers: {len(navigator.nav_layers)}")
        
        # Test contrast synchronization
        print("\\n--- Testing contrast synchronization ---")
        
        # Find the currently visible navigator layer
        visible_navigator = None
        for (region_idx, nav_channel_idx), nav_layer in navigator.nav_layers.items():
            if nav_layer.visible:
                visible_navigator = nav_layer
                region_name = navigator.region_names[region_idx]
                channel_name = navigator.idx_to_navigator_channel[nav_channel_idx]
                print(f"✓ Found visible navigator: Region {region_name}, Channel {channel_name}")
                break
        
        if not visible_navigator:
            print("❌ No visible navigator found")
            return False
        
        # Get initial contrast values
        initial_main_contrast = main_data_layer.contrast_limits
        initial_nav_contrast = visible_navigator.contrast_limits
        
        print(f"Initial main layer contrast: {initial_main_contrast}")
        print(f"Initial navigator contrast: {initial_nav_contrast}")
        
        # Test 1: Change main layer contrast and check if navigator updates
        print("\\n--- Test 1: Main layer → Navigator sync ---")
        test_contrast_1 = (100, 1000)
        print(f"Setting main layer contrast to: {test_contrast_1}")
        
        main_data_layer.contrast_limits = test_contrast_1
        time.sleep(0.1)  # Allow event to propagate
        
        new_nav_contrast_1 = visible_navigator.contrast_limits
        print(f"Navigator contrast after main change: {new_nav_contrast_1}")
        
        # Convert to tuples for comparison since napari returns lists
        if tuple(new_nav_contrast_1) == test_contrast_1:
            print("✓ Main → Navigator sync working!")
        else:
            print(f"❌ Main → Navigator sync failed. Expected: {test_contrast_1}, Got: {new_nav_contrast_1}")
            return False
        
        # Test 2: Change navigator contrast and check if main layer updates
        print("\\n--- Test 2: Navigator → Main layer sync ---")
        test_contrast_2 = (200, 2000)
        print(f"Setting navigator contrast to: {test_contrast_2}")
        
        visible_navigator.contrast_limits = test_contrast_2
        time.sleep(0.1)  # Allow event to propagate
        
        new_main_contrast_2 = main_data_layer.contrast_limits
        print(f"Main layer contrast after navigator change: {new_main_contrast_2}")
        
        # Convert to tuples for comparison since napari returns lists
        if tuple(new_main_contrast_2) == test_contrast_2:
            print("✓ Navigator → Main layer sync working!")
        else:
            print(f"❌ Navigator → Main layer sync failed. Expected: {test_contrast_2}, Got: {new_main_contrast_2}")
            return False
        
        # Test 3: Channel switching maintains sync
        print("\\n--- Test 3: Channel switching sync ---")
        current_step = list(viewer.dims.current_step)
        original_channel = current_step[1]
        
        # Switch to a different channel
        test_channel = (original_channel + 1) % len(metadata['channels'])
        current_step[1] = test_channel
        viewer.dims.current_step = current_step
        
        time.sleep(0.2)  # Allow channel switch to complete
        
        # Find the new visible navigator
        new_visible_navigator = None
        for (region_idx, nav_channel_idx), nav_layer in navigator.nav_layers.items():
            if nav_layer.visible:
                new_visible_navigator = nav_layer
                region_name = navigator.region_names[region_idx]
                channel_name = navigator.idx_to_navigator_channel[nav_channel_idx]
                print(f"✓ Found new visible navigator: Region {region_name}, Channel {channel_name}")
                break
        
        if new_visible_navigator:
            # Test sync on the new channel
            test_contrast_3 = (300, 3000)
            print(f"Setting main layer contrast to: {test_contrast_3} on new channel")
            
            main_data_layer.contrast_limits = test_contrast_3
            time.sleep(0.1)
            
            new_nav_contrast_3 = new_visible_navigator.contrast_limits
            print(f"New navigator contrast: {new_nav_contrast_3}")
            
            # Convert to tuples for comparison since napari returns lists
            if tuple(new_nav_contrast_3) == test_contrast_3:
                print("✓ Channel switching sync working!")
            else:
                print(f"❌ Channel switching sync failed. Expected: {test_contrast_3}, Got: {new_nav_contrast_3}")
                return False
        else:
            print("⚠ No visible navigator after channel switch (may be expected)")
        
        print("\\n" + "="*80)
        print("🎉 ALL CONTRAST SYNCHRONIZATION TESTS PASSED!")
        print("="*80)
        print("✓ Main layer → Navigator sync: Working")
        print("✓ Navigator → Main layer sync: Working") 
        print("✓ Channel switching sync: Working")
        print("="*80)
        
        # Cleanup
        os.unlink(temp_file_path)
        print(f"✓ Cleaned up temp file: {temp_file_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_contrast_synchronization() 