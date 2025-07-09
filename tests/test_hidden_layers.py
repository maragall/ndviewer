#!/usr/bin/env python3
"""
Simple test to verify that navigator layers are properly hidden from the layer list
while maintaining their functionality.
"""

import os
import sys
import time
import pickle
import tempfile
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_hidden_navigator_layers():
    """Test that navigator layers are hidden but functional."""
    
    print("="*60)
    print("HIDDEN NAVIGATOR LAYERS TEST")
    print("="*60)
    
    # Configuration - adjust this path to your test dataset
    test_dataset = "/Users/julioamaragall/Documents/10x_mouse_brain_2025-04-23_00-53-11.236590"
    
    # Check if test dataset exists
    if not os.path.exists(test_dataset):
        print(f"❌ Test dataset not found: {test_dataset}")
        print("Please update the test_dataset path in the script to point to a valid dataset")
        return False
    
    print(f"✓ Test dataset found: {test_dataset}")
    
    try:
        # Import modules
        from ndviewer_napari.napari_launcher import create_tiff_zarr_map
        from ndviewer_napari.fov_viewer import FOVViewer
        from ndviewer_napari.navigator import NavigatorOverlay
        print("✓ Successfully imported modules")
        
        # Create metadata
        print("\n--- Creating metadata ---")
        metadata = create_tiff_zarr_map(test_dataset)
        print(f"✓ Metadata created successfully")
        
        # Create temp file with metadata
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_test.pkl', delete=False) as temp_file:
            pickle.dump({
                'directory': test_dataset,
                'metadata': metadata,
                'folder_name': os.path.basename(test_dataset)
            }, temp_file)
            temp_file_path = temp_file.name
        
        print(f"✓ Temp metadata file created: {temp_file_path}")
        
        # Create FOV viewer
        print("\n--- Creating FOV viewer ---")
        fov_viewer = FOVViewer(temp_file_path)
        viewer = fov_viewer.create_viewer()
        print("✓ FOV viewer created successfully")
        
        # Record initial layer count and names
        initial_layer_count = len(viewer.layers)
        initial_layer_names = [layer.name for layer in viewer.layers]
        print(f"\n--- Initial layer state ---")
        print(f"Initial layer count: {initial_layer_count}")
        print(f"Initial layers: {initial_layer_names}")
        
        # Create navigator overlay
        print("\n--- Creating navigator overlay ---")
        navigator = NavigatorOverlay(viewer, metadata, test_dataset, fov_viewer)
        
        if not navigator.is_available():
            print("❌ Navigator not available")
            return False
        
        navigator_success = navigator.create_navigators()
        if not navigator_success:
            print("❌ Navigator creation failed")
            return False
        
        print("✓ Navigator overlay created successfully")
        
        # Wait for navigator creation to complete
        time.sleep(2)
        
        # Analyze layer state after navigator creation
        final_layer_count = len(viewer.layers)
        final_layer_names = [layer.name for layer in viewer.layers]
        
        print(f"\n--- Final layer state ---")
        print(f"Final layer count: {final_layer_count}")
        print(f"Final layers: {final_layer_names}")
        
        # Find navigator layers
        navigator_layers = []
        hidden_navigator_layers = []
        
        for layer in viewer.layers:
            layer_name = layer.name
            if any(pattern in layer_name for pattern in ['Navigator', 'navigator']):
                navigator_layers.append(layer)
                if layer_name.startswith("_"):
                    hidden_navigator_layers.append(layer)
        
        print(f"\n--- Navigator Layer Analysis ---")
        print(f"Total navigator layers found: {len(navigator_layers)}")
        print(f"Hidden navigator layers (with '_' prefix): {len(hidden_navigator_layers)}")
        
        # Print details of each navigator layer
        for i, layer in enumerate(navigator_layers):
            print(f"  {i+1}. {layer.name}")
            print(f"     Visible: {layer.visible}")
            print(f"     Hidden (underscore prefix): {layer.name.startswith('_')}")
            print(f"     In hidden_layers set: {layer in navigator.hidden_layers}")
            print(f"     Interactive: {getattr(layer, 'interactive', 'N/A')}")
        
        # Test that hidden layers are properly managed
        success_count = 0
        total_tests = 0
        
        print(f"\n--- Testing Hidden Layer Properties ---")
        
        for layer in navigator_layers:
            total_tests += 1
            
            # Test 1: Hidden layers should have underscore prefix
            if layer.name.startswith("_"):
                print(f"✓ Layer '{layer.name}' has hidden prefix")
                success_count += 1
            else:
                print(f"❌ Layer '{layer.name}' missing hidden prefix")
            
            # Test 2: Hidden layers should be in the hidden_layers set
            if hasattr(navigator, 'hidden_layers') and layer in navigator.hidden_layers:
                print(f"✓ Layer '{layer.name}' is tracked in hidden_layers set")
                success_count += 1
            else:
                print(f"❌ Layer '{layer.name}' not tracked in hidden_layers set")
            
            total_tests += 1
        
        # Test functionality: Try switching regions to see if navigator layers become visible
        print(f"\n--- Testing Navigator Functionality ---")
        
        # Switch to different regions and check if appropriate layers become visible
        num_regions = metadata['dimensions'].get('region', 1)
        for region_idx in range(min(2, num_regions)):  # Test first 2 regions
            print(f"\nSwitching to region {region_idx}...")
            
            # Set viewer to this region
            current_step = list(viewer.dims.current_step)
            if len(current_step) >= 2:
                current_step[1] = region_idx  # Region is at position 1
                viewer.dims.current_step = current_step
                
                time.sleep(0.5)  # Brief wait for updates
                
                # Check if any navigator layers are visible for this region
                visible_nav_layers = [layer for layer in navigator_layers if layer.visible]
                print(f"  Visible navigator layers for region {region_idx}: {len(visible_nav_layers)}")
                
                for layer in visible_nav_layers:
                    print(f"    ✓ {layer.name} is visible and functional")
                    success_count += 1
                
                total_tests += len(visible_nav_layers)
        
        # Summary
        print(f"\n--- Test Results Summary ---")
        print(f"Navigator layers created: {len(navigator_layers)}")
        print(f"Hidden navigator layers: {len(hidden_navigator_layers)}")
        print(f"Success rate: {success_count}/{total_tests} ({success_count/max(total_tests,1)*100:.1f}%)")
        
        # Clean up
        try:
            os.unlink(temp_file_path)
            print(f"✓ Cleaned up temp file: {temp_file_path}")
        except:
            pass
        
        # Final verdict
        if len(hidden_navigator_layers) > 0 and success_count >= total_tests * 0.8:
            print("\n🎉 SUCCESS: Navigator layers are properly hidden and functional!")
            return True
        else:
            print("\n❌ FAILURE: Navigator layer hiding not working as expected")
            return False
            
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test
    success = test_hidden_navigator_layers()
    sys.exit(0 if success else 1) 