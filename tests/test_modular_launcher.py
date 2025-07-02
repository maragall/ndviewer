#!/usr/bin/env python3

"""
Test script for the modular napari launcher
Tests that the launcher and viewer script work together correctly
"""

import os
import sys
import tempfile
import pickle
import subprocess
from pathlib import Path

def test_modular_launcher():
    """Test the modular launcher setup"""
    
    print("="*60)
    print("TESTING MODULAR NAPARI LAUNCHER")
    print("="*60)
    
    # Test dataset path
    test_dataset = "/Users/julioamaragall/Documents/10x_mouse_brain_2025-04-23_00-53-11.236590"
    
    if not os.path.exists(test_dataset):
        print("❌ Test dataset not found. Please update the path.")
        return False
    
    print(f"✓ Test dataset found: {test_dataset}")
    
    # Add parent directory to path for imports
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ndviewer_napari'))
    
    # Test imports
    try:
        from napari_launcher import get_zarr_store_with_lazy_tiff_mapping
        print("✓ napari_launcher imports working")
    except ImportError as e:
        print(f"❌ napari_launcher import failed: {e}")
        return False
    
    try:
        from downsampler import DownsampledNavigator
        print("✓ downsampler imports working")
    except ImportError as e:
        print(f"❌ downsampler import failed: {e}")
        return False
    
    # Test metadata creation
    try:
        print("\n--- Testing metadata creation ---")
        metadata = get_zarr_store_with_lazy_tiff_mapping(test_dataset)
        
        print(f"✓ Metadata created successfully")
        print(f"  Dimensions: {metadata['dimensions']}")
        print(f"  Regions: {metadata['regions']}")
        print(f"  Channels: {metadata['channels']}")
        print(f"  FOVs: {len(metadata['fovs'])}")
        
    except Exception as e:
        print(f"❌ Metadata creation failed: {e}")
        return False
    
    # Test downsampler creation
    try:
        print("\n--- Testing downsampler creation ---")
        navigator = DownsampledNavigator(
            Path(test_dataset), 
            tile_size=75,
            cache_enabled=True
        )
        print("✓ DownsampledNavigator created successfully")
        
        # Test multi-channel mosaic creation for first region
        if metadata['regions']:
            first_region = metadata['regions'][0]
            print(f"  Testing multi-channel mosaic for region '{first_region}'")
            
            channel_mosaics, nav_metadata = navigator.create_channel_mosaics_for_region(first_region, 0)
            
            print(f"✓ Multi-channel mosaics created: {len(channel_mosaics)} channels")
            for ch_name, mosaic in channel_mosaics.items():
                print(f"    {ch_name}: {mosaic.shape}, non-zero pixels: {(mosaic > 0).sum()}")
                
        else:
            print("⚠ No regions found to test")
            
    except Exception as e:
        print(f"❌ Downsampler creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test temporary file creation (like the launcher does)
    try:
        print("\n--- Testing temp file creation ---")
        temp_dir = tempfile.gettempdir()
        folder_name = os.path.basename(os.path.normpath(test_dataset))
        temp_file = os.path.join(temp_dir, f"ndv_test_{os.getpid()}_{hash(folder_name)}.pkl")
        
        with open(temp_file, 'wb') as f:
            pickle.dump({
                'directory': test_dataset,
                'metadata': metadata,
                'folder_name': folder_name
            }, f)
        
        print(f"✓ Temp file created: {temp_file}")
        
        # Test loading the temp file
        with open(temp_file, 'rb') as f:
            test_data = pickle.load(f)
        
        print("✓ Temp file can be loaded successfully")
        print(f"  Contains: directory, metadata, folder_name")
        
        # Cleanup
        os.remove(temp_file)
        print("✓ Temp file cleaned up")
        
    except Exception as e:
        print(f"❌ Temp file handling failed: {e}")
        return False
    
    # Test napari_viewer_script existence
    viewer_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ndviewer_napari', 'napari_viewer_script.py')
    if os.path.exists(viewer_script):
        print(f"✓ Viewer script found: {viewer_script}")
    else:
        print(f"❌ Viewer script not found: {viewer_script}")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED - Modular launcher is working correctly!")
    print("="*60)
    
    return True

def test_napari_launch():
    """Test actually launching napari (optional)"""
    
    print("\n--- Optional: Test napari launch ---")
    response = input("Launch napari to test the full modular setup? (y/n): ").lower().strip()
    
    if response == 'y':
        print("Launching napari...")
        launcher_script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ndviewer_napari', 'napari_launcher.py')
        try:
            subprocess.run([sys.executable, launcher_script], check=False)
            print("✓ Napari launcher executed (check GUI)")
        except Exception as e:
            print(f"❌ Napari launch failed: {e}")
            return False
    else:
        print("Skipping napari launch test")
    
    return True

if __name__ == "__main__":
    success = test_modular_launcher()
    
    if success:
        test_napari_launch()
    else:
        print("\n❌ Tests failed. Please fix issues before proceeding.")
        sys.exit(1) 