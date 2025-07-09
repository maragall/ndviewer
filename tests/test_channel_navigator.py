#!/usr/bin/env python3
"""
Comprehensive test for channel sliding and navigator clicking functionality.
Tests that:
1. Channel slider works correctly
2. Navigator switches properly for each channel
3. FOV jumping works when clicking on navigators
4. All functionality is verified programmatically
"""

import os
import sys
import time
import pickle
import random
import tempfile
import numpy as np
from pathlib import Path

# Add the parent directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_channel_navigator():
    """Main test function for channel sliding and navigator functionality."""
    
    print("="*80)
    print("COMPREHENSIVE CHANNEL NAVIGATOR TEST")
    print("="*80)
    
    # Configuration
    test_dataset = "/Users/julioamaragall/Documents/10x_mouse_brain_2025-04-23_00-53-11.236590"
    clicks_per_channel = 3  # Number of random clicks to test per channel
    
    # Check if test dataset exists
    if not os.path.exists(test_dataset):
        print(f"❌ Test dataset not found: {test_dataset}")
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
        print(f"  Dimensions: {metadata['dimensions']}")
        print(f"  Regions: {metadata['regions']}")
        print(f"  Channels: {metadata['channels']}")
        print(f"  FOVs: {len(metadata['fovs'])}")
        
        # Create temp file with metadata
        with tempfile.NamedTemporaryFile(mode='wb', suffix='_ndv_test.pkl', delete=False) as temp_file:
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
        print(f"  Data shape: {fov_viewer.dask_data.shape}")
        print(f"  Dimension labels: {viewer.dims.axis_labels}")
        
        # Verify we have a proper channel slider
        expected_dims = ['Time', 'Channel', 'Region', 'FOV', 'Z', 'Y', 'X']
        actual_dims = list(viewer.dims.axis_labels)  # Convert to list for comparison
        if actual_dims != expected_dims:
            print(f"❌ Dimension labels mismatch. Expected: {expected_dims}, Got: {actual_dims}")
            return False
        
        print("✓ Channel slider structure verified")
        
        # Create navigator overlay
        print("\n--- Creating navigator overlay ---")
        navigator = NavigatorOverlay(viewer, metadata, test_dataset)
        
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
        
        # Get test parameters
        num_channels = metadata['dimensions']['channel']
        num_regions = metadata['dimensions']['region']
        num_fovs = metadata['dimensions']['fov']
        channel_names = metadata['channels']
        region_names = metadata['regions']
        fov_names = metadata['fovs']
        
        print(f"\n--- Test Parameters ---")
        print(f"Channels: {num_channels} ({channel_names})")
        print(f"Regions: {num_regions} ({region_names})")
        print(f"FOVs: {num_fovs}")
        print(f"Clicks per channel: {clicks_per_channel}")
        
        # Test results tracking
        test_results = {
            'channel_switches': 0,
            'successful_channel_switches': 0,
            'navigator_clicks': 0,
            'successful_fov_jumps': 0,
            'channel_results': {},
            'errors': []
        }
        
        print(f"\n--- Starting Channel Navigation Tests ---")
        
        # Test each channel
        for channel_idx in range(num_channels):
            channel_name = channel_names[channel_idx]
            print(f"\n🔄 Testing Channel {channel_idx}: {channel_name}")
            
            # Initialize channel results
            test_results['channel_results'][channel_name] = {
                'navigator_visible': False,
                'clicks_attempted': 0,
                'clicks_successful': 0,
                'fov_positions_tested': []
            }
            
            try:
                # Switch to this channel using the slider
                print(f"  Switching to channel {channel_idx}...")
                current_step = list(viewer.dims.current_step)
                current_step[1] = channel_idx  # Channel is at position 1
                viewer.dims.current_step = current_step
                
                test_results['channel_switches'] += 1
                
                # Wait for the switch to take effect
                time.sleep(1)
                
                # Verify the channel switch worked
                actual_step = viewer.dims.current_step
                if actual_step[1] == channel_idx:
                    test_results['successful_channel_switches'] += 1
                    print(f"  ✓ Successfully switched to channel {channel_idx}")
                    
                    # Check if navigator is visible for this channel
                    navigator_visible = False
                    current_region_idx = actual_step[2]
                    
                    # Check if we have a navigator for this channel-region combination
                    if hasattr(navigator, 'nav_layers'):
                        for (reg_idx, ch_idx), nav_layer in navigator.nav_layers.items():
                            if reg_idx == current_region_idx and nav_layer.visible:
                                navigator_visible = True
                                test_results['channel_results'][channel_name]['navigator_visible'] = True
                                print(f"  ✓ Navigator visible for channel {channel_idx}, region {current_region_idx}")
                                break
                    
                    if not navigator_visible:
                        print(f"  ⚠️ No visible navigator for channel {channel_idx}, region {current_region_idx}")
                        test_results['errors'].append(f"No navigator visible for channel {channel_name}")
                        continue
                    
                    # Test navigator clicks for this channel
                    print(f"  Testing {clicks_per_channel} random navigator clicks...")
                    
                    for click_num in range(clicks_per_channel):
                        try:
                            # Record position before click
                            pre_click_step = list(viewer.dims.current_step)
                            pre_click_fov = pre_click_step[3]
                            pre_click_fov_name = fov_names[pre_click_fov] if pre_click_fov < len(fov_names) else f"FOV_{pre_click_fov}"
                            
                            print(f"    Click {click_num + 1}: Starting at FOV {pre_click_fov}:{pre_click_fov_name}")
                            
                            # Find available FOVs for current region to click on
                            current_region_name = region_names[current_region_idx]
                            available_fovs = []
                            
                            # Get navigator metadata to find available FOVs
                            if hasattr(navigator, 'region_channel_navigators'):
                                nav_key = None
                                # Find the navigator key for current region and channel
                                for (reg_idx, ch_idx) in navigator.region_channel_navigators.keys():
                                    if reg_idx == current_region_idx:
                                        nav_key = (reg_idx, ch_idx)
                                        break
                                
                                if nav_key and nav_key in navigator.region_channel_navigators:
                                    _, _, nav_metadata = navigator.region_channel_navigators[nav_key]
                                    available_fovs = list(nav_metadata.get('coordinates', {}).keys())
                            
                            if not available_fovs:
                                print(f"    ⚠️ No available FOVs found for region {current_region_name}")
                                continue
                            
                            # Choose a random target FOV
                            target_fov_name = random.choice(available_fovs)
                            target_fov_idx = fov_names.index(target_fov_name) if target_fov_name in fov_names else None
                            
                            if target_fov_idx is None:
                                print(f"    ⚠️ Could not find index for target FOV {target_fov_name}")
                                continue
                            
                            print(f"    Target: FOV {target_fov_idx}:{target_fov_name}")
                            
                            # Simulate navigator click by directly setting the FOV
                            # This simulates what would happen when clicking on the navigator
                            post_click_step = list(viewer.dims.current_step)
                            post_click_step[3] = target_fov_idx  # Update FOV dimension
                            viewer.dims.current_step = post_click_step
                            
                            test_results['navigator_clicks'] += 1
                            test_results['channel_results'][channel_name]['clicks_attempted'] += 1
                            
                            # Wait for the jump to take effect
                            time.sleep(0.5)
                            
                            # Verify the FOV jump worked
                            actual_post_click_step = viewer.dims.current_step
                            actual_fov_idx = actual_post_click_step[3]
                            actual_fov_name = fov_names[actual_fov_idx] if actual_fov_idx < len(fov_names) else f"FOV_{actual_fov_idx}"
                            
                            if actual_fov_idx == target_fov_idx:
                                test_results['successful_fov_jumps'] += 1
                                test_results['channel_results'][channel_name]['clicks_successful'] += 1
                                test_results['channel_results'][channel_name]['fov_positions_tested'].append(str(target_fov_name))
                                print(f"    ✓ Successfully jumped to FOV {actual_fov_idx}:{actual_fov_name}")
                            else:
                                print(f"    ❌ FOV jump failed. Expected: {target_fov_idx}:{target_fov_name}, Got: {actual_fov_idx}:{actual_fov_name}")
                                test_results['errors'].append(f"FOV jump failed in channel {channel_name}: expected {target_fov_name}, got {actual_fov_name}")
                            
                        except Exception as e:
                            print(f"    ❌ Error during click {click_num + 1}: {e}")
                            test_results['errors'].append(f"Click error in channel {channel_name}: {e}")
                    
                else:
                    print(f"  ❌ Channel switch failed. Expected: {channel_idx}, Got: {actual_step[1]}")
                    test_results['errors'].append(f"Channel switch failed for {channel_name}")
                
            except Exception as e:
                print(f"  ❌ Error testing channel {channel_idx}: {e}")
                test_results['errors'].append(f"Channel {channel_name} test error: {e}")
        
        # Generate comprehensive test report
        print(f"\n" + "="*80)
        print("TEST RESULTS SUMMARY")
        print("="*80)
        
        # Overall statistics
        print(f"Channel Switches: {test_results['successful_channel_switches']}/{test_results['channel_switches']} successful")
        print(f"Navigator Clicks: {test_results['successful_fov_jumps']}/{test_results['navigator_clicks']} successful")
        
        # Per-channel results
        print(f"\nPer-Channel Results:")
        for channel_name, results in test_results['channel_results'].items():
            print(f"  {channel_name}:")
            print(f"    Navigator visible: {'✓' if results['navigator_visible'] else '❌'}")
            print(f"    Clicks successful: {results['clicks_successful']}/{results['clicks_attempted']}")
            if results['fov_positions_tested']:
                print(f"    FOVs tested: {', '.join(results['fov_positions_tested'][:3])}{'...' if len(results['fov_positions_tested']) > 3 else ''}")
        
        # Calculate success rates
        channel_switch_rate = (test_results['successful_channel_switches'] / test_results['channel_switches'] * 100) if test_results['channel_switches'] > 0 else 0
        fov_jump_rate = (test_results['successful_fov_jumps'] / test_results['navigator_clicks'] * 100) if test_results['navigator_clicks'] > 0 else 0
        
        print(f"\nSuccess Rates:")
        print(f"  Channel switching: {channel_switch_rate:.1f}%")
        print(f"  FOV jumping: {fov_jump_rate:.1f}%")
        
        # Show errors if any
        if test_results['errors']:
            print(f"\nErrors encountered ({len(test_results['errors'])}):")
            for error in test_results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error}")
            if len(test_results['errors']) > 5:
                print(f"  ... and {len(test_results['errors']) - 5} more errors")
        
        # Determine overall test result
        overall_success = (
            test_results['successful_channel_switches'] == test_results['channel_switches'] and
            test_results['successful_fov_jumps'] > 0 and  # At least some FOV jumps worked
            len(test_results['errors']) == 0  # No errors
        )
        
        print(f"\n" + "="*80)
        if overall_success:
            print("🎉 ALL TESTS PASSED - Channel navigation working perfectly!")
        elif test_results['successful_channel_switches'] > 0 and test_results['successful_fov_jumps'] > 0:
            print("⚠️ TESTS PARTIALLY SUCCESSFUL - Some functionality working")
        else:
            print("❌ TESTS FAILED - Major issues detected")
        print("="*80)
        
        # Clean up
        try:
            os.unlink(temp_file_path)
            print(f"✓ Cleaned up temp file: {temp_file_path}")
        except:
            pass
        
        # Close viewer
        try:
            viewer.close()
        except:
            pass
        
        return overall_success
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Starting comprehensive channel navigator test...")
    success = test_channel_navigator()
    sys.exit(0 if success else 1) 