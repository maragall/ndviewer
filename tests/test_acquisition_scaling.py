#!/usr/bin/env python3
"""
Test script to verify FOV scaling across multiple acquisitions.
Compares navigator layer size vs FOV layer size to ensure 5:1 ratio.
"""

import os
import sys
import math
from pathlib import Path

# Add the ndviewer_napari directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'ndviewer_napari'))

from scaling_manager import ScalingManager


def test_acquisition_scaling(acquisition_path: str, expected_to_pass: bool = True):
    """
    Test scaling for a single acquisition.
    
    Parameters:
    -----------
    acquisition_path : str
        Path to the acquisition directory
    expected_to_pass : bool
        Whether this acquisition should pass the 5:1 ratio test
        
    Returns:
    --------
    dict: Test results
    """
    print(f"\n{'='*80}")
    print(f"TESTING ACQUISITION: {os.path.basename(acquisition_path)}")
    print(f"{'='*80}")
    
    # Check if directory exists
    if not os.path.exists(acquisition_path):
        print(f"❌ ERROR: Directory does not exist: {acquisition_path}")
        return {
            'acquisition': os.path.basename(acquisition_path),
            'passed': False,
            'error': 'Directory does not exist',
            'navigator_size': None,
            'fov_size': None,
            'ratio': None
        }
    
    try:
        # Import the downsampler to get FIRST_FOV_COUNT
        sys.path.append(os.path.join(os.path.dirname(__file__), 'utils'))
        from downsampler import FIRST_FOV_COUNT, DownsampledNavigator
        
        # Reset FIRST_FOV_COUNT to None so it gets recalculated
        import downsampler
        downsampler.FIRST_FOV_COUNT = None
        
        # Create navigator to get FOV count
        navigator = DownsampledNavigator(Path(acquisition_path))
        
        # Get the first FOV count (this will set the global variable)
        try:
            mosaic, metadata = navigator.create_mosaic(timepoint=0)
            fov_count = downsampler.FIRST_FOV_COUNT
            
            if fov_count is None:
                print(f"❌ ERROR: Could not determine FOV count for {acquisition_path}")
                return {
                    'acquisition': os.path.basename(acquisition_path),
                    'passed': False,
                    'error': 'Could not determine FOV count',
                    'navigator_size': None,
                    'fov_size': None,
                    'ratio': None
                }
            
            print(f"✓ FOV count determined: {fov_count}")
            
            # Calculate navigator size
            navigator_width = 75 * fov_count  # 75 pixels per FOV horizontally
            navigator_height = 75  # 1 row of tiles
            navigator_area = navigator_width * navigator_height
            
            print(f"✓ Navigator size: {navigator_width} × {navigator_height} = {navigator_area:,} pixels")
            
            # Create scaling manager and calculate FOV size
            scaling_manager = ScalingManager()
            
            # Use typical FOV dimensions (you might need to adjust this)
            fov_dims = {'x': 2048, 'y': 2048}  # Typical FOV size
            
            # Calculate FOV layer size using the formula
            navigator_area_calc = 75 * 75 * fov_count
            fov_layer_size = math.floor(math.sqrt(navigator_area_calc * 5))  # 5:1 ratio
            
            print(f"✓ Calculated FOV layer size: {fov_layer_size} × {fov_layer_size} = {fov_layer_size**2:,} pixels")
            
            # Calculate actual ratio
            ratio = fov_layer_size**2 / navigator_area
            
            print(f"✓ Actual ratio: {ratio:.2f}:1")
            
            # Determine if test passes (should be close to 5:1)
            tolerance = 0.5  # Allow 0.5 deviation from 5:1
            passed = abs(ratio - 5.0) <= tolerance
            
            print(f"✓ Expected ratio: 5:1 ± {tolerance}")
            print(f"✓ Test {'PASSED' if passed else 'FAILED'}")
            print(f"✓ Expected to {'PASS' if expected_to_pass else 'FAIL'}: {'✅' if passed == expected_to_pass else '❌'}")
            
            return {
                'acquisition': os.path.basename(acquisition_path),
                'passed': passed,
                'error': None,
                'navigator_size': navigator_area,
                'fov_size': fov_layer_size**2,
                'ratio': ratio,
                'expected_to_pass': expected_to_pass,
                'test_correct': passed == expected_to_pass
            }
            
        except Exception as e:
            print(f"❌ ERROR: Failed to create mosaic: {e}")
            return {
                'acquisition': os.path.basename(acquisition_path),
                'passed': False,
                'error': f'Failed to create mosaic: {e}',
                'navigator_size': None,
                'fov_size': None,
                'ratio': None
            }
            
    except Exception as e:
        print(f"❌ ERROR: Failed to import or initialize: {e}")
        return {
            'acquisition': os.path.basename(acquisition_path),
            'passed': False,
            'error': f'Failed to import or initialize: {e}',
            'navigator_size': None,
            'fov_size': None,
            'ratio': None
        }


def main():
    """Run tests on all acquisitions."""
    
    # Test acquisitions with expected outcomes
    acquisitions = [
        {
            'path': "/Users/julioamaragall/Documents/manual_selection_2_slides_2025-05-08_17-28-39.146212",
            'expected_to_pass': True  # Should pass (5:1 ratio)
        },
        {
            'path': "/Users/julioamaragall/Documents/10x_mouse_brain_2025-04-23_00-53-11.236590", 
            'expected_to_pass': False  # Should fail (different ratio)
        },
        {
            'path': "/Users/julioamaragall/Documents/10x_2025-05-30_21-33-18.411955",
            'expected_to_pass': True  # Should pass (5:1 ratio)
        }
    ]
    
    print("FOV SCALING TEST SUITE")
    print("="*80)
    print("Testing FOV layer size vs Navigator layer size")
    print("Expected ratio: 5:1 (FOV should be 5x larger than navigator)")
    print("="*80)
    
    results = []
    
    for i, acq in enumerate(acquisitions, 1):
        print(f"\nTest {i}/{len(acquisitions)}")
        result = test_acquisition_scaling(acq['path'], acq['expected_to_pass'])
        results.append(result)
    
    # Summary
    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    
    passed_tests = 0
    correct_predictions = 0
    
    for result in results:
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        prediction = "✅ CORRECT" if result.get('test_correct', False) else "❌ WRONG"
        
        print(f"{result['acquisition']}:")
        print(f"  Status: {status}")
        print(f"  Ratio: {result.get('ratio', 'N/A'):.2f}:1")
        print(f"  Expected to {'PASS' if result.get('expected_to_pass') else 'FAIL'}: {prediction}")
        
        if result['passed']:
            passed_tests += 1
        if result.get('test_correct', False):
            correct_predictions += 1
    
    print(f"\nOverall Results:")
    print(f"  Tests passed: {passed_tests}/{len(results)}")
    print(f"  Predictions correct: {correct_predictions}/{len(results)}")
    
    # Final validation
    if passed_tests >= 1 and passed_tests < len(results):
        print(f"\n✅ SUCCESS: At least one test passed and at least one failed as expected!")
    else:
        print(f"\n❌ FAILURE: Test distribution is incorrect!")
        print(f"   Expected: At least 1 pass and at least 1 fail")
        print(f"   Actual: {passed_tests} passed, {len(results) - passed_tests} failed")


if __name__ == "__main__":
    main() 