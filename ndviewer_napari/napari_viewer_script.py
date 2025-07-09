#!/usr/bin/env python3
"""
Napari Viewer Script - Modular Version
Main entry point that combines FOV viewer and navigator functionality.
"""
import sys
import os
import argparse

# Import our modular components
from fov_viewer import FOVViewer
from navigator import add_navigator_to_viewer


def parse_arguments():
    """Parse command line arguments with backward compatibility."""
    # Handle old-style command line arguments for backward compatibility
    if len(sys.argv) == 2 and not sys.argv[1].startswith('--'):
        # Old style: python script.py temp_file_path
        class Args:
            def __init__(self, temp_file_path):
                self.temp_file_path = temp_file_path
                self.navigator = False
                self.no_navigator = False
        return Args(sys.argv[1])
    
    # New style with argparse
    parser = argparse.ArgumentParser(description='NDViewer - Multi-dimensional image viewer')
    parser.add_argument('temp_file_path', help='Path to the temporary metadata file')
    parser.add_argument('--navigator', action='store_true', help='Enable navigator overlay')
    parser.add_argument('--no-navigator', action='store_true', help='Disable navigator overlay')
    return parser.parse_args()


def main():
    """Main function to launch the modular napari viewer."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Check for conflicting arguments
    if args.navigator and args.no_navigator:
        print("Error: Cannot specify both --navigator and --no-navigator")
        sys.exit(1)
    
    # Determine navigator preference (default to True if not specified)
    enable_navigator = True
    if args.no_navigator:
        enable_navigator = False
    elif args.navigator:
        enable_navigator = True
    
    temp_file_path = args.temp_file_path
    
    print("="*60)
    print("LAUNCHING MODULAR NAPARI VIEWER")
    print(f"Navigator: {'Enabled' if enable_navigator else 'Disabled'}")
    print("="*60)
    
    try:
        # Create the FOV viewer
        print("Step 1: Creating FOV viewer...")
        fov_viewer = FOVViewer(temp_file_path, enable_navigator=enable_navigator)
        viewer = fov_viewer.create_viewer()
        
        # Conditionally add navigator overlay
        if enable_navigator:
            print("Step 2: Adding navigator overlay...")
            navigator = add_navigator_to_viewer(
                viewer=viewer,
                metadata=fov_viewer.get_metadata(),
                directory=fov_viewer.directory,
                fov_viewer=fov_viewer  # Pass the fov_viewer for channel visibility tracking
            )
            
            if navigator:
                print("✓ Navigator overlay added successfully")
            else:
                print("⚠ Navigator overlay not available (continuing without navigator)")
        else:
            print("Step 2: Skipping navigator overlay (disabled by user)")
        
        print("Step 3: Starting napari...")
        print("="*60)
        
        # Run the napari event loop
        fov_viewer.run()
            
    except Exception as e:
        print(f"❌ Error launching viewer: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()