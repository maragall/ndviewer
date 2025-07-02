#!/usr/bin/env python3
"""
Napari Viewer Script - Modular Version
Main entry point that combines FOV viewer and navigator functionality.
"""
import sys
import os

# Import our modular components
from fov_viewer import FOVViewer
from navigator import add_navigator_to_viewer


def main():
    """Main function to launch the modular napari viewer."""
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python napari_viewer_script.py <temp_file_path>")
        sys.exit(1)
    
    temp_file_path = sys.argv[1]
    
    print("="*60)
    print("LAUNCHING MODULAR NAPARI VIEWER")
    print("="*60)
    
    try:
        # Create the FOV viewer
        print("Step 1: Creating FOV viewer...")
        fov_viewer = FOVViewer(temp_file_path)
        viewer = fov_viewer.create_viewer()
        
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