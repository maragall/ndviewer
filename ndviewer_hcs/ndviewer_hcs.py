"""NDViewer HCS - Main entry point

Modular high-content screening viewer with preprocessing and visualization.
"""

import sys
import argparse
from PyQt5.QtWidgets import QApplication

from .gui_config import ConfigurationGUI
from .viewer import ViewerMainWindow


def main():
    """Main entry point for NDViewer HCS"""
    parser = argparse.ArgumentParser(description="NDViewer HCS - High-Content Screening Viewer")
    parser.add_argument('--mode', choices=['config', 'viewer'], default='config',
                       help='Launch mode: config GUI or viewer directly')
    parser.add_argument('--data-dir', help='Acquisition data directory (for viewer mode)')
    parser.add_argument('--timepoint', type=int, default=0, help='Timepoint to load')
    parser.add_argument('--downsample', type=float, default=0.85, help='Downsample factor')
    
    args = parser.parse_args()
    
    app = QApplication(sys.argv)
    
    if args.mode == 'config':
        # Launch configuration GUI
        window = ConfigurationGUI()
        window.show()
    elif args.mode == 'viewer':
        # Launch viewer directly
        if not args.data_dir:
            print("Error: --data-dir required for viewer mode")
            sys.exit(1)
        window = ViewerMainWindow(args.data_dir, args.timepoint, args.downsample)
        window.show()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
