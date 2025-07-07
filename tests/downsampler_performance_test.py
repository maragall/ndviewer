#!/usr/bin/env python3
"""
Performance testing framework for the Downsampler module.
Measures execution time for each stage and generates performance reports.
"""

import time
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import traceback
import sys

# Import the downsampler module from utils
from utils.downsampler import DownsampledNavigator

class PerformanceProfiler:
    """Profile and analyze the performance of the downsampler."""
    
    def __init__(self, output_dir: Path = Path("./performance_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        self.timings = {}
        self.stage_times = []
        
    def time_stage(self, stage_name: str):
        """Context manager to time a specific stage."""
        class StageTimer:
            def __init__(self, profiler, name):
                self.profiler = profiler
                self.name = name
                self.start_time = None
                
            def __enter__(self):
                self.start_time = time.perf_counter()
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                elapsed = time.perf_counter() - self.start_time
                if self.name not in self.profiler.timings:
                    self.profiler.timings[self.name] = []
                self.profiler.timings[self.name].append(elapsed)
                
        return StageTimer(self, stage_name)
    
    def run_single_test(self, acquisition_dir: Path, timepoint: int = 0, 
                       tile_size: int = 75, cache_enabled: bool = True,
                       region_name: Optional[str] = None) -> Dict:
        """Run a single test and collect timing data."""
        test_result = {
            'acquisition_dir': str(acquisition_dir),
            'timepoint': timepoint,
            'tile_size': tile_size,
            'cache_enabled': cache_enabled,
            'region_name': region_name,
            'stages': {},
            'total_time': 0,
            'success': False,
            'error': None
        }
        
        # Track progress stages
        stage_timings = {}
        last_progress = 0
        current_stage = None
        stage_start_time = None
        
        def progress_callback(percent: int, message: str):
            nonlocal last_progress, current_stage, stage_start_time
            
            # Detect stage changes based on message content
            if "Loading coordinates" in message:
                new_stage = "load_coordinates"
            elif "Scanning files" in message:
                new_stage = "scan_files"
            elif "Building grid" in message:
                new_stage = "build_grid"
            elif "Creating mosaic" in message:
                new_stage = "create_mosaic"
            elif "Processing tile" in message:
                new_stage = "process_tiles"
            else:
                new_stage = "other"
            
            # If stage changed, record the timing
            if current_stage and current_stage != new_stage:
                elapsed = time.perf_counter() - stage_start_time
                if current_stage not in stage_timings:
                    stage_timings[current_stage] = 0
                stage_timings[current_stage] += elapsed
            
            # Start timing new stage
            if current_stage != new_stage:
                current_stage = new_stage
                stage_start_time = time.perf_counter()
            
            last_progress = percent
        
        try:
            # Create navigator with progress callback
            navigator = DownsampledNavigator(
                acquisition_dir=acquisition_dir,
                tile_size=tile_size,
                cache_enabled=cache_enabled,
                progress_callback=progress_callback
            )
            
            # Time the overall process
            start_time = time.perf_counter()
            
            # Initialize stage timing
            current_stage = "initialization"
            stage_start_time = time.perf_counter()
            
            if region_name:
                mosaic, metadata = navigator.create_mosaic_for_region(region_name, timepoint)
            else:
                mosaic, metadata = navigator.create_mosaic(timepoint)
            
            # Record final stage
            if current_stage:
                elapsed = time.perf_counter() - stage_start_time
                if current_stage not in stage_timings:
                    stage_timings[current_stage] = 0
                stage_timings[current_stage] += elapsed
            
            total_time = time.perf_counter() - start_time
            
            # Analyze the results
            test_result['total_time'] = total_time
            test_result['stages'] = stage_timings
            test_result['success'] = True
            test_result['mosaic_shape'] = mosaic.shape
            test_result['num_fovs'] = len(metadata.get('fov_grid', {}))
            test_result['num_channels'] = len(metadata.get('channels', []))
            
            # Detailed tile processing analysis
            with self.time_stage("tile_generation_analysis"):
                tile_times = []
                for fov in list(metadata.get('coordinates', {}).keys())[:10]:  # Sample 10 FOVs
                    tile_start = time.perf_counter()
                    navigator._get_tile_image(fov)
                    tile_times.append(time.perf_counter() - tile_start)
                
                if tile_times:
                    test_result['avg_tile_time'] = np.mean(tile_times)
                    test_result['max_tile_time'] = np.max(tile_times)
                    test_result['min_tile_time'] = np.min(tile_times)
            
        except Exception as e:
            test_result['success'] = False
            test_result['error'] = str(e)
            test_result['traceback'] = traceback.format_exc()
            
        return test_result
    
    def run_performance_suite(self, acquisition_dirs: List[Path], 
                            variations: Dict = None) -> Dict:
        """Run a suite of performance tests with different parameters."""
        if variations is None:
            variations = {
                'tile_sizes': [50, 75, 100, 150],
                'cache_enabled': [True, False],
                'regions': [None]  # Add specific regions if needed
            }
        
        results = {
            'test_date': datetime.now().isoformat(),
            'acquisition_dirs': [str(d) for d in acquisition_dirs],
            'variations': variations,
            'tests': []
        }
        
        total_tests = len(acquisition_dirs) * len(variations['tile_sizes']) * len(variations['cache_enabled'])
        current_test = 0
        
        for acq_dir in acquisition_dirs:
            for tile_size in variations['tile_sizes']:
                for cache_enabled in variations['cache_enabled']:
                    current_test += 1
                    print(f"\nTest {current_test}/{total_tests}:")
                    print(f"  Directory: {acq_dir}")
                    print(f"  Tile size: {tile_size}")
                    print(f"  Cache: {cache_enabled}")
                    
                    result = self.run_single_test(
                        acq_dir, 
                        tile_size=tile_size,
                        cache_enabled=cache_enabled
                    )
                    results['tests'].append(result)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.output_dir / f"performance_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        return results
    
    def analyze_results(self, results: Dict) -> Dict:
        """Analyze performance results and identify bottlenecks."""
        analysis = {
            'summary': {},
            'bottlenecks': [],
            'recommendations': []
        }
        
        # Extract successful tests
        successful_tests = [t for t in results['tests'] if t['success']]
        
        if not successful_tests:
            analysis['summary']['error'] = "No successful tests to analyze"
            return analysis
        
        # Aggregate stage timings
        stage_totals = {}
        stage_counts = {}
        
        for test in successful_tests:
            for stage, time_val in test['stages'].items():
                if stage not in stage_totals:
                    stage_totals[stage] = 0
                    stage_counts[stage] = 0
                stage_totals[stage] += time_val
                stage_counts[stage] += 1
        
        # Calculate averages and percentages
        stage_averages = {}
        total_avg_time = sum(stage_totals.values()) / len(successful_tests)
        
        for stage, total in stage_totals.items():
            avg = total / stage_counts[stage]
            percentage = (avg / total_avg_time) * 100 if total_avg_time > 0 else 0
            stage_averages[stage] = {
                'avg_time': avg,
                'percentage': percentage,
                'total_time': total
            }
        
        # Sort stages by time consumption
        sorted_stages = sorted(stage_averages.items(), 
                             key=lambda x: x[1]['avg_time'], 
                             reverse=True)
        
        analysis['summary']['stage_breakdown'] = dict(sorted_stages)
        analysis['summary']['total_tests'] = len(successful_tests)
        analysis['summary']['avg_total_time'] = total_avg_time
        
        # Identify bottlenecks (stages taking >20% of time)
        for stage, data in sorted_stages:
            if data['percentage'] > 20:
                analysis['bottlenecks'].append({
                    'stage': stage,
                    'avg_time': data['avg_time'],
                    'percentage': data['percentage']
                })
        
        # Analyze cache impact
        cache_on = [t for t in successful_tests if t['cache_enabled']]
        cache_off = [t for t in successful_tests if not t['cache_enabled']]
        
        if cache_on and cache_off:
            cache_on_avg = np.mean([t['total_time'] for t in cache_on])
            cache_off_avg = np.mean([t['total_time'] for t in cache_off])
            cache_speedup = (cache_off_avg - cache_on_avg) / cache_off_avg * 100
            
            analysis['summary']['cache_impact'] = {
                'with_cache_avg': cache_on_avg,
                'without_cache_avg': cache_off_avg,
                'speedup_percentage': cache_speedup
            }
        
        # Generate recommendations
        if analysis['bottlenecks']:
            top_bottleneck = analysis['bottlenecks'][0]
            if 'tile' in top_bottleneck['stage'].lower() or 'process' in top_bottleneck['stage'].lower():
                analysis['recommendations'].append(
                    "Tile processing is the main bottleneck. Consider:\n"
                    "- Parallel tile processing\n"
                    "- More aggressive downsampling\n"
                    "- Pre-computing tile statistics"
                )
            elif 'scan' in top_bottleneck['stage'].lower():
                analysis['recommendations'].append(
                    "File scanning is slow. Consider:\n"
                    "- Caching file listings\n"
                    "- Parallel file scanning\n"
                    "- Using file system metadata"
                )
        
        return analysis
    
    def generate_report(self, results: Dict, analysis: Dict):
        """Generate visual performance report."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Downsampler Performance Analysis', fontsize=16)
        
        # 1. Stage breakdown pie chart
        if 'stage_breakdown' in analysis['summary']:
            stages = list(analysis['summary']['stage_breakdown'].keys())
            times = [d['avg_time'] for d in analysis['summary']['stage_breakdown'].values()]
            
            ax1.pie(times, labels=stages, autopct='%1.1f%%')
            ax1.set_title('Time Distribution by Stage')
        
        # 2. Performance across tile sizes
        tile_size_data = {}
        for test in results['tests']:
            if test['success']:
                size = test['tile_size']
                if size not in tile_size_data:
                    tile_size_data[size] = []
                tile_size_data[size].append(test['total_time'])
        
        if tile_size_data:
            sizes = sorted(tile_size_data.keys())
            means = [np.mean(tile_size_data[s]) for s in sizes]
            stds = [np.std(tile_size_data[s]) for s in sizes]
            
            ax2.errorbar(sizes, means, yerr=stds, marker='o', capsize=5)
            ax2.set_xlabel('Tile Size (pixels)')
            ax2.set_ylabel('Total Time (seconds)')
            ax2.set_title('Performance vs Tile Size')
            ax2.grid(True, alpha=0.3)
        
        # 3. Cache impact
        if 'cache_impact' in analysis['summary']:
            cache_data = analysis['summary']['cache_impact']
            categories = ['With Cache', 'Without Cache']
            values = [cache_data['with_cache_avg'], cache_data['without_cache_avg']]
            
            bars = ax3.bar(categories, values)
            ax3.set_ylabel('Average Time (seconds)')
            ax3.set_title(f'Cache Impact (Speedup: {cache_data["speedup_percentage"]:.1f}%)')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{val:.2f}s', ha='center', va='bottom')
        
        # 4. Stage timing histogram
        all_stage_times = []
        stage_labels = []
        
        for test in results['tests']:
            if test['success']:
                for stage, time_val in test['stages'].items():
                    all_stage_times.append(time_val)
                    stage_labels.append(stage)
        
        if all_stage_times:
            ax4.hist(all_stage_times, bins=30, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Time (seconds)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Distribution of Stage Execution Times')
            ax4.set_yscale('log')  # Log scale for better visibility
        
        plt.tight_layout()
        
        # Save the report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"performance_report_{timestamp}.png"
        plt.savefig(report_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance report saved to: {report_file}")
        
        # Generate text summary
        summary_file = self.output_dir / f"performance_summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write("DOWNSAMPLER PERFORMANCE ANALYSIS SUMMARY\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Total tests run: {analysis['summary']['total_tests']}\n")
            f.write(f"Average total time: {analysis['summary']['avg_total_time']:.2f} seconds\n\n")
            
            f.write("STAGE BREAKDOWN:\n")
            for stage, data in analysis['summary']['stage_breakdown'].items():
                f.write(f"  {stage}: {data['avg_time']:.3f}s ({data['percentage']:.1f}%)\n")
            
            f.write("\nBOTTLENECKS:\n")
            for bottleneck in analysis['bottlenecks']:
                f.write(f"  - {bottleneck['stage']}: {bottleneck['avg_time']:.3f}s "
                       f"({bottleneck['percentage']:.1f}% of total)\n")
            
            f.write("\nRECOMMENDATIONS:\n")
            for i, rec in enumerate(analysis['recommendations'], 1):
                f.write(f"{i}. {rec}\n")
        
        print(f"Performance summary saved to: {summary_file}")


def main():
    """Run performance analysis on specified acquisition directories."""
    # Parse command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_downsampler_performance.py <acquisition_dir1> [<acquisition_dir2> ...]")
        sys.exit(1)
    
    acquisition_dirs = [Path(arg) for arg in sys.argv[1:]]
    
    # Verify directories exist
    for dir_path in acquisition_dirs:
        if not dir_path.exists():
            print(f"Error: Directory {dir_path} does not exist")
            sys.exit(1)
    
    # Create profiler
    profiler = PerformanceProfiler()
    
    # Define test variations
    variations = {
        'tile_sizes': [50, 75, 100],  # Different tile sizes
        'cache_enabled': [True, False],  # Test with and without cache
        'regions': [None]  # Can add specific regions if needed
    }
    
    print("Starting performance analysis...")
    print(f"Testing directories: {[str(d) for d in acquisition_dirs]}")
    print(f"Test variations: {variations}")
    
    # Run performance suite
    results = profiler.run_performance_suite(acquisition_dirs, variations)
    
    # Analyze results
    analysis = profiler.analyze_results(results)
    
    # Generate report
    profiler.generate_report(results, analysis)
    
    # Print summary to console
    print("\n" + "="*50)
    print("PERFORMANCE ANALYSIS COMPLETE")
    print("="*50)
    
    if analysis['summary']:
        print(f"\nAverage total time: {analysis['summary']['avg_total_time']:.2f} seconds")
        
        print("\nTop bottlenecks:")
        for bottleneck in analysis['bottlenecks'][:3]:
            print(f"  - {bottleneck['stage']}: {bottleneck['percentage']:.1f}% of time")
        
        if 'cache_impact' in analysis['summary']:
            cache_data = analysis['summary']['cache_impact']
            print(f"\nCache speedup: {cache_data['speedup_percentage']:.1f}%")
    
    print(f"\nDetailed results saved to: {profiler.output_dir}")


if __name__ == "__main__":
    main()