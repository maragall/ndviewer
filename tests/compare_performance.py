#!/usr/bin/env python3
"""
Compare performance between original and optimized downsampler implementations.
"""

import time
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
import json
from datetime import datetime

# Import both implementations
from downsampler import DownsampledNavigator
from downsampler_optimized import OptimizedDownsampledNavigator


class PerformanceComparison:
    """Compare performance of original vs optimized implementations."""
    
    def __init__(self, output_dir: Path = Path("./comparison_results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
    def compare_implementations(self, acquisition_dir: Path, 
                              tile_sizes: list = [75],
                              test_cache: bool = True,
                              test_regions: bool = False) -> Dict:
        """Run comparison tests between implementations."""
        
        results = {
            'acquisition_dir': str(acquisition_dir),
            'timestamp': datetime.now().isoformat(),
            'tests': []
        }
        
        # Test configurations
        configurations = []
        for tile_size in tile_sizes:
            for cache_enabled in ([True, False] if test_cache else [True]):
                configurations.append({
                    'tile_size': tile_size,
                    'cache_enabled': cache_enabled
                })
        
        # Run tests for each configuration
        for config in configurations:
            print(f"\n{'='*60}")
            print(f"Testing configuration: {config}")
            print(f"{'='*60}")
            
            # Test original implementation
            print("\n[ORIGINAL] Starting test...")
            original_result = self._test_implementation(
                acquisition_dir,
                DownsampledNavigator,
                config,
                "Original"
            )
            
            # Clear cache between tests for fair comparison
            cache_dir = acquisition_dir / "cache"
            if cache_dir.exists() and not config['cache_enabled']:
                import shutil
                shutil.rmtree(cache_dir)
                cache_dir.mkdir()
            
            # Test optimized implementation
            print("\n[OPTIMIZED] Starting test...")
            optimized_result = self._test_implementation(
                acquisition_dir,
                OptimizedDownsampledNavigator,
                config,
                "Optimized"
            )
            
            # Calculate speedup
            if original_result['success'] and optimized_result['success']:
                speedup = original_result['total_time'] / optimized_result['total_time']
                improvement_pct = (1 - optimized_result['total_time'] / original_result['total_time']) * 100
            else:
                speedup = 0
                improvement_pct = 0
            
            # Store comparison results
            comparison = {
                'configuration': config,
                'original': original_result,
                'optimized': optimized_result,
                'speedup': speedup,
                'improvement_percentage': improvement_pct
            }
            
            results['tests'].append(comparison)
            
            # Print summary
            print(f"\n{'='*60}")
            print(f"COMPARISON SUMMARY")
            print(f"{'='*60}")
            print(f"Original time: {original_result['total_time']:.2f}s")
            print(f"Optimized time: {optimized_result['total_time']:.2f}s")
            print(f"Speedup: {speedup:.2f}x")
            print(f"Improvement: {improvement_pct:.1f}%")
            
            # Print stage-by-stage comparison
            if 'stages' in original_result and 'stages' in optimized_result:
                print(f"\nStage-by-stage comparison:")
                all_stages = set(list(original_result['stages'].keys()) + 
                               list(optimized_result['stages'].keys()))
                
                for stage in sorted(all_stages):
                    orig_time = original_result['stages'].get(stage, 0)
                    opt_time = optimized_result['stages'].get(stage, 0)
                    if orig_time > 0:
                        stage_speedup = orig_time / opt_time if opt_time > 0 else float('inf')
                        print(f"  {stage}: {orig_time:.3f}s → {opt_time:.3f}s ({stage_speedup:.2f}x)")
        
        # Save results
        self._save_results(results)
        
        # Generate visualization
        self._generate_comparison_plots(results)
        
        return results
    
    def _test_implementation(self, acquisition_dir: Path, 
                           navigator_class, config: Dict, 
                           impl_name: str) -> Dict:
        """Test a single implementation."""
        
        result = {
            'implementation': impl_name,
            'success': False,
            'total_time': 0,
            'stages': {},
            'error': None
        }
        
        # Track stage timings
        stage_times = {}
        current_stage = None
        stage_start = None
        
        def progress_callback(percent: int, message: str):
            nonlocal current_stage, stage_start
            
            # Identify stage from message
            if "Loading coordinates" in message:
                new_stage = "load_coordinates"
            elif "Scanning files" in message:
                new_stage = "scan_files"
            elif "Building grid" in message:
                new_stage = "build_grid"
            elif "Creating mosaic" in message:
                new_stage = "create_mosaic"
            elif "Processing" in message or "tile" in message.lower():
                new_stage = "tile_processing"
            elif "Pre-scanning" in message:
                new_stage = "channel_prescanning"
            else:
                new_stage = "other"
            
            # Record timing for previous stage
            if current_stage and current_stage != new_stage and stage_start:
                elapsed = time.perf_counter() - stage_start
                if current_stage not in stage_times:
                    stage_times[current_stage] = 0
                stage_times[current_stage] += elapsed
            
            # Start new stage
            if current_stage != new_stage:
                current_stage = new_stage
                stage_start = time.perf_counter()
        
        try:
            # Create navigator instance (both now support n_workers)
            navigator = navigator_class(
                acquisition_dir=acquisition_dir,
                tile_size=config['tile_size'],
                cache_enabled=config['cache_enabled'],
                progress_callback=progress_callback,
                n_workers=4  # Use 4 workers for testing
            )
            
            # Time the mosaic creation
            start_time = time.perf_counter()
            current_stage = "initialization"
            stage_start = start_time
            
            mosaic, metadata = navigator.create_mosaic(timepoint=0)
            
            # Record final stage
            if current_stage and stage_start:
                elapsed = time.perf_counter() - stage_start
                if current_stage not in stage_times:
                    stage_times[current_stage] = 0
                stage_times[current_stage] += elapsed
            
            total_time = time.perf_counter() - start_time
            
            result['success'] = True
            result['total_time'] = total_time
            result['stages'] = stage_times
            result['mosaic_shape'] = mosaic.shape
            result['num_fovs'] = len(metadata.get('fov_grid', {}))
            
        except Exception as e:
            result['error'] = str(e)
            print(f"[{impl_name}] ERROR: {e}")
            
        return result
    
    def _save_results(self, results: Dict):
        """Save comparison results to JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.output_dir / f"comparison_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {filename}")
    
    def _generate_comparison_plots(self, results: Dict):
        """Generate comparison visualizations."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Downsampler Performance Comparison: Original vs Optimized', fontsize=16)
        
        # Extract successful tests
        successful_tests = [t for t in results['tests'] 
                          if t['original']['success'] and t['optimized']['success']]
        
        if not successful_tests:
            print("No successful tests to plot")
            return
        
        # 1. Overall performance comparison
        test_labels = [f"Tile:{t['configuration']['tile_size']} Cache:{t['configuration']['cache_enabled']}" 
                      for t in successful_tests]
        original_times = [t['original']['total_time'] for t in successful_tests]
        optimized_times = [t['optimized']['total_time'] for t in successful_tests]
        
        x = np.arange(len(test_labels))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, original_times, width, label='Original')
        bars2 = ax1.bar(x + width/2, optimized_times, width, label='Optimized')
        
        ax1.set_xlabel('Test Configuration')
        ax1.set_ylabel('Total Time (seconds)')
        ax1.set_title('Overall Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(test_labels, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}s', ha='center', va='bottom', fontsize=8)
        
        # 2. Speedup factors
        speedups = [t['speedup'] for t in successful_tests]
        colors = ['green' if s > 1 else 'red' for s in speedups]
        
        bars = ax2.bar(x, speedups, color=colors, alpha=0.7)
        ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Test Configuration')
        ax2.set_ylabel('Speedup Factor')
        ax2.set_title('Optimization Speedup (Higher is Better)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(test_labels, rotation=45, ha='right')
        
        # Add value labels
        for bar, speedup in zip(bars, speedups):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{speedup:.2f}x', ha='center', va='bottom', fontsize=8)
        
        # 3. Stage-by-stage comparison (average across all tests)
        stage_data = {}
        
        for test in successful_tests:
            for impl in ['original', 'optimized']:
                stages = test[impl].get('stages', {})
                for stage, time_val in stages.items():
                    if stage not in stage_data:
                        stage_data[stage] = {'original': [], 'optimized': []}
                    stage_data[stage][impl].append(time_val)
        
        # Calculate averages
        stages = []
        orig_avg = []
        opt_avg = []
        
        for stage, data in stage_data.items():
            if data['original'] and data['optimized']:
                stages.append(stage)
                orig_avg.append(np.mean(data['original']))
                opt_avg.append(np.mean(data['optimized']))
        
        if stages:
            x_stages = np.arange(len(stages))
            width = 0.35
            
            bars1 = ax3.bar(x_stages - width/2, orig_avg, width, label='Original')
            bars2 = ax3.bar(x_stages + width/2, opt_avg, width, label='Optimized')
            
            ax3.set_xlabel('Processing Stage')
            ax3.set_ylabel('Average Time (seconds)')
            ax3.set_title('Stage-by-Stage Performance')
            ax3.set_xticks(x_stages)
            ax3.set_xticklabels(stages, rotation=45, ha='right')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Improvement percentage
        improvements = [t['improvement_percentage'] for t in successful_tests]
        
        bars = ax4.bar(x, improvements, color='darkgreen', alpha=0.7)
        ax4.set_xlabel('Test Configuration')
        ax4.set_ylabel('Improvement (%)')
        ax4.set_title('Performance Improvement Percentage')
        ax4.set_xticks(x)
        ax4.set_xticklabels(test_labels, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, imp in zip(bars, improvements):
            ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                    f'{imp:.1f}%', ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = self.output_dir / f"comparison_plot_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Comparison plot saved to: {plot_file}")
        
        # Generate detailed report
        self._generate_detailed_report(results)
    
    def _generate_detailed_report(self, results: Dict):
        """Generate a detailed text report of the comparison."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.output_dir / f"detailed_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("DOWNSAMPLER PERFORMANCE COMPARISON REPORT\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated: {results['timestamp']}\n")
            f.write(f"Acquisition Directory: {results['acquisition_dir']}\n")
            f.write("=" * 70 + "\n\n")
            
            # Overall summary
            successful_tests = [t for t in results['tests'] 
                              if t['original']['success'] and t['optimized']['success']]
            
            if successful_tests:
                avg_speedup = np.mean([t['speedup'] for t in successful_tests])
                avg_improvement = np.mean([t['improvement_percentage'] for t in successful_tests])
                
                f.write("OVERALL SUMMARY\n")
                f.write("-" * 50 + "\n")
                f.write(f"Total tests run: {len(results['tests'])}\n")
                f.write(f"Successful comparisons: {len(successful_tests)}\n")
                f.write(f"Average speedup: {avg_speedup:.2f}x\n")
                f.write(f"Average improvement: {avg_improvement:.1f}%\n\n")
                
                # Find best and worst improvements
                best_test = max(successful_tests, key=lambda t: t['speedup'])
                worst_test = min(successful_tests, key=lambda t: t['speedup'])
                
                f.write(f"Best speedup: {best_test['speedup']:.2f}x ")
                f.write(f"(Tile size: {best_test['configuration']['tile_size']}, ")
                f.write(f"Cache: {best_test['configuration']['cache_enabled']})\n")
                
                f.write(f"Worst speedup: {worst_test['speedup']:.2f}x ")
                f.write(f"(Tile size: {worst_test['configuration']['tile_size']}, ")
                f.write(f"Cache: {worst_test['configuration']['cache_enabled']})\n\n")
            
            # Detailed test results
            f.write("DETAILED TEST RESULTS\n")
            f.write("=" * 70 + "\n\n")
            
            for i, test in enumerate(results['tests'], 1):
                f.write(f"Test {i}: ")
                f.write(f"Tile size={test['configuration']['tile_size']}, ")
                f.write(f"Cache={test['configuration']['cache_enabled']}\n")
                f.write("-" * 50 + "\n")
                
                if test['original']['success'] and test['optimized']['success']:
                    f.write(f"Original implementation: {test['original']['total_time']:.2f}s\n")
                    f.write(f"Optimized implementation: {test['optimized']['total_time']:.2f}s\n")
                    f.write(f"Speedup: {test['speedup']:.2f}x\n")
                    f.write(f"Improvement: {test['improvement_percentage']:.1f}%\n\n")
                    
                    # Stage breakdown
                    f.write("Stage breakdown:\n")
                    all_stages = set(list(test['original'].get('stages', {}).keys()) + 
                                   list(test['optimized'].get('stages', {}).keys()))
                    
                    for stage in sorted(all_stages):
                        orig_time = test['original']['stages'].get(stage, 0)
                        opt_time = test['optimized']['stages'].get(stage, 0)
                        if orig_time > 0:
                            stage_speedup = orig_time / opt_time if opt_time > 0 else float('inf')
                            f.write(f"  {stage:20s}: {orig_time:6.3f}s → {opt_time:6.3f}s ")
                            f.write(f"({stage_speedup:.2f}x)\n")
                else:
                    if not test['original']['success']:
                        f.write(f"Original failed: {test['original'].get('error', 'Unknown error')}\n")
                    if not test['optimized']['success']:
                        f.write(f"Optimized failed: {test['optimized'].get('error', 'Unknown error')}\n")
                
                f.write("\n")
            
            # Optimization techniques summary
            f.write("OPTIMIZATION TECHNIQUES APPLIED\n")
            f.write("=" * 70 + "\n")
            f.write("1. Parallel Processing:\n")
            f.write("   - Parallel file scanning with ThreadPoolExecutor\n")
            f.write("   - Parallel tile processing in batches\n")
            f.write("   - Concurrent channel intensity pre-scanning\n\n")
            
            f.write("2. I/O Optimizations:\n")
            f.write("   - Memory-mapped file access for large images\n")
            f.write("   - Batch CSV reading\n")
            f.write("   - Reduced file system calls\n\n")
            
            f.write("3. Algorithm Improvements:\n")
            f.write("   - Fast area averaging for large downsampling factors\n")
            f.write("   - Optimized 8-bit conversion with vectorized operations\n")
            f.write("   - Smart caching of channel intensities\n\n")
            
            f.write("4. Memory Efficiency:\n")
            f.write("   - Process tiles in batches to manage memory\n")
            f.write("   - Use memory mapping instead of loading full images\n")
            f.write("   - Efficient numpy operations\n\n")
        
        print(f"Detailed report saved to: {report_file}")


def main():
    """Run performance comparison."""
    if len(sys.argv) < 2:
        print("Usage: python compare_performance.py <acquisition_dir>")
        print("\nThis script compares the performance of the original and optimized")
        print("downsampler implementations.")
        sys.exit(1)
    
    acquisition_dir = Path(sys.argv[1])
    
    if not acquisition_dir.exists():
        print(f"Error: Directory {acquisition_dir} does not exist")
        sys.exit(1)
    
    # Create comparison object
    comparison = PerformanceComparison()
    
    print("Starting performance comparison...")
    print(f"Acquisition directory: {acquisition_dir}")
    print("\nThis will test both implementations with various configurations.")
    print("Tests include different tile sizes and cache settings.\n")
    
    # Run comparison
    results = comparison.compare_implementations(
        acquisition_dir,
        tile_sizes=[50, 75, 100],  # Test different tile sizes
        test_cache=True,  # Test with and without cache
        test_regions=False  # Can enable region-specific tests if needed
    )
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {comparison.output_dir}")
    print("\nKey findings:")
    
    # Print summary of improvements
    successful_tests = [t for t in results['tests'] 
                      if t['original']['success'] and t['optimized']['success']]
    
    if successful_tests:
        avg_speedup = np.mean([t['speedup'] for t in successful_tests])
        max_speedup = max(t['speedup'] for t in successful_tests)
        avg_improvement = np.mean([t['improvement_percentage'] for t in successful_tests])
        
        print(f"- Average speedup: {avg_speedup:.2f}x")
        print(f"- Maximum speedup: {max_speedup:.2f}x")
        print(f"- Average performance improvement: {avg_improvement:.1f}%")
        
        # Identify which stages benefited most
        stage_improvements = {}
        for test in successful_tests:
            orig_stages = test['original'].get('stages', {})
            opt_stages = test['optimized'].get('stages', {})
            
            for stage in orig_stages:
                if stage in opt_stages and orig_stages[stage] > 0:
                    improvement = (orig_stages[stage] - opt_stages[stage]) / orig_stages[stage] * 100
                    if stage not in stage_improvements:
                        stage_improvements[stage] = []
                    stage_improvements[stage].append(improvement)
        
        print("\nTop improved stages:")
        stage_avg_improvements = [(stage, np.mean(improvements)) 
                                for stage, improvements in stage_improvements.items()]
        stage_avg_improvements.sort(key=lambda x: x[1], reverse=True)
        
        for stage, improvement in stage_avg_improvements[:3]:
            print(f"- {stage}: {improvement:.1f}% improvement")


if __name__ == "__main__":
    main()