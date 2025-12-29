#!/usr/bin/env python3
"""
30 Functions Benchmark Test - Zhang et al. 2024 Format
======================================================
"""
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import numpy as np
# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from algorithm_testing_framework import AlgorithmTestingFramework
from original_hbo import OriginalHBO
from gohbo_algorithm import GOHBO
from benchmark_functions import BenchmarkSuite
def run_comprehensive_benchmark(output_dir: str = "./30_functions_results",
dimension: int = 30,
num_runs: int = 30,
max_iterations: int = 1000,
population_size: int = 40,
test_subset: bool = False):
"""
Run comprehensive benchmark testing
Args:
output_dir: Directory to save all results
dimension: Problem dimension (30 for Zhang et al. format)
num_runs: Number of independent runs per function (30 standard)
max_iterations: Maximum iterations per run
population_size: Algorithm population size
test_subset: If True, test only first 5 functions for quick validation
"""
print(" Starting Comprehensive 30 Functions Benchmark Test")
print("=" * 80)
print(f" Output Directory: {output_dir}")
print(f" Dimension: {dimension}")
print(f"Runs per Function: {num_runs}")
print(f"Max Iterations: {max_iterations}")
print(f" Population Size: {population_size}")
print(f"Test Mode: {'Subset (5 functions)' if test_subset else 'Complete (30 functions)'}")
print("=" * 80)
# Initialize testing framework
framework = AlgorithmTestingFramework(
output_dir=output_dir,
dimension=dimension,
num_runs=num_runs,
max_iterations=max_iterations,
population_size=population_size,
verbose=True
)
# Determine which functions to test
if test_subset:
function_ids = list(range(1, 6)) # Test F1-F5 only
print("Running subset test on functions F1-F5")
else:
function_ids = list(range(1, 31)) # Test F1-F30
print("Running complete test on functions F1-F30")
print(f"Testing {len(function_ids)} functions: {function_ids}")
print()
# Algorithm configurations
algorithms_to_test = [
{
'class': OriginalHBO,
'name': 'Original_HBO',
'kwargs': {}
},
{
'class': GOHBO,
'name': 'GOHBO', 'kwargs': {
'gwo_weight': 0.4,
'hbo_weight': 0.4,
'ol_weight': 0.2,
'ol_frequency': 10
}
}
]
# Store all results
all_algorithm_results = {}
# Test each algorithm
for algo_config in algorithms_to_test:
print(f"\n Testing {algo_config['name']} Algorithm")
print("-" * 60)
start_time = time.time()
# Run algorithm test
try:
algorithm_results = framework.run_algorithm_test(
algorithm_class=algo_config['class'],
algorithm_name=algo_config['name'],
function_ids=function_ids,
**algo_config['kwargs']
)
all_algorithm_results[algo_config['name']] = algorithm_results
# Calculate summary statistics
success_rates = [stats.success_rate for stats in algorithm_results.values()]
mean_fitness_values = [stats.mean_fitness for stats in algorithm_results.values() if stats.mean_fitness != float('inf')]
execution_time = time.time() - start_time
print(f"\n{algo_config['name']} Testing Complete!")
print(f" Total Time: {execution_time:.1f} seconds")
print(f" Average Success Rate: {np.mean(success_rates):.1%}")
print(f" Functions Solved (>90% success): {sum(1 for rate in success_rates if rate > 0.9)}/{len(function_ids)}")
if mean_fitness_values:
print(f" Average Fitness (successful): {np.mean(mean_fitness_values):.6e}")
except Exception as e:
print(f"Error testing {algo_config['name']}: {str(e)}")
import traceback
traceback.print_exc()
# Compare algorithms if multiple were tested
if len(all_algorithm_results) > 1:
print(f"\n Algorithm Comparison Analysis")
print("-" * 60)
try:
comparison_results = framework.compare_algorithms(list(all_algorithm_results.keys()))
print("Overall Performance Ranking:")
for algo_name, ranking in comparison_results['overall_ranking'].items():
print(f" {algo_name:15s}: {ranking['average_success_rate']:.1%} success, "
f"{ranking['functions_solved']} functions solved")
# Function-wise winner analysis
print("\nFunction-wise Winners:")
winners = {}
for func_key, func_results in comparison_results['function_comparison'].items():
best_algo = None
best_success = 0.0
for algo_name, results in func_results.items():
if results['success_rate'] > best_success:
best_success = results['success_rate']
best_algo = algo_name
if best_algo:
if best_algo not in winners:
winners[best_algo] = 0
winners[best_algo] += 1
print(f" {func_key}: {best_algo} ({best_success:.1%})")
print(f"\n Winner Summary:")
for algo_name, count in winners.items():
print(f" {algo_name}: {count}/{len(function_ids)} functions won")
except Exception as e:
print(f"Error in comparison analysis: {str(e)}")
# Generate comprehensive report
print(f"\nGenerating Comprehensive Report")
print("-" * 60)
report_file = Path(output_dir) / "benchmark_report.txt"
with open(report_file, 'w') as f:
f.write("30 Functions Benchmark Test Results\n")
f.write("=" * 80 + "\n")
f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
f.write(f"Test Configuration:\n")
f.write(f" - Dimension: {dimension}\n")
f.write(f" - Runs per Function: {num_runs}\n")
f.write(f" - Max Iterations: {max_iterations}\n")
f.write(f" - Population Size: {population_size}\n")
f.write(f" - Functions Tested: {len(function_ids)}\n\n")
# Algorithm summary
for algo_name, results in all_algorithm_results.items():
f.write(f"Algorithm: {algo_name}\n")
f.write("-" * 40 + "\n")
success_rates = [stats.success_rate for stats in results.values()]
execution_times = [stats.mean_execution_time for stats in results.values()]
f.write(f"Overall Performance:\n")
f.write(f" - Average Success Rate: {np.mean(success_rates):.3f}\n")
f.write(f" - Functions Solved (>90%): {sum(1 for rate in success_rates if rate > 0.9)}\n")
f.write(f" - Average Execution Time: {np.mean(execution_times):.3f} seconds\n\n")
f.write("Function-wise Results:\n")
for func_id in sorted(results.keys()):
stats = results[func_id]
f.write(f" F{func_id:2d} ({stats.function_name:15s}): ")
f.write(f"Best={stats.best_fitness:.6e}, ")
f.write(f"Mean={stats.mean_fitness:.6e}, ")
f.write(f"Success={stats.success_rate:.1%}\n")
f.write("\n")
print(f"Comprehensive report saved: {report_file}")
# Final summary
print(f"\nBenchmark Testing Complete!")
print("=" * 80)
print(f" All results saved in: {output_dir}")
print(f"Individual function results: M_X_D{dimension}.txt format")
print(f"Algorithm summaries: [Algorithm]_summary.json")
print(f" Comparison results: algorithm_comparison.json")
print(f"Comprehensive report: benchmark_report.txt")
if test_subset:
print("\nThis was a subset test. For complete results, run without --test-subset flag.")
else:
print(f"\nComplete 30 functions benchmark test finished successfully!")
return all_algorithm_results
def main():
"""Main function with command-line interface"""
parser = argparse.ArgumentParser(
description='30 Functions Benchmark Test for HBO and GOHBO Algorithms',
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
Examples:
%(prog)s # Full test (30 functions, D=30, 30 runs each)
%(prog)s --test-subset # Quick test (5 functions only)
%(prog)s --dimension 10 --runs 10 # Reduced test for development
%(prog)s --output-dir ./my_results # Custom output directory
Full test will take several hours. Use --test-subset for validation.
"""
)
# Test configuration
parser.add_argument('--output-dir', type=str, default='./30_functions_results',
help='Output directory for results (default: ./30_functions_results)')
parser.add_argument('--dimension', type=int, default=30,
help='Problem dimension (default: 30)')
parser.add_argument('--runs', type=int, default=30,
help='Number of runs per function (default: 30)')
parser.add_argument('--iterations', type=int, default=1000,
help='Maximum iterations per run (default: 1000)')
parser.add_argument('--population', type=int, default=40,
help='Population size (default: 40)')
# Test mode options
parser.add_argument('--test-subset', action='store_true',
help='Test only first 5 functions for quick validation')
parser.add_argument('--verbose', action='store_true', default=True,
help='Enable verbose output (default: True)')
args = parser.parse_args()
# Validate arguments
if args.dimension < 2:
parser.error("Dimension must be at least 2")
if args.runs < 1:
parser.error("Number of runs must be at least 1")
if args.iterations < 10:
parser.error("Max iterations must be at least 10")
if args.population < 10:
parser.error("Population size must be at least 10")
# Confirm long-running test
if not args.test_subset:
print("WARNING: Full 30 functions test will take several hours!")
print(f" Testing {args.runs} runs × 30 functions × 2 algorithms = {args.runs * 30 * 2} total experiments")
print(f" Estimated time: 2-4 hours depending on hardware")
print()
response = input("Continue with full test? (y/N): ")
if response.lower() not in ['y', 'yes']:
print("Test cancelled. Use --test-subset for quick validation.")
return
# Run benchmark
start_time = time.time()
try:
results = run_comprehensive_benchmark(
output_dir=args.output_dir,
dimension=args.dimension,
num_runs=args.runs,
max_iterations=args.iterations,
population_size=args.population,
test_subset=args.test_subset
)
total_time = time.time() - start_time
print(f"\nTotal Execution Time: {total_time/60:.1f} minutes ({total_time:.1f} seconds)")
except KeyboardInterrupt:
print("\n Test interrupted by user")
except Exception as e:
print(f"\nTest failed with error: {str(e)}")
import traceback
traceback.print_exc()
if __name__ == "__main__":
main()