#!/usr/bin/env python3
"""
Use Standard Benchmark Test Data Files
======================================
Demonstrates how to use the downloaded M_X_D30.txt benchmark test data files
for optimization algorithm testing with proper input data and transformations.
These files contain:
- M_X_D30.txt: 30x30 rotation matrices for each function (X = 1 to 30)
- shift_data_X_D30.txt: Shift vectors for function transformations
- fX_o.txt: Bias values for each function
Author: Medical AI Research Team
Version: 1.0.0
"""
import numpy as np
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Callable
import json
import time
# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'algorithms'))
from original_hbo import OriginalHBO
from gohbo_algorithm import GOHBO
from benchmark_functions import BenchmarkSuite
class StandardBenchmarkTester:
"""
Test optimization algorithms using standard benchmark test data files
"""
def __init__(self, test_data_dir: str = "./benchmark_test_data",
results_dir: str = "./standard_benchmark_results",
dimension: int = 30):
"""
Initialize tester with standard test data
Args:
test_data_dir: Directory containing M_X_D30.txt files
results_dir: Directory to save results
dimension: Problem dimension (30 for standard testing)
"""
self.test_data_dir = Path(test_data_dir)
self.results_dir = Path(results_dir)
self.dimension = dimension
# Create results directory
self.results_dir.mkdir(parents=True, exist_ok=True)
# Load benchmark functions
self.benchmark_suite = BenchmarkSuite(dimension=dimension)
# Cache for loaded test data
self.rotation_matrices = {}
self.shift_vectors = {}
self.bias_values = {}
print(f"StandardBenchmarkTester initialized")
print(f"Test data directory: {self.test_data_dir}")
print(f"Results directory: {self.results_dir}")
print(f"Dimension: {self.dimension}")
def load_test_data(self, function_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
"""
Load standard test data for specific function
Args:
function_id: Function ID (1-30)
Returns:
Tuple of (rotation_matrix, shift_vector, bias_value)
"""
# Load rotation matrix
matrix_file = self.test_data_dir / f"M_{function_id}_D{self.dimension}.txt"
if matrix_file.exists():
rotation_matrix = np.loadtxt(matrix_file)
else:
print(f"Matrix file not found: {matrix_file}")
# Generate identity matrix as fallback
rotation_matrix = np.eye(self.dimension)
# Load shift vector
shift_file = self.test_data_dir / f"shift_data_{function_id}_D{self.dimension}.txt"
if shift_file.exists():
shift_vector = np.loadtxt(shift_file)
else:
print(f"Shift file not found: {shift_file}")
# Generate zero shift as fallback
shift_vector = np.zeros(self.dimension)
# Load bias value
bias_file = self.test_data_dir / f"f{function_id}_o.txt"
if bias_file.exists():
with open(bias_file, 'r') as f:
bias_value = float(f.read().strip())
else:
print(f"Bias file not found: {bias_file}")
bias_value = 0.0
return rotation_matrix, shift_vector, bias_value
def create_transformed_function(self, original_function: Callable,
rotation_matrix: np.ndarray,
shift_vector: np.ndarray,
bias_value: float) -> Callable:
"""
Create transformed function using standard test data
Args:
original_function: Original benchmark function
rotation_matrix: Rotation matrix from M_X_D30.txt
shift_vector: Shift vector from shift_data_X_D30.txt
bias_value: Bias value from fX_o.txt
Returns:
Transformed function f(x) = original_function(M * (x - shift)) + bias
"""
def transformed_function(x):
# Ensure x has correct dimension
x = np.array(x)
if len(x) != len(shift_vector):
# Pad or truncate to match
if len(x) < len(shift_vector):
x = np.pad(x, (0, len(shift_vector) - len(x)), mode='constant')
else:
x = x[:len(shift_vector)]
# Apply transformation: M * (x - shift) + bias
shifted = x - shift_vector
rotated = rotation_matrix @ shifted
result = original_function(rotated) + bias_value
return result
return transformed_function
def test_algorithm_on_function(self, algorithm_class,
function_id: int,
num_runs: int = 30,
max_iterations: int = 1000,
population_size: int = 30) -> Dict:
"""
Test algorithm on specific function using standard test data
Args:
algorithm_class: Algorithm class (OriginalHBO or GOHBO)
function_id: Function ID (1-30)
num_runs: Number of independent runs
max_iterations: Maximum iterations per run
population_size: Algorithm population size
Returns:
Dictionary with test results
"""
print(f"\nTesting {algorithm_class.__name__} on F{function_id}")
# Get original benchmark function
benchmark_func = self.benchmark_suite.get_function(function_id)
if not benchmark_func:
raise ValueError(f"Function {function_id} not found")
# Load standard test data
rotation_matrix, shift_vector, bias_value = self.load_test_data(function_id)
print(f" Using rotation matrix: {rotation_matrix.shape}")
print(f" Using shift vector: {shift_vector.shape}")
print(f" Using bias value: {bias_value:.6e}")
# Create transformed function
transformed_function = self.create_transformed_function(
benchmark_func.function,
rotation_matrix,
shift_vector,
bias_value
)
# Run algorithm multiple times
results = []
start_time = time.time()
for run in range(num_runs):
print(f" Run {run + 1}/{num_runs}...", end=" ")
# Initialize algorithm with transformed function
algorithm = algorithm_class(
objective_function=transformed_function,
dimension=self.dimension,
bounds=(benchmark_func.bounds[0] * np.ones(self.dimension),
benchmark_func.bounds[1] * np.ones(self.dimension)),
population_size=population_size,
max_iterations=max_iterations,
verbose=False
)
# Run optimization
run_start = time.time()
result = algorithm.optimize()
run_time = time.time() - run_start
# Store results
run_result = {
'run': run + 1,
'best_fitness': result['best_fitness'],
'best_position': result['best_position'].tolist(),
'execution_time': run_time,
'convergence_history': result['convergence_history'].tolist()
}
results.append(run_result)
print(f"Fitness: {result['best_fitness']:.6e}")
total_time = time.time() - start_time
# Calculate statistics
fitness_values = [r['best_fitness'] for r in results]
execution_times = [r['execution_time'] for r in results]
statistics = {
'function_id': function_id,
'function_name': benchmark_func.name,
'algorithm': algorithm_class.__name__,
'dimension': self.dimension,
'num_runs': num_runs,
'test_data_used': {
'rotation_matrix_file': f"M_{function_id}_D{self.dimension}.txt",
'shift_vector_file': f"shift_data_{function_id}_D{self.dimension}.txt",
'bias_file': f"f{function_id}_o.txt",
'bias_value': bias_value
},
'statistics': {
'best_fitness': float(np.min(fitness_values)),
'worst_fitness': float(np.max(fitness_values)),
'mean_fitness': float(np.mean(fitness_values)),
'median_fitness': float(np.median(fitness_values)),
'std_fitness': float(np.std(fitness_values)),
'mean_execution_time': float(np.mean(execution_times)),
'total_time': total_time
},
'raw_results': results
}
# Save individual function results
results_file = self.results_dir / f"{algorithm_class.__name__}_F{function_id}_D{self.dimension}_results.json"
with open(results_file, 'w') as f:
json.dump(statistics, f, indent=2)
print(f" Results saved to: {results_file}")
return statistics
def test_algorithm_on_all_functions(self, algorithm_class,
function_range: Tuple[int, int] = (1, 30),
num_runs: int = 30) -> Dict:
"""
Test algorithm on all functions using standard test data
Args:
algorithm_class: Algorithm class to test
function_range: Range of functions to test (start, end+1)
num_runs: Number of runs per function
Returns:
Complete test results
"""
print(f"\nTesting {algorithm_class.__name__} on Functions {function_range[0]}-{function_range[1]-1}")
print("="*80)
all_results = {
'algorithm': algorithm_class.__name__,
'test_configuration': {
'dimension': self.dimension,
'function_range': function_range,
'num_runs': num_runs,
'test_data_directory': str(self.test_data_dir)
},
'function_results': {},
'summary_statistics': {}
}
# Test each function
for func_id in range(function_range[0], function_range[1]):
try:
func_results = self.test_algorithm_on_function(
algorithm_class, func_id, num_runs=num_runs
)
all_results['function_results'][f'F{func_id}'] = func_results
except Exception as e:
print(f"Failed to test F{func_id}: {str(e)}")
all_results['function_results'][f'F{func_id}'] = {'error': str(e)}
# Calculate summary statistics
successful_functions = []
all_best_fitness = []
all_mean_fitness = []
for func_name, func_result in all_results['function_results'].items():
if 'error' not in func_result:
successful_functions.append(func_name)
all_best_fitness.append(func_result['statistics']['best_fitness'])
all_mean_fitness.append(func_result['statistics']['mean_fitness'])
if successful_functions:
all_results['summary_statistics'] = {
'total_functions_tested': len(all_results['function_results']),
'successful_functions': len(successful_functions),
'failed_functions': len(all_results['function_results']) - len(successful_functions),
'overall_best_fitness': float(np.min(all_best_fitness)),
'overall_worst_fitness': float(np.max(all_best_fitness)),
'average_best_fitness': float(np.mean(all_best_fitness)),
'average_mean_fitness': float(np.mean(all_mean_fitness)),
'successful_function_list': successful_functions
}
# Save complete results
complete_results_file = self.results_dir / f"{algorithm_class.__name__}_complete_results.json"
with open(complete_results_file, 'w') as f:
json.dump(all_results, f, indent=2)
print(f"\nCOMPLETE TEST SUMMARY for {algorithm_class.__name__}:")
print(f" Functions tested: {len(all_results['function_results'])}")
print(f" Successful: {len(successful_functions)}")
print(f" Failed: {len(all_results['function_results']) - len(successful_functions)}")
if successful_functions:
print(f" Average best fitness: {all_results['summary_statistics']['average_best_fitness']:.6e}")
print(f" Overall best fitness: {all_results['summary_statistics']['overall_best_fitness']:.6e}")
print(f" Complete results saved: {complete_results_file}")
return all_results
def compare_algorithms(self, function_range: Tuple[int, int] = (1, 11)) -> Dict:
"""
Compare HBO vs GOHBO using standard test data
Args:
function_range: Range of functions to test
Returns:
Comparison results
"""
print(f"\n ALGORITHM COMPARISON using Standard Test Data")
print("="*80)
# Test both algorithms
hbo_results = self.test_algorithm_on_all_functions(OriginalHBO, function_range, num_runs=10)
gohbo_results = self.test_algorithm_on_all_functions(GOHBO, function_range, num_runs=10)
# Create comparison
comparison = {
'test_configuration': {
'dimension': self.dimension,
'function_range': function_range,
'test_data_directory': str(self.test_data_dir),
'algorithms_compared': ['HBO', 'GOHBO']
},
'function_comparison': {},
'overall_comparison': {}
}
# Compare function by function
hbo_wins = 0
gohbo_wins = 0
for func_id in range(function_range[0], function_range[1]):
func_key = f'F{func_id}'
if (func_key in hbo_results['function_results'] and func_key in gohbo_results['function_results']):
hbo_best = hbo_results['function_results'][func_key]['statistics']['best_fitness']
gohbo_best = gohbo_results['function_results'][func_key]['statistics']['best_fitness']
winner = 'GOHBO' if gohbo_best < hbo_best else 'HBO'
improvement = abs(hbo_best - gohbo_best) / max(abs(hbo_best), 1e-16) * 100
comparison['function_comparison'][func_key] = {
'hbo_best_fitness': hbo_best,
'gohbo_best_fitness': gohbo_best,
'winner': winner,
'improvement_percent': improvement
}
if winner == 'GOHBO':
gohbo_wins += 1
else:
hbo_wins += 1
# Overall comparison
comparison['overall_comparison'] = {
'hbo_wins': hbo_wins,
'gohbo_wins': gohbo_wins,
'total_comparisons': hbo_wins + gohbo_wins,
'gohbo_win_rate': gohbo_wins / (hbo_wins + gohbo_wins) * 100 if (hbo_wins + gohbo_wins) > 0 else 0
}
# Save comparison
comparison_file = self.results_dir / 'algorithm_comparison_standard_data.json'
with open(comparison_file, 'w') as f:
json.dump(comparison, f, indent=2)
print(f"\nFINAL COMPARISON RESULTS:")
print(f" HBO wins: {hbo_wins}")
print(f" GOHBO wins: {gohbo_wins}")
print(f" GOHBO win rate: {comparison['overall_comparison']['gohbo_win_rate']:.1f}%")
print(f" Comparison saved: {comparison_file}")
return comparison
def main():
"""Main function to demonstrate standard benchmark testing"""
import argparse
parser = argparse.ArgumentParser(
description="Test algorithms using standard benchmark test data files",
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
Examples:
python use_benchmark_test_data.py --single 1 # Test single function F1
python use_benchmark_test_data.py --compare # Compare HBO vs GOHBO
python use_benchmark_test_data.py --all-functions # Test all functions (1-30)
"""
)
parser.add_argument('--test-data-dir', type=str, default='./benchmark_test_data',
help='Directory containing M_X_D30.txt files')
parser.add_argument('--single', type=int, metavar='FUNC_ID',
help='Test single function ID (1-30)')
parser.add_argument('--compare', action='store_true',
help='Compare HBO vs GOHBO on functions 1-10')
parser.add_argument('--all-functions', action='store_true',
help='Test all functions 1-30')
args = parser.parse_args()
print("Standard Benchmark Test Data Usage")
print("="*50)
print("Using M_X_D30.txt files for algorithm testing")
print("="*50)
# Initialize tester
tester = StandardBenchmarkTester(
test_data_dir=args.test_data_dir,
results_dir="./standard_benchmark_results"
)
if args.single:
# Test single function
print(f"\nTesting single function F{args.single}")
hbo_result = tester.test_algorithm_on_function(OriginalHBO, args.single, num_runs=5)
gohbo_result = tester.test_algorithm_on_function(GOHBO, args.single, num_runs=5)
print(f"\nRESULTS for F{args.single}:")
print(f" HBO best: {hbo_result['statistics']['best_fitness']:.6e}")
print(f" GOHBO best: {gohbo_result['statistics']['best_fitness']:.6e}")
elif args.compare:
# Compare algorithms
tester.compare_algorithms(function_range=(1, 11))
elif args.all_functions:
# Test all functions
tester.test_algorithm_on_all_functions(OriginalHBO, function_range=(1, 31), num_runs=5)
tester.test_algorithm_on_all_functions(GOHBO, function_range=(1, 31), num_runs=5)
else:
# Default: quick comparison
print("\nRunning default comparison on functions 1-5...")
tester.compare_algorithms(function_range=(1, 6))
if __name__ == "__main__":
main()