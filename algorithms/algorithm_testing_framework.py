#!/usr/bin/env python3
import numpy as np
import time
import json
import os
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, asdict
try:
import matplotlib.pyplot as plt
import seaborn as sns
PLOTTING_AVAILABLE = True
except ImportError:
PLOTTING_AVAILABLE = False
print("Matplotlib/Seaborn not available. Plotting functionality disabled.")
from datetime import datetime
from benchmark_functions import BenchmarkSuite
from original_hbo import OriginalHBO
from gohbo_algorithm import GOHBO
@dataclass
class ExperimentResults:
"""Container for single experiment results"""
algorithm: str
function_id: int
function_name: str
dimension: int
run_number: int
best_fitness: float
final_fitness: float
convergence_history: List[float]
execution_time: float
success: bool
iterations_to_best: int
@dataclass
class StatisticalResults:
"""Statistical summary of multiple runs"""
algorithm: str
function_id: int
function_name: str
dimension: int
total_runs: int
successful_runs: int
success_rate: float
best_fitness: float
worst_fitness: float
mean_fitness: float
median_fitness: float
std_fitness: float
mean_execution_time: float
std_execution_time: float
convergence_rate: float
final_diversity: float
class AlgorithmTestingFramework:
"""
Comprehensive testing framework for optimization algorithms
Tests algorithms on 30 benchmark functions with statistical analysis
and result formatting matching Zhang et al. 2024 paper.
"""
def __init__(self, output_dir: str = "./benchmark_results",
test_data_dir: str = "./benchmark_test_data",
dimension: int = 30,
num_runs: int = 30,
max_iterations: int = 1000,
population_size: int = 40,
verbose: bool = True):
"""
Initialize testing framework
Args:
output_dir: Directory to save results
test_data_dir: Directory containing M_X_D30.txt test data files
dimension: Problem dimension
num_runs: Number of independent runs per function
max_iterations: Maximum iterations per run
population_size: Algorithm population size
verbose: Enable detailed logging
"""
self.output_dir = Path(output_dir)
self.test_data_dir = Path(test_data_dir)
self.dimension = dimension
self.num_runs = num_runs
self.max_iterations = max_iterations
self.population_size = population_size
self.verbose = verbose
# Create output directory
self.output_dir.mkdir(parents=True, exist_ok=True)
# Initialize benchmark suite
self.benchmark_suite = BenchmarkSuite(dimension=dimension)
# Setup logging
self._setup_logging()
# Check for standard test data files
self.use_standard_test_data = self._check_test_data_availability()
if self.use_standard_test_data:
self.logger.info(f"Using standard test data from: {self.test_data_dir}")
else:
self.logger.warning(f"WARNING: Standard test data not found. Using random initialization.")
# Storage for results
self.experiment_results: List[ExperimentResults] = []
self.statistical_results: Dict[str, Dict[int, StatisticalResults]] = {}
self.dimension_warnings_shown = set() # Track which functions we've warned about
self.logger.info(f"Algorithm Testing Framework Initialized")
self.logger.info(f"Output Directory: {self.output_dir}")
self.logger.info(f"Dimension: {self.dimension}")
self.logger.info(f"Runs per Function: {self.num_runs}")
self.logger.info(f"Max Iterations: {self.max_iterations}")
def _setup_logging(self):
"""Setup logging configuration"""
if self.verbose:
logging.basicConfig(
level=logging.INFO,
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
handlers=[
logging.FileHandler(self.output_dir / 'testing_framework.log'),
logging.StreamHandler()
]
)
else:
logging.basicConfig(level=logging.WARNING)
self.logger = logging.getLogger(__name__)
def _check_test_data_availability(self) -> bool:
"""Check if standard test data files are available"""
if not self.test_data_dir.exists():
return False
# Check for at least a few key files
required_files = [
f"M_1_D{self.dimension}.txt",
f"M_2_D{self.dimension}.txt",
f"shift_data_1_D{self.dimension}.txt",
f"shift_data_2_D{self.dimension}.txt"
]
for filename in required_files:
if not (self.test_data_dir / filename).exists():
return False
return True
def _load_test_data(self, function_id: int) -> Tuple[np.ndarray, np.ndarray, float]:
"""Load standard test data for specific function"""
# Load rotation matrix
matrix_file = self.test_data_dir / f"M_{function_id}_D{self.dimension}.txt"
if matrix_file.exists():
rotation_matrix = np.loadtxt(matrix_file)
else:
rotation_matrix = np.eye(self.dimension)
# Load shift vector
shift_file = self.test_data_dir / f"shift_data_{function_id}_D{self.dimension}.txt"
if shift_file.exists():
shift_vector = np.loadtxt(shift_file)
else:
shift_vector = np.zeros(self.dimension)
# Load bias value
bias_file = self.test_data_dir / f"f{function_id}_o.txt"
if bias_file.exists():
with open(bias_file, 'r') as f:
bias_value = float(f.read().strip())
else:
bias_value = 0.0
return rotation_matrix, shift_vector, bias_value
def _create_transformed_function(self, original_function: Callable,
rotation_matrix: np.ndarray,
shift_vector: np.ndarray,
bias_value: float) -> Callable:
"""Create transformed function using standard test data"""
def transformed_function(x):
x = np.array(x)
# Ensure x has correct dimension
if len(x) != len(shift_vector):
if len(x) < len(shift_vector):
x = np.pad(x, (0, len(shift_vector) - len(x)), mode='constant')
else:
x = x[:len(shift_vector)]
# Apply transformation: f(M * (x - shift)) + bias
shifted = x - shift_vector
rotated = rotation_matrix @ shifted
result = original_function(rotated) + bias_value
return result
return transformed_function
def run_single_experiment(self,
algorithm_class,
algorithm_name: str,
function_id: int,
run_number: int,
**algorithm_kwargs) -> ExperimentResults:
"""
Run single optimization experiment
Args:
algorithm_class: Algorithm class to instantiate
algorithm_name: Name of algorithm for logging
function_id: Benchmark function ID (1-30)
run_number: Current run number
**algorithm_kwargs: Additional algorithm parameters
Returns:
ExperimentResults object with experiment data
"""
# Get benchmark function
benchmark_func = self.benchmark_suite.get_function(function_id)
if not benchmark_func:
raise ValueError(f"Function {function_id} not found")
# Determine effective dimension for function
effective_dim = min(benchmark_func.dimension, self.dimension)
if benchmark_func.dimension < self.dimension and function_id not in self.dimension_warnings_shown:
self.logger.warning(f"F{function_id} ({benchmark_func.name}) uses dimension {benchmark_func.dimension} < {self.dimension}")
self.dimension_warnings_shown.add(function_id)
# Setup bounds
lower_bounds = np.full(effective_dim, benchmark_func.bounds[0])
upper_bounds = np.full(effective_dim, benchmark_func.bounds[1])
# Create objective function (with or without standard test data)
if self.use_standard_test_data:
# Load standard test data and create transformed function
rotation_matrix, shift_vector, bias_value = self._load_test_data(function_id)
objective_function = self._create_transformed_function(
benchmark_func.function, rotation_matrix, shift_vector, bias_value
)
self.logger.debug(f"F{function_id}: Using transformed function with test data")
else:
# Standard wrapper function for dimension handling
def objective_wrapper(x):
if len(x) > benchmark_func.dimension:
return benchmark_func.function(x[:benchmark_func.dimension])
else:
return benchmark_func.function(x)
objective_function = objective_wrapper
try:
# Initialize algorithm
start_time = time.time()
algorithm = algorithm_class(
objective_function=objective_function,
dimension=effective_dim,
bounds=(lower_bounds, upper_bounds),
population_size=self.population_size,
max_iterations=self.max_iterations,
verbose=False,
**algorithm_kwargs
)
# Run optimization
results = algorithm.optimize()
execution_time = time.time() - start_time
# Extract results
best_fitness = results['best_fitness']
convergence_history = results['convergence_history'].tolist()
iterations_to_best = np.argmin(results['convergence_history']) + 1
# Determine success (within tolerance of global optimum)
tolerance = 1e-6 if benchmark_func.global_optimum == 0.0 else abs(benchmark_func.global_optimum) * 1e-3
success = abs(best_fitness - benchmark_func.global_optimum) < tolerance
# Create experiment result
experiment_result = ExperimentResults(
algorithm=algorithm_name,
function_id=function_id,
function_name=benchmark_func.name,
dimension=effective_dim,
run_number=run_number,
best_fitness=best_fitness,
final_fitness=convergence_history[-1] if convergence_history else best_fitness,
convergence_history=convergence_history,
execution_time=execution_time,
success=success,
iterations_to_best=iterations_to_best
)
return experiment_result
except Exception as e:
self.logger.error(f"Error in {algorithm_name} F{function_id} Run {run_number}: {str(e)}")
# Return failed experiment result
return ExperimentResults(
algorithm=algorithm_name,
function_id=function_id,
function_name=benchmark_func.name,
dimension=effective_dim,
run_number=run_number,
best_fitness=float('inf'),
final_fitness=float('inf'),
convergence_history=[],
execution_time=0.0,
success=False,
iterations_to_best=0
)
def run_algorithm_test(self,
algorithm_class,
algorithm_name: str,
function_ids: Optional[List[int]] = None,
**algorithm_kwargs) -> Dict[int, StatisticalResults]:
"""
Run complete test of algorithm on specified functions
Args:
algorithm_class: Algorithm class to test
algorithm_name: Name of algorithm
function_ids: List of function IDs to test (default: 1-30)
**algorithm_kwargs: Additional algorithm parameters
Returns:
Dictionary mapping function_id -> StatisticalResults
"""
if function_ids is None:
function_ids = list(range(1, 31)) # Functions 1-30
self.logger.info(f"Starting {algorithm_name} testing on {len(function_ids)} functions")
algorithm_results = {}
for function_id in function_ids:
self.logger.info(f"Testing {algorithm_name} on F{function_id}")
# Run multiple independent experiments
function_experiments = []
for run in range(self.num_runs):
if self.verbose:
print(f" Run {run + 1}/{self.num_runs}", end='\r')
experiment_result = self.run_single_experiment(
algorithm_class=algorithm_class,
algorithm_name=algorithm_name,
function_id=function_id,
run_number=run + 1,
**algorithm_kwargs
)
function_experiments.append(experiment_result)
self.experiment_results.append(experiment_result)
print() # New line after progress
# Calculate statistics
stats = self._calculate_statistics(function_experiments)
algorithm_results[function_id] = stats
# Save individual function results
self._save_function_results(function_experiments, algorithm_name, function_id)
self.logger.info(f"F{function_id} Complete - Mean: {stats.mean_fitness:.6e}, Success: {stats.success_rate:.1%}")
# Store results
self.statistical_results[algorithm_name] = algorithm_results
# Save complete algorithm results
self._save_algorithm_summary(algorithm_results, algorithm_name)
self.logger.info(f"{algorithm_name} testing complete")
return algorithm_results
def _calculate_statistics(self, experiments: List[ExperimentResults]) -> StatisticalResults:
"""Calculate statistical summary from multiple experiments"""
if not experiments:
return None
# Extract fitness values
fitness_values = [exp.best_fitness for exp in experiments if exp.success or exp.best_fitness != float('inf')]
execution_times = [exp.execution_time for exp in experiments if exp.execution_time > 0]
if not fitness_values:
fitness_values = [float('inf')]
# Calculate statistics
successful_runs = sum(1 for exp in experiments if exp.success)
success_rate = successful_runs / len(experiments)
# Calculate convergence rate (improvement per iteration)
convergence_rates = []
for exp in experiments:
if exp.convergence_history and len(exp.convergence_history) > 1:
initial_fitness = exp.convergence_history[0]
final_fitness = exp.convergence_history[-1]
if initial_fitness != final_fitness:
rate = (initial_fitness - final_fitness) / len(exp.convergence_history)
convergence_rates.append(rate)
avg_convergence_rate = np.mean(convergence_rates) if convergence_rates else 0.0
# Calculate final diversity (spread of final solutions)
final_diversity = np.std(fitness_values) if len(fitness_values) > 1 else 0.0
return StatisticalResults(
algorithm=experiments[0].algorithm,
function_id=experiments[0].function_id,
function_name=experiments[0].function_name,
dimension=experiments[0].dimension,
total_runs=len(experiments),
successful_runs=successful_runs,
success_rate=success_rate,
best_fitness=np.min(fitness_values),
worst_fitness=np.max(fitness_values),
mean_fitness=np.mean(fitness_values),
median_fitness=np.median(fitness_values),
std_fitness=np.std(fitness_values),
mean_execution_time=np.mean(execution_times) if execution_times else 0.0,
std_execution_time=np.std(execution_times) if execution_times else 0.0,
convergence_rate=avg_convergence_rate,
final_diversity=final_diversity
)
def _save_function_results(self, experiments: List[ExperimentResults], algorithm_name: str, function_id: int):
"""Save results for single function in Zhang et al. 2024 format"""
# Create algorithm directory
algo_dir = self.output_dir / algorithm_name
algo_dir.mkdir(exist_ok=True)
# Prepare data for M_X_D30.txt format
results_data = {
'function_id': function_id,
'function_name': experiments[0].function_name,
'algorithm': algorithm_name,
'dimension': self.dimension,
'total_runs': len(experiments),
'successful_runs': sum(1 for exp in experiments if exp.success),
'runs_data': []
}
for exp in experiments:
run_data = {
'run_number': int(exp.run_number),
'best_fitness': float(exp.best_fitness),
'final_fitness': float(exp.final_fitness),
'execution_time': float(exp.execution_time),
'success': bool(exp.success),
'iterations_to_best': int(exp.iterations_to_best),
'convergence_history': [float(x) for x in (exp.convergence_history[:50] if len(exp.convergence_history) > 50 else exp.convergence_history)] # Limit size and ensure float
}
results_data['runs_data'].append(run_data)
# Calculate summary statistics
fitness_values = [exp.best_fitness for exp in experiments if exp.best_fitness != float('inf')]
if fitness_values:
results_data['statistics'] = {
'best_fitness': float(np.min(fitness_values)),
'worst_fitness': float(np.max(fitness_values)),
'mean_fitness': float(np.mean(fitness_values)),
'median_fitness': float(np.median(fitness_values)),
'std_fitness': float(np.std(fitness_values)),
'success_rate': float(sum(1 for exp in experiments if exp.success) / len(experiments))
}
# Save as M_X_D30.txt
filename = f"M_{function_id}_D{self.dimension}.txt"
filepath = algo_dir / filename
with open(filepath, 'w') as f:
json.dump(results_data, f, indent=2)
self.logger.debug(f"Saved {filepath}")
def _save_algorithm_summary(self, results: Dict[int, StatisticalResults], algorithm_name: str):
"""Save complete algorithm summary"""
algo_dir = self.output_dir / algorithm_name
# Create summary data
summary_data = {
'algorithm': algorithm_name,
'test_configuration': {
'dimension': self.dimension,
'num_runs': self.num_runs,
'max_iterations': self.max_iterations,
'population_size': self.population_size,
'test_date': datetime.now().isoformat()
},
'overall_statistics': {},
'function_results': {}
}
# Calculate overall statistics
all_success_rates = [stats.success_rate for stats in results.values()]
all_mean_fitness = [stats.mean_fitness for stats in results.values() if stats.mean_fitness != float('inf')]
all_execution_times = [stats.mean_execution_time for stats in results.values()]
summary_data['overall_statistics'] = {
'total_functions_tested': len(results),
'average_success_rate': float(np.mean(all_success_rates)),
'successful_functions': sum(1 for rate in all_success_rates if rate > 0),
'average_execution_time': float(np.mean(all_execution_times)) if all_execution_times else 0.0,
'algorithm_ranking_score': float(np.mean(all_success_rates)) # Simple ranking metric
}
# Add function-wise results
for func_id, stats in results.items():
summary_data['function_results'][f'F{func_id}'] = asdict(stats)
# Save summary
summary_file = algo_dir / f'{algorithm_name}_summary.json'
with open(summary_file, 'w') as f:
json.dump(summary_data, f, indent=2)
# Save CSV format for easy analysis
try:
self._save_csv_summary(results, algorithm_name)
except ImportError:
self.logger.warning("Pandas not available. Skipping CSV export.")
def _save_csv_summary(self, results: Dict[int, StatisticalResults], algorithm_name: str):
"""Save results in CSV format for analysis"""
try:
import pandas as pd
except ImportError:
# Fallback to manual CSV creation
self._save_csv_manual(results, algorithm_name)
return
# Create DataFrame
data = []
for func_id, stats in results.items():
data.append({
'Algorithm': algorithm_name,
'Function_ID': func_id,
'Function_Name': stats.function_name,
'Dimension': stats.dimension,
'Best_Fitness': stats.best_fitness,
'Mean_Fitness': stats.mean_fitness,
'Std_Fitness': stats.std_fitness,
'Success_Rate': stats.success_rate,
'Mean_Time': stats.mean_execution_time,
'Convergence_Rate': stats.convergence_rate
})
df = pd.DataFrame(data)
# Save CSV
csv_file = self.output_dir / algorithm_name / f'{algorithm_name}_results.csv'
df.to_csv(csv_file, index=False)
def _save_csv_manual(self, results: Dict[int, StatisticalResults], algorithm_name: str):
"""Manual CSV creation without pandas"""
csv_file = self.output_dir / algorithm_name / f'{algorithm_name}_results.csv'
with open(csv_file, 'w') as f:
# Write header
header = 'Algorithm,Function_ID,Function_Name,Dimension,Best_Fitness,Mean_Fitness,Std_Fitness,Success_Rate,Mean_Time,Convergence_Rate\n'
f.write(header)
# Write data
for func_id, stats in results.items():
row = f'{algorithm_name},{func_id},{stats.function_name},{stats.dimension},{stats.best_fitness},{stats.mean_fitness},{stats.std_fitness},{stats.success_rate},{stats.mean_execution_time},{stats.convergence_rate}\n'
f.write(row)
def compare_algorithms(self, algorithm_names: List[str]) -> Dict:
"""Compare multiple algorithms across all functions"""
if not all(name in self.statistical_results for name in algorithm_names):
missing = [name for name in algorithm_names if name not in self.statistical_results]
raise ValueError(f"Missing algorithm results: {missing}")
comparison_data = {
'algorithms': algorithm_names,
'function_comparison': {},
'overall_ranking': {},
'statistical_tests': {}
}
# Function-wise comparison
all_function_ids = set()
for algo_results in self.statistical_results.values():
all_function_ids.update(algo_results.keys())
for func_id in sorted(all_function_ids):
comparison_data['function_comparison'][f'F{func_id}'] = {}
for algo_name in algorithm_names:
if func_id in self.statistical_results[algo_name]:
stats = self.statistical_results[algo_name][func_id]
comparison_data['function_comparison'][f'F{func_id}'][algo_name] = {
'best_fitness': stats.best_fitness,
'mean_fitness': stats.mean_fitness,
'success_rate': stats.success_rate
}
# Overall ranking
for algo_name in algorithm_names:
algo_results = self.statistical_results[algo_name]
success_rates = [stats.success_rate for stats in algo_results.values()]
mean_fitness = [stats.mean_fitness for stats in algo_results.values() if stats.mean_fitness != float('inf')]
comparison_data['overall_ranking'][algo_name] = {
'average_success_rate': float(np.mean(success_rates)),
'functions_solved': sum(1 for rate in success_rates if rate > 0.9),
'average_fitness': float(np.mean(mean_fitness)) if mean_fitness else float('inf')
}
# Save comparison
comparison_file = self.output_dir / 'algorithm_comparison.json'
with open(comparison_file, 'w') as f:
json.dump(comparison_data, f, indent=2)
return comparison_data
def generate_plots(self, algorithm_names: List[str]):
"""Generate comparison plots"""
if not PLOTTING_AVAILABLE:
self.logger.warning("Plotting libraries not available. Skipping plot generation.")
return
# Set style
plt.style.use('default')
sns.set_palette("husl")
# Create plots directory
plots_dir = self.output_dir / 'plots'
plots_dir.mkdir(exist_ok=True)
# Success rate comparison
self._plot_success_rates(algorithm_names, plots_dir)
# Mean fitness comparison
self._plot_mean_fitness(algorithm_names, plots_dir)
# Convergence plots for selected functions
self._plot_convergence_curves(algorithm_names, plots_dir, [1, 9, 10, 23])
def _plot_success_rates(self, algorithm_names: List[str], plots_dir: Path):
"""Plot success rates comparison"""
plt.figure(figsize=(15, 8))
function_ids = sorted(set().union(*(self.statistical_results[algo].keys() for algo in algorithm_names)))
x = np.arange(len(function_ids))
width = 0.8 / len(algorithm_names)
for i, algo_name in enumerate(algorithm_names):
success_rates = []
for func_id in function_ids:
if func_id in self.statistical_results[algo_name]:
success_rates.append(self.statistical_results[algo_name][func_id].success_rate)
else:
success_rates.append(0.0)
plt.bar(x + i * width, success_rates, width, label=algo_name, alpha=0.8)
plt.xlabel('Function ID')
plt.ylabel('Success Rate')
plt.title('Success Rate Comparison Across Benchmark Functions')
plt.xticks(x + width * (len(algorithm_names) - 1) / 2, [f'F{fid}' for fid in function_ids])
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / 'success_rates_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
def _plot_mean_fitness(self, algorithm_names: List[str], plots_dir: Path):
"""Plot mean fitness comparison"""
plt.figure(figsize=(15, 8))
function_ids = sorted(set().union(*(self.statistical_results[algo].keys() for algo in algorithm_names)))
for algo_name in algorithm_names:
mean_fitness_values = []
for func_id in function_ids:
if func_id in self.statistical_results[algo_name]:
fitness = self.statistical_results[algo_name][func_id].mean_fitness
mean_fitness_values.append(np.log10(max(fitness, 1e-10)) if fitness != float('inf') else 10)
else:
mean_fitness_values.append(10)
plt.plot(function_ids, mean_fitness_values, marker='o', label=algo_name, linewidth=2, markersize=4)
plt.xlabel('Function ID')
plt.ylabel('Log10(Mean Fitness)')
plt.title('Mean Fitness Comparison (Log Scale)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(function_ids[::2]) # Show every other function ID
plt.tight_layout()
plt.savefig(plots_dir / 'mean_fitness_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
def _plot_convergence_curves(self, algorithm_names: List[str], plots_dir: Path, selected_functions: List[int]):
"""Plot convergence curves for selected functions"""
for func_id in selected_functions:
plt.figure(figsize=(10, 6))
for algo_name in algorithm_names:
if func_id in self.statistical_results[algo_name]:
# Get representative run (median performance)
algo_experiments = [exp for exp in self.experiment_results if exp.algorithm == algo_name and exp.function_id == func_id]
if algo_experiments:
# Find median run
fitness_values = [exp.best_fitness for exp in algo_experiments]
median_idx = np.argsort(fitness_values)[len(fitness_values)//2]
median_exp = algo_experiments[median_idx]
if median_exp.convergence_history:
plt.plot(median_exp.convergence_history, label=algo_name, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Fitness')
plt.title(f'Convergence Curves - F{func_id}')
plt.yscale('log')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(plots_dir / f'convergence_F{func_id}.png', dpi=300, bbox_inches='tight')
plt.close()
if __name__ == "__main__":
# Quick test of the framework
print("Algorithm Testing Framework")
print("=" * 50)
# Initialize framework with reduced settings for quick test
framework = AlgorithmTestingFramework(
output_dir="./test_benchmark_results",
dimension=10, # Reduced for quick test
num_runs=5, # Reduced for quick test
max_iterations=100, # Reduced for quick test
population_size=20, # Reduced for quick test
verbose=True
)
print("Testing framework initialized successfully!")
print(f"Will test on dimension {framework.dimension} with {framework.num_runs} runs per function")
# Test single experiment
print("\nTesting single experiment...")
experiment = framework.run_single_experiment(
algorithm_class=OriginalHBO,
algorithm_name="Original_HBO",
function_id=1, # Sphere function
run_number=1
)
print(f"Single experiment result:")
print(f" Function: {experiment.function_name}")
print(f" Best Fitness: {experiment.best_fitness:.6e}")
print(f" Success: {experiment.success}")
print(f" Execution Time: {experiment.execution_time:.3f}s")
print("\nFramework ready for full algorithm testing!")
print("Use framework.run_algorithm_test() to test complete algorithms")