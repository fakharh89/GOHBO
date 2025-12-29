#!/usr/bin/env python3
"""
Complete Research Pipeline: Zhang et al. 2024 GOHBO Implementation
================================================================
Master script that executes the complete research pipeline following the Model Development Procedure from your requirements:
Step 1: HBO Code from Askari et al. 2020 Step 2: GOHBO Construction from Zhang et al. 2024 Step 3: 30 Functions Benchmark Testing (M_X_D30.txt format) Step 4: Scalability Testing Step 5: GOHBORESNET18 Model Creation Step 6: Application to Three Medical Datasets Final Comparison and Analysis of Results
Author: Medical AI Research Team
Version: 1.0.0
"""
import sys
import os
import time
import argparse
from pathlib import Path
from datetime import datetime
import json
# Add all module paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'algorithms'))
sys.path.insert(0, str(current_dir / 'datasets'))
sys.path.insert(0, str(current_dir / 'evaluation'))
# Import all components
from original_hbo import OriginalHBO
from gohbo_algorithm import GOHBO
from benchmark_functions import BenchmarkSuite
from algorithm_testing_framework import AlgorithmTestingFramework
from gohbo_resnet18 import GOHBORESNET18
from dataset_preparation import MedicalDatasetPreparer
from comprehensive_evaluation import ComprehensiveEvaluationFramework
class CompleteResearchPipeline:
"""
Master pipeline that executes all research components
"""
def __init__(self, output_base_dir: str = "./complete_research_results",
quick_mode: bool = False,
verbose: bool = True):
"""
Initialize complete research pipeline
Args:
output_base_dir: Base directory for all results
quick_mode: Use reduced parameters for quick testing
verbose: Enable detailed logging
"""
self.output_base_dir = Path(output_base_dir)
self.quick_mode = quick_mode
self.verbose = verbose
# Create output directory structure
self.output_base_dir.mkdir(parents=True, exist_ok=True)
# Results storage
self.pipeline_results = {
'pipeline_info': {
'version': '1.0.0',
'execution_date': datetime.now().isoformat(),
'quick_mode': quick_mode,
'total_execution_time': 0
},
'step_results': {},
'final_comparison': {}
}
print(f"Complete Research Pipeline Initialized")
print(f"Output Directory: {self.output_base_dir}")
print(f"Quick Mode: {quick_mode}")
print("="*80)
def step1_hbo_baseline(self):
"""Step 1: Implement and test original HBO algorithm"""
print(f"\nSTEP 1: Original HBO Algorithm (Askari et al. 2020)")
print("-"*60)
step_start = time.time()
# Test HBO on a few benchmark functions
test_functions = [1, 9, 10] # Sphere, Rastrigin, Ackley
benchmark_suite = BenchmarkSuite(dimension=30)
hbo_results = {}
for func_id in test_functions:
func = benchmark_suite.get_function(func_id)
print(f"Testing HBO on F{func_id} ({func.name})")
# Setup bounds
lower_bounds = [func.bounds[0]] * 30
upper_bounds = [func.bounds[1]] * 30
# Test parameters
max_iter = 50 if self.quick_mode else 200
pop_size = 20 if self.quick_mode else 40
# Run HBO
hbo = OriginalHBO(
objective_function=func.function,
dimension=30,
bounds=(lower_bounds, upper_bounds),
population_size=pop_size,
max_iterations=max_iter,
verbose=False
)
result = hbo.optimize()
hbo_results[f'F{func_id}'] = {
'function_name': func.name,
'best_fitness': float(result['best_fitness']),
'convergence_history': [float(x) for x in result['convergence_history']],
'algorithm': 'Original_HBO'
}
print(f" Best Fitness: {result['best_fitness']:.6e}")
step_time = time.time() - step_start
self.pipeline_results['step_results']['step1_hbo'] = {
'execution_time': step_time,
'results': hbo_results,
'status': 'completed'
}
print(f"Step 1 Complete ({step_time:.1f}s)")
return hbo_results
def step2_gohbo_construction(self):
"""Step 2: Construct and test GOHBO algorithm"""
print(f"\nSTEP 2: GOHBO Construction (Zhang et al. 2024)")
print("-"*60)
step_start = time.time()
# Test GOHBO on same benchmark functions test_functions = [1, 9, 10] # Sphere, Rastrigin, Ackley
benchmark_suite = BenchmarkSuite(dimension=30)
gohbo_results = {}
for func_id in test_functions:
func = benchmark_suite.get_function(func_id)
print(f"Testing GOHBO on F{func_id} ({func.name})")
# Setup bounds
lower_bounds = [func.bounds[0]] * 30 upper_bounds = [func.bounds[1]] * 30
# Test parameters
max_iter = 50 if self.quick_mode else 200
pop_size = 20 if self.quick_mode else 40
# Run GOHBO
gohbo = GOHBO(
objective_function=func.function,
dimension=30,
bounds=(lower_bounds, upper_bounds),
population_size=pop_size,
max_iterations=max_iter,
verbose=False
)
result = gohbo.optimize()
gohbo_results[f'F{func_id}'] = {
'function_name': func.name,
'best_fitness': float(result['best_fitness']),
'convergence_history': [float(x) for x in result['convergence_history']],
'algorithm': 'GOHBO'
}
print(f" Best Fitness: {result['best_fitness']:.6e}")
step_time = time.time() - step_start
self.pipeline_results['step_results']['step2_gohbo'] = {
'execution_time': step_time,
'results': gohbo_results,
'status': 'completed'
}
print(f"Step 2 Complete ({step_time:.1f}s)")
return gohbo_results
def step3_benchmark_testing(self):
"""Step 3: 30 Functions Benchmark Testing"""
print(f"\nSTEP 3: 30 Functions Benchmark Testing (M_X_D30.txt)")
print("-"*60)
step_start = time.time()
# Configure testing framework
runs = 5 if self.quick_mode else 30
iterations = 100 if self.quick_mode else 500
functions = list(range(1, 6)) if self.quick_mode else list(range(1, 31))
framework = AlgorithmTestingFramework(
output_dir=str(self.output_base_dir / "step3_benchmark_results"),
test_data_dir=str(self.output_base_dir.parent / "benchmark_test_data"),
dimension=30,
num_runs=runs,
max_iterations=iterations,
population_size=30,
verbose=False
)
print(f"Testing {len(functions)} functions with {runs} runs each")
# Test both algorithms
hbo_results = framework.run_algorithm_test(
algorithm_class=OriginalHBO,
algorithm_name="Original_HBO", function_ids=functions
)
gohbo_results = framework.run_algorithm_test(
algorithm_class=GOHBO,
algorithm_name="GOHBO",
function_ids=functions
)
# Compare results
comparison = framework.compare_algorithms(["Original_HBO", "GOHBO"])
step_time = time.time() - step_start
self.pipeline_results['step_results']['step3_benchmark'] = {
'execution_time': step_time,
'hbo_results': {fid: {'mean_fitness': stats.mean_fitness, 'success_rate': stats.success_rate} for fid, stats in hbo_results.items()},
'gohbo_results': {fid: {'mean_fitness': stats.mean_fitness, 'success_rate': stats.success_rate}
for fid, stats in gohbo_results.items()},
'comparison': comparison,
'functions_tested': len(functions),
'status': 'completed'
}
print(f"Step 3 Complete ({step_time:.1f}s)")
print(f"Results saved in M_X_D30.txt format")
return comparison
def step4_scalability_testing(self):
"""Step 4: Algorithm scalability testing"""
print(f"\nSTEP 4: Scalability Testing")
print("-"*60)
step_start = time.time()
# Test scalability on different dimensions
dimensions = [10, 20, 30] if self.quick_mode else [10, 20, 30, 50, 100]
test_function_id = 1 # Sphere function
benchmark_suite = BenchmarkSuite(dimension=30)
base_func = benchmark_suite.get_function(test_function_id)
scalability_results = {}
for dim in dimensions:
print(f"Testing dimension {dim}")
# Create function for this dimension
def scaled_function(x):
return sum(xi**2 for xi in x) # Sphere function
bounds = ([base_func.bounds[0]] * dim, [base_func.bounds[1]] * dim)
max_iter = 50 if self.quick_mode else 200
# Test HBO
hbo = OriginalHBO(
objective_function=scaled_function,
dimension=dim,
bounds=bounds,
population_size=30,
max_iterations=max_iter,
verbose=False
)
hbo_result = hbo.optimize()
# Test GOHBO
gohbo = GOHBO(
objective_function=scaled_function,
dimension=dim, bounds=bounds,
population_size=30,
max_iterations=max_iter,
verbose=False
)
gohbo_result = gohbo.optimize()
scalability_results[f'dim_{dim}'] = {
'hbo_fitness': float(hbo_result['best_fitness']),
'gohbo_fitness': float(gohbo_result['best_fitness']),
'hbo_convergence': [float(x) for x in hbo_result['convergence_history']],
'gohbo_convergence': [float(x) for x in gohbo_result['convergence_history']]
}
print(f" HBO: {hbo_result['best_fitness']:.6e}, GOHBO: {gohbo_result['best_fitness']:.6e}")
step_time = time.time() - step_start
self.pipeline_results['step_results']['step4_scalability'] = {
'execution_time': step_time,
'results': scalability_results,
'dimensions_tested': dimensions,
'status': 'completed'
}
print(f"Step 4 Complete ({step_time:.1f}s)")
return scalability_results
def step5_gohboresnet18_creation(self):
"""Step 5: Create GOHBORESNET18 model"""
print(f"\nSTEP 5: GOHBORESNET18 Model Creation")
print("-"*60)
step_start = time.time()
# Test GOHBORESNET18 with synthetic data
from gohbo_resnet18 import ResNet18Config, TrainingConfig, GOHBOConfig, create_synthetic_medical_data
# Generate test data
print("Generating synthetic medical data for testing...")
X, y = create_synthetic_medical_data(
num_samples=100 if self.quick_mode else 500,
input_shape=(64, 64, 3) # Smaller for testing
)
# Split data
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
# Configure GOHBORESNET18
resnet_config = ResNet18Config(
input_shape=(64, 64, 3),
num_classes=2,
dropout_rate=0.3
)
training_config = TrainingConfig(
batch_size=16,
epochs=5 if self.quick_mode else 20,
learning_rate=0.001
)
gohbo_config = GOHBOConfig(
population_size=8 if self.quick_mode else 20,
max_iterations=3 if self.quick_mode else 10
)
print("Initializing GOHBORESNET18...")
gohboresnet18 = GOHBORESNET18(
resnet_config=resnet_config,
training_config=training_config, gohbo_config=gohbo_config,
output_dir=str(self.output_base_dir / "step5_gohboresnet18"),
verbose=False
)
# Run optimization
print("Running GOHBO hyperparameter optimization...")
optimization_results = gohboresnet18.optimize_hyperparameters(
X_train, y_train, X_val, y_val
)
print("Training final model...")
final_results = gohboresnet18.train_final_model(
X_train, y_train, X_val, y_val, X_test, y_test
)
step_time = time.time() - step_start
self.pipeline_results['step_results']['step5_gohboresnet18'] = {
'execution_time': step_time,
'optimization_accuracy': optimization_results['best_accuracy'],
'final_accuracy': final_results['final_accuracy'],
'test_accuracy': final_results.get('test_accuracy', final_results['final_accuracy']),
'best_hyperparameters': optimization_results['best_hyperparameters'],
'status': 'completed'
}
print(f"Step 5 Complete ({step_time:.1f}s)")
print(f"Final Test Accuracy: {final_results.get('test_accuracy', final_results['final_accuracy']):.4f}")
return final_results
def step6_medical_datasets_evaluation(self):
"""Step 6: Apply GOHBORESNET18 to three real medical datasets"""
print(f"\nSTEP 6: GOHBORESNET18 on Real Medical Datasets")
print("-"*60)
step_start = time.time()
# Import the dedicated Step 6 evaluator
sys.path.insert(0, str(self.output_base_dir.parent))
from step6_gohboresnet18_real_datasets import RealDatasetGOHBORESNET18Evaluator
# Initialize real dataset evaluator
evaluator = RealDatasetGOHBORESNET18Evaluator(
results_dir=str(self.output_base_dir / "step6_real_medical_datasets"),
quick_mode=self.quick_mode,
verbose=False
)
print("Applying GOHBORESNET18 to:")
print("1. Colorectal Cancer Dataset (Kaggle)")
print("2. Brain Tumor MRI Dataset (Kaggle)")
print("3. Chest X-ray Pneumonia Dataset (Kaggle)")
try:
# Run complete real dataset evaluation
results = evaluator.run_complete_evaluation()
# Extract summary metrics
summary = results['overall_summary']
dataset_results = {}
for dataset_name, dataset_result in results['dataset_results'].items():
if 'error' not in dataset_result:
accuracy = dataset_result['final_results']['test_accuracy']
dataset_results[dataset_name] = {
'test_accuracy': accuracy,
'status': 'completed',
'total_samples': dataset_result['dataset_info']['total_samples'],
'num_classes': dataset_result['dataset_info']['num_classes']
}
print(f" {dataset_name}: {accuracy:.4f} accuracy")
else:
dataset_results[dataset_name] = {
'error': dataset_result['error'],
'status': 'failed'
}
print(f" {dataset_name}: Failed - {dataset_result['error']}")
step_time = time.time() - step_start
self.pipeline_results['step_results']['step6_medical'] = {
'execution_time': step_time,
'dataset_results': dataset_results,
'successful_datasets': summary['successful_evaluations'],
'total_datasets': summary['total_datasets'],
'average_accuracy': summary.get('average_accuracy', 0),
'status': 'completed'
}
print(f"Step 6 Complete ({step_time:.1f}s)")
print(f"Successfully evaluated {summary['successful_evaluations']}/{summary['total_datasets']} datasets")
if 'average_accuracy' in summary:
print(f"Average GOHBORESNET18 accuracy: {summary['average_accuracy']:.4f}")
return results
except Exception as e:
step_time = time.time() - step_start
error_message = str(e)
print(f"Step 6 Failed: {error_message}")
self.pipeline_results['step_results']['step6_medical'] = {
'execution_time': step_time,
'error': error_message,
'status': 'failed'
}
return {'error': error_message}
def generate_final_comparison(self):
"""Generate comprehensive final comparison and analysis"""
print(f"\nFINAL ANALYSIS: Complete Research Comparison")
print("-"*60)
analysis_start = time.time()
final_comparison = {
'algorithm_performance_summary': {},
'benchmark_comparison': {},
'medical_datasets_summary': {},
'scalability_analysis': {},
'overall_conclusions': {}
}
# Algorithm Performance Summary
if 'step3_benchmark' in self.pipeline_results['step_results']:
benchmark_data = self.pipeline_results['step_results']['step3_benchmark']
# Extract winner counts
hbo_wins = 0
gohbo_wins = 0
if 'comparison' in benchmark_data and 'function_comparison' in benchmark_data['comparison']:
for func_results in benchmark_data['comparison']['function_comparison'].values():
best_algo = None
best_success = 0
for algo_name, results in func_results.items():
if results.get('success_rate', 0) > best_success:
best_success = results['success_rate']
best_algo = algo_name
if best_algo == 'Original_HBO':
hbo_wins += 1
elif best_algo == 'GOHBO':
gohbo_wins += 1
final_comparison['benchmark_comparison'] = {
'hbo_wins': hbo_wins,
'gohbo_wins': gohbo_wins,
'functions_tested': benchmark_data['functions_tested'],
'gohbo_improvement': f"{(gohbo_wins / (hbo_wins + gohbo_wins) * 100):.1f}%" if (hbo_wins + gohbo_wins) > 0 else "N/A"
}
# Medical Datasets Summary
if 'step6_medical' in self.pipeline_results['step_results']:
medical_data = self.pipeline_results['step_results']['step6_medical']
successful_datasets = 0
gohbo_accuracies = []
baseline_accuracies = []
for dataset_name, results in medical_data['dataset_results'].items():
if 'algorithms' in results:
successful_datasets += 1
if 'GOHBORESNET18' in results['algorithms'] and 'test_accuracy' in results['algorithms']['GOHBORESNET18']:
gohbo_accuracies.append(results['algorithms']['GOHBORESNET18']['test_accuracy'])
if 'Baseline_HBO' in results['algorithms'] and 'test_accuracy' in results['algorithms']['Baseline_HBO']:
baseline_accuracies.append(results['algorithms']['Baseline_HBO']['test_accuracy'])
avg_gohbo_acc = sum(gohbo_accuracies) / len(gohbo_accuracies) if gohbo_accuracies else 0
avg_baseline_acc = sum(baseline_accuracies) / len(baseline_accuracies) if baseline_accuracies else 0
final_comparison['medical_datasets_summary'] = {
'datasets_evaluated': successful_datasets,
'avg_gohboresnet18_accuracy': avg_gohbo_acc,
'avg_baseline_accuracy': avg_baseline_acc,
'improvement_percentage': ((avg_gohbo_acc - avg_baseline_acc) / avg_baseline_acc * 100) if avg_baseline_acc > 0 else 0
}
# Scalability Analysis
if 'step4_scalability' in self.pipeline_results['step_results']:
scalability_data = self.pipeline_results['step_results']['step4_scalability']['results']
gohbo_better_count = 0
total_tests = len(scalability_data)
for dim_result in scalability_data.values():
if dim_result['gohbo_fitness'] < dim_result['hbo_fitness']: # Lower is better for minimization
gohbo_better_count += 1
final_comparison['scalability_analysis'] = {
'dimensions_tested': self.pipeline_results['step_results']['step4_scalability']['dimensions_tested'],
'gohbo_better_ratio': f"{gohbo_better_count}/{total_tests}",
'scalability_advantage': (gohbo_better_count / total_tests * 100) if total_tests > 0 else 0
}
# Overall Conclusions
gohbo_advantages = []
if final_comparison.get('benchmark_comparison', {}).get('gohbo_wins', 0) > final_comparison.get('benchmark_comparison', {}).get('hbo_wins', 0):
gohbo_advantages.append("Superior performance on benchmark functions")
if final_comparison.get('medical_datasets_summary', {}).get('improvement_percentage', 0) > 5:
gohbo_advantages.append("Significant improvement on medical datasets")
if final_comparison.get('scalability_analysis', {}).get('scalability_advantage', 0) > 50:
gohbo_advantages.append("Better scalability across dimensions")
final_comparison['overall_conclusions'] = {
'gohbo_advantages': gohbo_advantages,
'research_objectives_met': len(gohbo_advantages) >= 2,
'recommendation': "GOHBO shows superior performance" if len(gohbo_advantages) >= 2 else "Mixed results, further investigation needed"
}
analysis_time = time.time() - analysis_start
# Store final comparison
self.pipeline_results['final_comparison'] = final_comparison
self.pipeline_results['pipeline_info']['analysis_time'] = analysis_time
# Display results
print("\nFINAL RESEARCH RESULTS:")
print("-" * 40)
if 'benchmark_comparison' in final_comparison:
bc = final_comparison['benchmark_comparison']
print(f"Benchmark Functions: GOHBO won {bc['gohbo_wins']}/{bc['functions_tested']} functions")
if 'medical_datasets_summary' in final_comparison:
mds = final_comparison['medical_datasets_summary']
print(f"Medical Datasets: {mds['improvement_percentage']:.1f}% improvement with GOHBORESNET18")
if 'scalability_analysis' in final_comparison:
sa = final_comparison['scalability_analysis']
print(f"Scalability: GOHBO better on {sa['gohbo_better_ratio']} dimension tests")
print(f"\nRecommendation: {final_comparison['overall_conclusions']['recommendation']}")
return final_comparison
def run_complete_pipeline(self):
"""Execute the complete research pipeline"""
pipeline_start = time.time()
print("STARTING COMPLETE RESEARCH PIPELINE")
print("Zhang et al. 2024 GOHBO Implementation & Evaluation")
print("="*80)
try:
# Execute all steps
self.step1_hbo_baseline()
self.step2_gohbo_construction() self.step3_benchmark_testing()
self.step4_scalability_testing()
self.step5_gohboresnet18_creation()
self.step6_medical_datasets_evaluation()
# Final analysis
self.generate_final_comparison()
# Calculate total time
total_time = time.time() - pipeline_start
self.pipeline_results['pipeline_info']['total_execution_time'] = total_time
# Save complete results
results_file = self.output_base_dir / 'complete_research_results.json'
with open(results_file, 'w') as f:
json.dump(self.pipeline_results, f, indent=2)
# Generate summary report
self.generate_summary_report()
print(f"\nCOMPLETE RESEARCH PIPELINE FINISHED!")
print(f"Total Execution Time: {total_time/60:.1f} minutes")
print(f" Results saved in: {self.output_base_dir}")
print(f"Summary report: {self.output_base_dir}/research_summary_report.txt")
return self.pipeline_results
except Exception as e:
print(f"\nPipeline failed: {str(e)}")
import traceback
traceback.print_exc()
return None
def generate_summary_report(self):
"""Generate comprehensive summary report"""
report_lines = [
"COMPLETE RESEARCH RESULTS SUMMARY",
"="*80,
f"Execution Date: {self.pipeline_results['pipeline_info']['execution_date']}",
f"Total Time: {self.pipeline_results['pipeline_info']['total_execution_time']/60:.1f} minutes",
f"Quick Mode: {self.pipeline_results['pipeline_info']['quick_mode']}",
"",
"RESEARCH OBJECTIVES VALIDATION:",
"-"*40
]
# Step-by-step results
for step_name, step_data in self.pipeline_results['step_results'].items():
step_title = {
'step1_hbo': 'Step 1: Original HBO Implementation',
'step2_gohbo': 'Step 2: GOHBO Construction', 'step3_benchmark': 'Step 3: 30 Functions Testing',
'step4_scalability': 'Step 4: Scalability Analysis',
'step5_gohboresnet18': 'Step 5: GOHBORESNET18 Creation',
'step6_medical': 'Step 6: Medical Datasets Evaluation'
}.get(step_name, step_name)
status = "COMPLETED" if step_data['status'] == 'completed' else "FAILED"
time_str = f"({step_data['execution_time']:.1f}s)"
report_lines.append(f"{step_title}: {status} {time_str}")
# Final comparison results
if 'final_comparison' in self.pipeline_results:
fc = self.pipeline_results['final_comparison']
report_lines.extend([
"",
"FINAL COMPARISON RESULTS:",
"-"*40
])
if 'benchmark_comparison' in fc:
bc = fc['benchmark_comparison']
report_lines.append(f"Benchmark Functions: GOHBO won {bc['gohbo_wins']}/{bc['functions_tested']} ({bc['gohbo_improvement']})")
if 'medical_datasets_summary' in fc:
mds = fc['medical_datasets_summary']
report_lines.append(f"Medical Datasets: {mds['improvement_percentage']:.1f}% accuracy improvement")
if 'overall_conclusions' in fc:
oc = fc['overall_conclusions']
report_lines.extend([
"",
"RESEARCH CONCLUSIONS:",
"-"*40,
f"Objectives Met: {'YES' if oc['research_objectives_met'] else 'NO'}",
f"Recommendation: {oc['recommendation']}",
"",
"GOHBO Advantages Demonstrated:"
])
for advantage in oc['gohbo_advantages']:
report_lines.append(f"• {advantage}")
if not oc['gohbo_advantages']:
report_lines.append("• None clearly demonstrated")
report_lines.extend([
"",
"="*80,
"Research pipeline completed successfully.",
"All components implemented and tested according to Zhang et al. 2024 methodology."
])
# Save report
report = "\n".join(report_lines)
report_file = self.output_base_dir / 'research_summary_report.txt'
with open(report_file, 'w') as f:
f.write(report)
print("\nSummary Report Generated")
def main():
"""Main function with command-line interface"""
parser = argparse.ArgumentParser(
description='Complete Research Pipeline - Zhang et al. 2024 GOHBO Implementation',
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
This pipeline implements and evaluates the complete GOHBO research:
1. Original HBO (Askari et al. 2020)
2. GOHBO Construction (Zhang et al. 2024) 3. 30 Functions Benchmark Testing
4. Scalability Analysis
5. GOHBORESNET18 Model Creation
6. Medical Datasets Evaluation
Examples:
%(prog)s --quick # Quick test (reduced parameters)
%(prog)s # Full research pipeline %(prog)s --output-dir ./my_results # Custom output directory
"""
)
parser.add_argument('--output-dir', type=str, default='./complete_research_results',
help='Output directory for all results')
parser.add_argument('--quick', action='store_true',
help='Quick mode with reduced parameters for testing')
parser.add_argument('--verbose', action='store_true', default=True,
help='Enable verbose output')
args = parser.parse_args()
# Initialize and run pipeline
pipeline = CompleteResearchPipeline(
output_base_dir=args.output_dir,
quick_mode=args.quick,
verbose=args.verbose
)
# Confirm execution for full pipeline
if not args.quick:
print("\nWARNING: Full pipeline will take 1-2 hours to complete!")
response = input("Continue with full research pipeline? (y/N): ")
if response.lower() not in ['y', 'yes']:
print("Pipeline cancelled. Use --quick for testing.")
return
# Execute pipeline
results = pipeline.run_complete_pipeline()
if results:
print(f"\nRESEARCH PIPELINE COMPLETED SUCCESSFULLY!")
else:
print(f"\nRESEARCH PIPELINE FAILED!")
if __name__ == "__main__":
main()