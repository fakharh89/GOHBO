#!/usr/bin/env python3
"""
Medical Image Classification Pipeline - Command Line Interface
=============================================================
This CLI provides access to individual pipeline steps and the complete
pipeline for medical image classification with Improved HBO optimization.
Usage:
python cli.py --help
python cli.py run-pipeline --config configs/default_config.yaml
python cli.py step1 --input-dir ./data/images --labels ./data/labels.csv
python cli.py step2 --data-dir ./preprocessed_data
python cli.py step3 --data-dir ./preprocessed_data --baseline-model ./model.pkl
Author: Medical AI Research Team
Version: 1.0.0
"""
import sys
import os
import argparse
import json
import yaml
from pathlib import Path
from typing import Dict, Any
import warnings
warnings.filterwarnings('ignore')
# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))
# Import pipeline components
try:
from pipeline_runner.medical_classification_pipeline import MedicalClassificationPipeline
PIPELINE_AVAILABLE = True
except ImportError:
PIPELINE_AVAILABLE = False
try:
from step1_preprocessing.preprocessing_step import MedicalImagePreprocessor
STEP1_AVAILABLE = True
except ImportError:
STEP1_AVAILABLE = False
try:
from step2_model_training.model_training_step import CNNModelTrainer
STEP2_AVAILABLE = True
except ImportError:
STEP2_AVAILABLE = False
try:
from step3_optimization.optimization_step import HBOOptimizer
STEP3_AVAILABLE = True
except ImportError:
STEP3_AVAILABLE = False
try:
from step4_validation.validation_step import CrossValidator
STEP4_AVAILABLE = True
except ImportError:
STEP4_AVAILABLE = False
try:
from step5_evaluation.evaluation_step import PerformanceEvaluator
STEP5_AVAILABLE = True
except ImportError:
STEP5_AVAILABLE = False
try:
from step6_reporting.reporting_step import ReportGenerator
STEP6_AVAILABLE = True
except ImportError:
STEP6_AVAILABLE = False
class MedicalCLI:
"""Command Line Interface for Medical Image Classification Pipeline"""
def __init__(self):
self.parser = self.create_parser()
def create_parser(self) -> argparse.ArgumentParser:
"""Create the main argument parser"""
parser = argparse.ArgumentParser(
description='Medical Image Classification Pipeline with Improved HBO',
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
Examples:
# Run complete pipeline
%(prog)s run-pipeline --config configs/default_config.yaml
%(prog)s run-pipeline --quick-test
# Run individual steps
%(prog)s step1 --input-dir ./data/images --labels ./data/labels.csv --output-dir ./step1_out
%(prog)s step2 --data-dir ./step1_out --output-dir ./step2_out
%(prog)s step3 --data-dir ./step1_out --baseline-model ./step2_out/best_model.pkl --output-dir ./step3_out
%(prog)s step4 --data-dir ./step1_out --optimized-params ./step3_out/optimization_results.json --output-dir ./step4_out
%(prog)s step5 --cv-results ./step4_out/cross_validation_results.json --output-dir ./step5_out
%(prog)s step6 --results-dir ./results --output-dir ./step6_out
# Get step information
%(prog)s info step1
%(prog)s info step2
"""
)
# Add subcommands
subparsers = parser.add_subparsers(dest='command', help='Available commands')
# Complete pipeline command
self.add_pipeline_command(subparsers)
# Individual step commands
self.add_step1_command(subparsers)
self.add_step2_command(subparsers)
self.add_step3_command(subparsers)
self.add_step4_command(subparsers)
self.add_step5_command(subparsers)
self.add_step6_command(subparsers)
# Utility commands
self.add_info_command(subparsers)
self.add_config_command(subparsers)
return parser
def add_pipeline_command(self, subparsers):
"""Add complete pipeline command"""
pipeline_parser = subparsers.add_parser(
'run-pipeline',
help='Run the complete medical image classification pipeline',
description='Execute all 6 steps of the medical image classification pipeline'
)
pipeline_parser.add_argument(
'--config', '-c',
type=str,
help='Path to YAML configuration file'
)
pipeline_parser.add_argument(
'--quick-test',
action='store_true',
help='Run quick test with generated sample data'
)
pipeline_parser.add_argument(
'--data-dir',
type=str,
help='Path to medical images directory'
)
pipeline_parser.add_argument(
'--labels-file',
type=str,
help='Path to labels CSV file'
)
pipeline_parser.add_argument(
'--output-dir',
type=str,
default='./results',
help='Output directory for results (default: ./results)'
)
pipeline_parser.add_argument(
'--experiment-name',
type=str,
help='Name for the experiment'
)
pipeline_parser.add_argument(
'--resume',
type=str,
help='Resume from checkpoint (pipeline state file)'
)
pipeline_parser.add_argument(
'--verbose', '-v',
action='store_true',
help='Enable verbose output'
)
def add_step1_command(self, subparsers):
"""Add Step 1 (Preprocessing) command"""
step1_parser = subparsers.add_parser(
'step1',
help='Run Step 1: Medical Image Preprocessing',
description='Preprocess medical images with quality assessment and augmentation'
)
step1_parser.add_argument(
'--input-dir', '-i',
type=str,
required=True,
help='Directory containing medical images'
)
step1_parser.add_argument(
'--labels', '-l',
type=str,
required=True,
help='CSV file with image labels and metadata'
)
step1_parser.add_argument(
'--output-dir', '-o',
type=str,
default='./step1_preprocessing',
help='Output directory for preprocessed data (default: ./step1_preprocessing)'
)
step1_parser.add_argument(
'--target-size',
type=int,
nargs=2,
default=[224, 224],
metavar=('HEIGHT', 'WIDTH'),
help='Target image size (default: 224 224)'
)
step1_parser.add_argument(
'--normalization',
type=str,
choices=['min_max', 'z_score', 'percentile'],
default='z_score',
help='Normalization method (default: z_score)'
)
step1_parser.add_argument(
'--augmentation',
action='store_true',
help='Enable data augmentation for training set'
)
step1_parser.add_argument(
'--no-quality-check',
action='store_true',
help='Skip image quality assessment'
)
step1_parser.add_argument(
'--validation-split',
type=float,
default=0.2,
help='Validation set fraction (default: 0.2)'
)
step1_parser.add_argument(
'--test-split',
type=float,
default=0.2,
help='Test set fraction (default: 0.2)'
)
def add_step2_command(self, subparsers):
"""Add Step 2 (Model Training) command"""
step2_parser = subparsers.add_parser(
'step2',
help='Run Step 2: CNN Model Training',
description='Train CNN model on preprocessed medical images'
)
step2_parser.add_argument(
'--data-dir', '-d',
type=str,
required=True,
help='Directory containing preprocessed data from Step 1'
)
step2_parser.add_argument(
'--output-dir', '-o',
type=str,
default='./step2_model_training',
help='Output directory for trained model (default: ./step2_model_training)'
)
step2_parser.add_argument(
'--architecture',
type=str,
choices=['custom_cnn', 'transfer_resnet50', 'transfer_vgg16', 'transfer_densenet121'],
default='custom_cnn',
help='Model architecture (default: custom_cnn)'
)
step2_parser.add_argument(
'--epochs',
type=int,
default=100,
help='Number of training epochs (default: 100)'
)
step2_parser.add_argument(
'--batch-size',
type=int,
default=32,
help='Training batch size (default: 32)'
)
step2_parser.add_argument(
'--learning-rate',
type=float,
default=0.001,
help='Learning rate (default: 0.001)'
)
step2_parser.add_argument(
'--conv-blocks',
type=int,
default=4,
help='Number of convolutional blocks (default: 4)'
)
step2_parser.add_argument(
'--base-filters',
type=int,
default=64,
help='Base number of filters (default: 64)'
)
step2_parser.add_argument(
'--dropout-rate',
type=float,
default=0.3,
help='Dropout rate (default: 0.3)'
)
def add_step3_command(self, subparsers):
"""Add Step 3 (HBO Optimization) command"""
step3_parser = subparsers.add_parser(
'step3',
help='Run Step 3: HBO Hyperparameter Optimization',
description='Optimize CNN hyperparameters using Improved HBO algorithm'
)
step3_parser.add_argument(
'--data-dir', '-d',
type=str,
required=True,
help='Directory containing preprocessed data from Step 1'
)
step3_parser.add_argument(
'--baseline-model',
type=str,
required=True,
help='Path to baseline model from Step 2'
)
step3_parser.add_argument(
'--output-dir', '-o',
type=str,
default='./step3_optimization',
help='Output directory for optimization results (default: ./step3_optimization)'
)
step3_parser.add_argument(
'--iterations',
type=int,
default=100,
help='Number of HBO iterations (default: 100)'
)
step3_parser.add_argument(
'--population-size',
type=int,
default=30,
help='HBO population size (default: 30)'
)
step3_parser.add_argument(
'--degree',
type=int,
default=3,
help='HBO degree parameter (default: 3)'
)
step3_parser.add_argument(
'--no-adaptive-params',
action='store_true',
help='Disable adaptive parameters'
)
step3_parser.add_argument(
'--parallel',
action='store_true',
help='Enable parallel optimization (experimental)'
)
def add_step4_command(self, subparsers):
"""Add Step 4 (Cross-Validation) command"""
step4_parser = subparsers.add_parser(
'step4',
help='Run Step 4: Cross-Validation',
description='Perform k-fold cross-validation with optimized hyperparameters'
)
step4_parser.add_argument(
'--data-dir', '-d',
type=str,
required=True,
help='Directory containing preprocessed data from Step 1'
)
step4_parser.add_argument(
'--optimized-params',
type=str,
required=True,
help='JSON file with optimized parameters from Step 3'
)
step4_parser.add_argument(
'--output-dir', '-o',
type=str,
default='./step4_validation',
help='Output directory for validation results (default: ./step4_validation)'
)
step4_parser.add_argument(
'--folds',
type=int,
default=5,
help='Number of cross-validation folds (default: 5)'
)
step4_parser.add_argument(
'--strategy',
type=str,
choices=['patient_level', 'stratified', 'standard'],
default='patient_level',
help='Cross-validation strategy (default: patient_level)'
)
step4_parser.add_argument(
'--metrics',
type=str,
nargs='+',
default=['accuracy', 'precision', 'recall', 'f1', 'auc'],
help='Metrics to calculate (default: accuracy precision recall f1 auc)'
)
def add_step5_command(self, subparsers):
"""Add Step 5 (Evaluation) command"""
step5_parser = subparsers.add_parser(
'step5',
help='Run Step 5: Performance Evaluation',
description='Comprehensive performance evaluation and clinical assessment'
)
step5_parser.add_argument(
'--cv-results',
type=str,
required=True,
help='JSON file with cross-validation results from Step 4'
)
step5_parser.add_argument(
'--output-dir', '-o',
type=str,
default='./step5_evaluation',
help='Output directory for evaluation results (default: ./step5_evaluation)'
)
step5_parser.add_argument(
'--plots-dir',
type=str,
help='Directory to save plots (default: output-dir/plots)'
)
step5_parser.add_argument(
'--no-clinical-metrics',
action='store_true',
help='Skip clinical metrics calculation'
)
step5_parser.add_argument(
'--no-statistical-tests',
action='store_true',
help='Skip statistical significance tests'
)
step5_parser.add_argument(
'--no-visualizations',
action='store_true',
help='Skip generating visualization plots'
)
def add_step6_command(self, subparsers):
"""Add Step 6 (Reporting) command"""
step6_parser = subparsers.add_parser(
'step6',
help='Run Step 6: Report Generation',
description='Generate comprehensive reports and documentation'
)
step6_parser.add_argument(
'--results-dir',
type=str,
required=True,
help='Directory containing results from all previous steps'
)
step6_parser.add_argument(
'--output-dir', '-o',
type=str,
default='./step6_reporting',
help='Output directory for reports (default: ./step6_reporting)'
)
step6_parser.add_argument(
'--plots-dir',
type=str,
help='Directory containing plots from Step 5'
)
step6_parser.add_argument(
'--no-executive-summary',
action='store_true',
help='Skip executive summary generation'
)
step6_parser.add_argument(
'--no-clinical-report',
action='store_true',
help='Skip clinical report generation'
)
step6_parser.add_argument(
'--no-technical-report',
action='store_true',
help='Skip technical report generation'
)
step6_parser.add_argument(
'--no-dashboard',
action='store_true',
help='Skip HTML dashboard generation'
)
def add_info_command(self, subparsers):
"""Add info command for getting step information"""
info_parser = subparsers.add_parser(
'info',
help='Get information about pipeline steps',
description='Display detailed information about pipeline steps and requirements'
)
info_parser.add_argument(
'step',
type=str,
choices=['step1', 'step2', 'step3', 'step4', 'step5', 'step6', 'pipeline', 'all'],
help='Step to get information about'
)
info_parser.add_argument(
'--detailed',
action='store_true',
help='Show detailed information including parameters'
)
def add_config_command(self, subparsers):
"""Add config command for configuration management"""
config_parser = subparsers.add_parser(
'config',
help='Configuration file management',
description='Generate and manage configuration files'
)
config_subparsers = config_parser.add_subparsers(dest='config_action')
# Generate config command
generate_parser = config_subparsers.add_parser(
'generate',
help='Generate a new configuration file'
)
generate_parser.add_argument(
'--output', '-o',
type=str,
default='./custom_config.yaml',
help='Output configuration file (default: ./custom_config.yaml)'
)
generate_parser.add_argument(
'--template',
type=str,
choices=['default', 'quick_test', 'minimal'],
default='default',
help='Configuration template (default: default)'
)
# Validate config command
validate_parser = config_subparsers.add_parser(
'validate',
help='Validate a configuration file'
)
validate_parser.add_argument(
'config_file',
type=str,
help='Configuration file to validate'
)
def run_complete_pipeline(self, args):
"""Run the complete pipeline"""
if not PIPELINE_AVAILABLE:
print("Complete pipeline not available. Check dependencies.")
return False
print("Running Complete Medical Image Classification Pipeline")
print("=" * 80)
try:
if args.quick_test:
# Import and run quick test
sys.path.append(str(Path(__file__).parent))
from run_example import run_quick_test
return run_quick_test()
# Load configuration
if not args.config:
print("Configuration file required (use --config or --quick-test)")
return False
if not Path(args.config).exists():
print(f"Configuration file not found: {args.config}")
return False
# Create and configure pipeline
pipeline = MedicalClassificationPipeline(config_path=args.config)
# Apply command line overrides
if args.data_dir:
pipeline.config['data']['input_directory'] = args.data_dir
if args.labels_file:
pipeline.config['data']['labels_file'] = args.labels_file
if args.output_dir:
pipeline.config['data']['output_directory'] = args.output_dir
if args.experiment_name:
pipeline.config['experiment']['name'] = args.experiment_name
if args.verbose:
pipeline.config['pipeline']['verbose'] = True
# Run pipeline
results = pipeline.run_complete_pipeline()
print("\nPIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
# Display summary
summary = results['pipeline_summary']
key_metrics = results['key_metrics']
print(f"Execution Summary:")
print(f" • Pipeline ID: {summary['pipeline_id']}")
print(f" • Total Time: {summary['total_execution_time']:.2f} seconds")
print(f" • Success Rate: {summary['success_rate']:.1f}%")
print(f"\nKey Results:")
print(f" • Final Accuracy: {key_metrics['final_accuracy']:.4f}")
print(f" • Clinical Grade: {key_metrics['clinical_grade']}")
print(f" • HBO Improvement: {key_metrics['hbo_improvement']:.4f}")
print(f"\n Results Location:")
print(f" • Main Directory: {results['output_paths']['results_directory']}")
return True
except Exception as e:
print(f"Pipeline failed: {str(e)}")
if args.verbose:
import traceback
traceback.print_exc()
return False
def run_step1(self, args):
"""Run Step 1: Preprocessing"""
if not STEP1_AVAILABLE:
print("Step 1 not available. Check dependencies.")
return False
print("Step 1: Medical Image Preprocessing")
print("-" * 50)
try:
# Create configuration
config = {
'target_size': args.target_size,
'normalization': args.normalization,
'quality_check': not args.no_quality_check,
'clahe_enabled': True,
'augmentation': args.augmentation
}
# Create preprocessor
preprocessor = MedicalImagePreprocessor(config, args.output_dir)
# Run preprocessing
results = preprocessor.preprocess_dataset(
input_dir=args.input_dir,
labels_file=args.labels,
validation_split=args.validation_split,
test_split=args.test_split
)
print(f"Step 1 completed successfully")
print(f"Processed {results['summary']['total_images']} images")
print(f" Output directory: {args.output_dir}")
return True
except Exception as e:
print(f"Step 1 failed: {str(e)}")
return False
def run_step2(self, args):
"""Run Step 2: Model Training"""
if not STEP2_AVAILABLE:
print("Step 2 not available. Check dependencies.")
return False
print(" Step 2: CNN Model Training")
print("-" * 50)
try:
# Create configurations
model_config = {
'architecture': args.architecture,
'input_shape': [args.target_size[0] if hasattr(args, 'target_size') else 224,
args.target_size[1] if hasattr(args, 'target_size') else 224, 3],
'num_classes': 2, # Default to binary classification
'conv_blocks': args.conv_blocks,
'base_filters': args.base_filters,
'dropout_rate': args.dropout_rate
}
training_config = {
'epochs': args.epochs,
'batch_size': args.batch_size,
'learning_rate': args.learning_rate,
'early_stopping': True,
'patience': 10
}
# Create data paths from preprocessed data
data_paths = {}
for split in ['train', 'validation', 'test']:
split_dir = Path(args.data_dir) / split
if split_dir.exists():
data_paths[split] = {
'images': str(split_dir / 'images.npy'),
'labels': str(split_dir / 'labels.npy'),
'metadata': str(split_dir / 'metadata.json'),
'dataframe': str(split_dir / 'dataframe.csv')
}
if not data_paths:
print(f"No preprocessed data found in {args.data_dir}")
return False
# Create trainer
trainer = CNNModelTrainer(model_config, training_config, args.output_dir)
# Train model
results = trainer.train_baseline_model(data_paths)
# Save model
model_path = Path(args.output_dir) / 'best_model.pkl'
trainer.save_model(str(model_path))
print(f"Step 2 completed successfully")
print(f"Best validation accuracy: {results['validation_metrics'].get('accuracy', 'N/A'):.4f}")
print(f" Model saved: {model_path}")
print(f" Output directory: {args.output_dir}")
return True
except Exception as e:
print(f"Step 2 failed: {str(e)}")
return False
def run_step3(self, args):
"""Run Step 3: HBO Optimization"""
if not STEP3_AVAILABLE:
print("Step 3 not available. Check dependencies.")
return False
print("Step 3: HBO Hyperparameter Optimization")
print("-" * 50)
try:
# Create configuration
config = {
'algorithm': 'improved_hbo',
'iterations': args.iterations,
'population_size': args.population_size,
'degree': args.degree,
'adaptive_params': not args.no_adaptive_params,
'parallel': args.parallel
}
# Create data paths from preprocessed data
data_paths = {}
for split in ['train', 'validation']:
split_dir = Path(args.data_dir) / split
if split_dir.exists():
data_paths[split] = {
'images': str(split_dir / 'images.npy'),
'labels': str(split_dir / 'labels.npy'),
'metadata': str(split_dir / 'metadata.json'),
'dataframe': str(split_dir / 'dataframe.csv')
}
if not data_paths:
print(f"No preprocessed data found in {args.data_dir}")
return False
# Create optimizer
optimizer = HBOOptimizer(config, args.output_dir)
# Run optimization
results = optimizer.optimize_hyperparameters(
data_paths=data_paths,
baseline_model=args.baseline_model,
model_config={'num_classes': 2}, # Default config
training_config={'epochs': 100} # Default config
)
# Save optimized model
optimized_model_path = Path(args.output_dir) / 'optimized_model.pkl'
optimizer.save_best_model(str(optimized_model_path))
print(f"Step 3 completed successfully")
print(f"Best fitness: {results['best_fitness']:.6f}")
print(f"Improvement: {results['improvement']:.4f}")
print(f" Optimized model saved: {optimized_model_path}")
print(f" Output directory: {args.output_dir}")
return True
except Exception as e:
print(f"Step 3 failed: {str(e)}")
return False
def run_step4(self, args):
"""Run Step 4: Cross-Validation"""
if not STEP4_AVAILABLE:
print("Step 4 not available. Check dependencies.")
return False
print("Step 4: Cross-Validation")
print("-" * 50)
try:
# Load optimized parameters
if not Path(args.optimized_params).exists():
print(f"Optimized parameters file not found: {args.optimized_params}")
return False
with open(args.optimized_params, 'r') as f:
opt_results = json.load(f)
optimized_params = opt_results.get('best_params', {})
# Create configuration
config = {
'method': 'k_fold',
'folds': args.folds,
'strategy': args.strategy,
'metrics': args.metrics,
'random_state': 42
}
# Create data paths from preprocessed data
data_paths = {}
for split in ['train', 'validation']:
split_dir = Path(args.data_dir) / split
if split_dir.exists():
data_paths[split] = {
'images': str(split_dir / 'images.npy'),
'labels': str(split_dir / 'labels.npy'),
'metadata': str(split_dir / 'metadata.json'),
'dataframe': str(split_dir / 'dataframe.csv')
}
# Create validator
validator = CrossValidator(config, args.output_dir)
# Run cross-validation
results = validator.run_cross_validation(
data_paths=data_paths,
optimized_params=optimized_params,
model_config={'num_classes': 2},
training_config={'epochs': 100}
)
print(f"Step 4 completed successfully")
print(f"CV Accuracy: {results['mean_metrics'].get('accuracy', 0):.4f} ± {results['std_metrics'].get('accuracy', 0):.4f}")
print(f" Output directory: {args.output_dir}")
return True
except Exception as e:
print(f"Step 4 failed: {str(e)}")
return False
def run_step5(self, args):
"""Run Step 5: Performance Evaluation"""
if not STEP5_AVAILABLE:
print("Step 5 not available. Check dependencies.")
return False
print("Step 5: Performance Evaluation")
print("-" * 50)
try:
# Load cross-validation results
if not Path(args.cv_results).exists():
print(f"Cross-validation results file not found: {args.cv_results}")
return False
with open(args.cv_results, 'r') as f:
cv_results = json.load(f)
# Create configuration
config = {
'clinical_metrics': not args.no_clinical_metrics,
'statistical_tests': not args.no_statistical_tests,
'confidence_intervals': True,
'visualizations': not args.no_visualizations
}
# Setup plots directory
plots_dir = Path(args.plots_dir) if args.plots_dir else Path(args.output_dir) / 'plots'
plots_dir.mkdir(exist_ok=True)
# Create evaluator
evaluator = PerformanceEvaluator(config, args.output_dir)
# Run evaluation
results = evaluator.evaluate_model_performance(
cv_results=cv_results,
generate_plots=config['visualizations'],
plots_dir=plots_dir
)
# Display clinical assessment
clinical_assessment = results.get('clinical_assessment', {})
print(f"Step 5 completed successfully")
print(f" Clinical Grade: {clinical_assessment.get('grade', 'Unknown')}")
print(f"Clinical Readiness: {clinical_assessment.get('clinical_readiness', 'unknown')}")
print(f" Output directory: {args.output_dir}")
return True
except Exception as e:
print(f"Step 5 failed: {str(e)}")
return False
def run_step6(self, args):
"""Run Step 6: Report Generation"""
if not STEP6_AVAILABLE:
print("Step 6 not available. Check dependencies.")
return False
print("Step 6: Report Generation")
print("-" * 50)
try:
# Load results from all steps
results_dir = Path(args.results_dir)
if not results_dir.exists():
print(f"Results directory not found: {args.results_dir}")
return False
# Try to load complete results
complete_results = {}
# Look for results from each step
for step_num in range(1, 6):
step_dir = results_dir / f"step{step_num}"
if step_dir.exists():
# Try to find results files
for results_file in step_dir.glob("*results.json"):
try:
with open(results_file, 'r') as f:
step_results = json.load(f)
complete_results[f'step{step_num}'] = step_results
break
except:
continue
if not complete_results:
print("No results files found. Creating minimal report...")
complete_results = {'note': 'Minimal report - no step results found'}
# Create configuration
config = {
'generate_plots': True,
'clinical_assessment': not args.no_clinical_report,
'detailed_report': not args.no_technical_report,
'summary_dashboard': not args.no_dashboard,
'executive_summary': not args.no_executive_summary
}
# Setup plots directory
plots_dir = Path(args.plots_dir) if args.plots_dir else None
# Create reporter
reporter = ReportGenerator(config, args.output_dir)
# Generate reports
report_results = reporter.generate_comprehensive_report(
results=complete_results,
plots_dir=plots_dir
)
print(f"Step 6 completed successfully")
print(f"Reports generated: {len(report_results['generated_reports'])}")
# List generated reports
for report_type, report_path in report_results['generated_reports'].items():
print(f" • {report_type.replace('_', ' ').title()}: {report_path}")
print(f" Output directory: {args.output_dir}")
return True
except Exception as e:
print(f"Step 6 failed: {str(e)}")
return False
def show_info(self, args):
"""Show information about pipeline steps"""
step_info = {
'step1': {
'name': 'Medical Image Preprocessing',
'description': 'Preprocess medical images with quality assessment, normalization, and augmentation',
'inputs': ['Medical images directory', 'Labels CSV file'],
'outputs': ['Preprocessed images (NPY)', 'Labels (NPY)', 'Metadata (JSON)', 'Quality metrics'],
'key_features': [
'DICOM and NIfTI support',
'Quality assessment and filtering',
'CLAHE enhancement',
'Medical-appropriate augmentation',
'Patient-level data splitting'
]
},
'step2': {
'name': 'CNN Model Training',
'description': 'Train CNN models optimized for medical image classification',
'inputs': ['Preprocessed data from Step 1'],
'outputs': ['Trained CNN model', 'Training metrics', 'Validation results'],
'key_features': [
'Custom CNN architectures',
'Transfer learning support',
'Class imbalance handling',
'Early stopping and checkpointing',
'Medical-specific configurations'
]
},
'step3': {
'name': 'HBO Hyperparameter Optimization',
'description': 'Optimize CNN hyperparameters using Improved HBO algorithm',
'inputs': ['Preprocessed data', 'Baseline model from Step 2'],
'outputs': ['Optimized hyperparameters', 'Convergence history', 'Best model configuration'],
'key_features': [
'Improved Heap-Based Optimization',
'Adaptive parameter tuning',
'Medical-specific bounds',
'Convergence tracking',
'Parallel optimization support'
]
},
'step4': {
'name': 'Cross-Validation',
'description': 'Robust validation with patient-level splitting and statistical analysis',
'inputs': ['Preprocessed data', 'Optimized parameters from Step 3'],
'outputs': ['CV scores', 'Statistical tests', 'Fold-wise results'],
'key_features': [
'Patient-level k-fold CV',
'Multiple validation strategies',
'Statistical significance testing',
'Medical-specific metrics',
'Comprehensive fold analysis'
]
},
'step5': {
'name': 'Performance Evaluation',
'description': 'Comprehensive medical performance evaluation and clinical assessment',
'inputs': ['Cross-validation results from Step 4'],
'outputs': ['Clinical metrics', 'Clinical grading', 'Performance visualizations'],
'key_features': [
'Medical-specific metrics',
'Clinical acceptability grading',
'Statistical significance analysis',
'ROC analysis and calibration',
'Performance visualization'
]
},
'step6': {
'name': 'Report Generation',
'description': 'Generate comprehensive reports and documentation for stakeholders',
'inputs': ['Results from all previous steps'],
'outputs': ['Executive summary', 'Clinical report', 'Technical report', 'HTML dashboard'],
'key_features': [
'Executive summaries',
'Clinical reports for medical professionals',
'Technical reports for researchers',
'Interactive HTML dashboards',
'Regulatory compliance documentation'
]
},
'pipeline': {
'name': 'Complete Pipeline',
'description': 'End-to-end medical image classification with HBO optimization',
'inputs': ['Medical images', 'Labels file', 'Configuration'],
'outputs': ['Complete analysis results', 'Trained models', 'Comprehensive reports'],
'key_features': [
'Six-step integrated pipeline',
'Automatic step orchestration',
'Checkpoint and resume support',
'Comprehensive logging',
'Clinical-grade validation'
]
}
}
if args.step == 'all':
print(" Medical Image Classification Pipeline - All Steps")
print("=" * 80)
for step_id, info in step_info.items():
print(f"\n{step_id.upper()}: {info['name']}")
print(f"Description: {info['description']}")
if args.detailed:
print(f"Inputs: {', '.join(info['inputs'])}")
print(f"Outputs: {', '.join(info['outputs'])}")
else:
info = step_info.get(args.step, {})
if not info:
print(f"Unknown step: {args.step}")
return False
print(f" {info['name']}")
print("=" * 60)
print(f"Description: {info['description']}")
print(f"\nInputs:")
for inp in info['inputs']:
print(f" • {inp}")
print(f"\nOutputs:")
for out in info['outputs']:
print(f" • {out}")
if args.detailed:
print(f"\nKey Features:")
for feature in info['key_features']:
print(f" • {feature}")
return True
def manage_config(self, args):
"""Manage configuration files"""
if args.config_action == 'generate':
print(f"Generating configuration file: {args.output}")
# Load template
template_path = Path(__file__).parent / 'configs' / f'{args.template}_config.yaml'
if template_path.exists():
import shutil
shutil.copy(str(template_path), args.output)
print(f"Configuration generated from {args.template} template")
else:
# Generate basic configuration
basic_config = {
'experiment': {
'name': 'custom_medical_classification',
'description': 'Custom medical image classification experiment'
},
'data': {
'input_directory': './data/medical_images',
'labels_file': './data/labels.csv',
'output_directory': './results'
},
'pipeline': {
'verbose': True
}
}
with open(args.output, 'w') as f:
yaml.dump(basic_config, f, default_flow_style=False)
print(f"Basic configuration generated")
print(f" File saved: {args.output}")
return True
elif args.config_action == 'validate':
print(f"Validating configuration file: {args.config_file}")
if not Path(args.config_file).exists():
print(f"Configuration file not found: {args.config_file}")
return False
try:
with open(args.config_file, 'r') as f:
config = yaml.safe_load(f)
# Basic validation
required_sections = ['experiment', 'data']
missing_sections = []
for section in required_sections:
if section not in config:
missing_sections.append(section)
if missing_sections:
print(f"Missing required sections: {', '.join(missing_sections)}")
return False
print("Configuration file is valid")
print(f"Experiment: {config.get('experiment', {}).get('name', 'Unknown')}")
print(f" Data directory: {config.get('data', {}).get('input_directory', 'Not specified')}")
return True
except Exception as e:
print(f"Configuration validation failed: {str(e)}")
return False
return False
def run(self):
"""Run the CLI"""
args = self.parser.parse_args()
if not args.command:
self.parser.print_help()
return False
# Route to appropriate function
if args.command == 'run-pipeline':
return self.run_complete_pipeline(args)
elif args.command == 'step1':
return self.run_step1(args)
elif args.command == 'step2':
return self.run_step2(args)
elif args.command == 'step3':
return self.run_step3(args)
elif args.command == 'step4':
return self.run_step4(args)
elif args.command == 'step5':
return self.run_step5(args)
elif args.command == 'step6':
return self.run_step6(args)
elif args.command == 'info':
return self.show_info(args)
elif args.command == 'config':
return self.manage_config(args)
else:
print(f"Unknown command: {args.command}")
return False
def main():
"""Main entry point"""
print(" Medical Image Classification Pipeline - CLI")
print("=" * 60)
cli = MedicalCLI()
success = cli.run()
if not success:
print("\nCommand failed. Use --help for usage information.")
sys.exit(1)
else:
print("\nCommand completed successfully!")
sys.exit(0)
if __name__ == "__main__":
main()