"""
Medical Image Classification Pipeline - Main Runner
====================================================
This is the main pipeline that orchestrates all steps of the medical image classification process using Improved HBO optimization.
Pipeline Steps:
1. Data Preprocessing
2. Model Training 3. HBO Optimization
4. Cross Validation
5. Performance Evaluation
6. Report Generation
Author: Medical AI Research Team
Version: 1.0.0
"""
import os
import sys
import json
import yaml
import time
import shutil
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings
warnings.filterwarnings('ignore')
# Add all step directories to Python path
current_dir = Path(__file__).parent.parent
for step_dir in ['step1-preprocessing', 'step2-model-training', 'step3-optimization', 'step4-validation', 'step5-evaluation', 'step6-reporting']:
sys.path.insert(0, str(current_dir / step_dir))
# Import pipeline steps
from preprocessing_step import MedicalImagePreprocessor
from model_training_step import CNNModelTrainer from optimization_step import HBOOptimizer
from validation_step import CrossValidator
from evaluation_step import PerformanceEvaluator
from reporting_step import ReportGenerator
class MedicalClassificationPipeline:
"""Main pipeline class that orchestrates all steps"""
def __init__(self, config_path: str = None):
"""
Initialize the medical classification pipeline
Args:
config_path: Path to YAML configuration file
"""
self.config_path = config_path
self.config = self._load_configuration()
self.pipeline_id = self._generate_pipeline_id()
# Setup directories
self.setup_directories()
# Setup logging
self.logger = self._setup_logging()
# Initialize pipeline components
self.preprocessor = None
self.trainer = None
self.optimizer = None
self.validator = None
self.evaluator = None
self.reporter = None
# Pipeline state tracking
self.pipeline_state = {
'current_step': 0,
'completed_steps': [],
'failed_steps': [],
'start_time': None,
'end_time': None,
'results': {}
}
self.logger.info(f"Medical Classification Pipeline initialized")
self.logger.info(f"Pipeline ID: {self.pipeline_id}")
self.logger.info(f" Results directory: {self.results_dir}")
def _load_configuration(self) -> Dict[str, Any]:
"""Load pipeline configuration"""
if self.config_path and Path(self.config_path).exists():
with open(self.config_path, 'r') as f:
if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
return yaml.safe_load(f)
else:
return json.load(f)
else:
# Return default configuration
return self._get_default_config()
def _get_default_config(self) -> Dict[str, Any]:
"""Get default pipeline configuration"""
return {
'experiment': {
'name': 'medical_classification_hbo',
'description': 'Medical image classification with Improved HBO optimization',
'version': '1.0.0'
},
'data': {
'input_directory': './data/medical_images',
'labels_file': './data/labels.csv',
'output_directory': './results',
'image_formats': ['.png', '.jpg', '.jpeg', '.dcm', '.nii'],
'validation_split': 0.2,
'test_split': 0.2,
'patient_level_split': True
},
'preprocessing': {
'target_size': [224, 224],
'normalization': 'z_score',
'augmentation': True,
'quality_check': True,
'clahe_enabled': True
},
'model': {
'architecture': 'custom_cnn',
'input_shape': [224, 224, 3],
'num_classes': 2,
'conv_blocks': 4,
'base_filters': 64,
'dropout_rate': 0.3
},
'training': {
'epochs': 100,
'batch_size': 32,
'learning_rate': 0.001,
'early_stopping': True,
'patience': 10,
'class_weights': 'balanced'
},
'optimization': {
'algorithm': 'improved_hbo',
'iterations': 100,
'population_size': 30,
'degree': 3,
'adaptive_params': True,
'parallel': False
},
'validation': {
'method': 'k_fold',
'folds': 5,
'strategy': 'patient_level',
'metrics': ['accuracy', 'precision', 'recall', 'f1', 'auc']
},
'evaluation': {
'clinical_metrics': True,
'statistical_tests': True,
'confidence_intervals': True,
'visualizations': True
},
'reporting': {
'generate_plots': True,
'clinical_assessment': True,
'detailed_report': True,
'summary_dashboard': True
},
'pipeline': {
'save_intermediate': True,
'resume_from_checkpoint': False,
'parallel_execution': False,
'verbose': True
}
}
def _generate_pipeline_id(self) -> str:
"""Generate unique pipeline ID"""
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
exp_name = self.config.get('experiment', {}).get('name', 'pipeline')
return f"{exp_name}_{timestamp}"
def setup_directories(self):
"""Setup pipeline directories"""
base_dir = Path(self.config.get('data', {}).get('output_directory', './results'))
self.results_dir = base_dir / self.pipeline_id
# Create directory structure
directories = [
'step1-preprocessing',
'step2-model-training', 'step3-optimization',
'step4-validation',
'step5-evaluation',
'step6-reporting',
'logs',
'models',
'plots',
'intermediate'
]
for directory in directories:
(self.results_dir / directory).mkdir(parents=True, exist_ok=True)
def _setup_logging(self) -> logging.Logger:
"""Setup pipeline logging"""
logger = logging.getLogger(f'pipeline_{self.pipeline_id}')
logger.setLevel(logging.DEBUG if self.config.get('pipeline', {}).get('verbose', True) else logging.INFO)
# Remove existing handlers
for handler in logger.handlers[:]:
logger.removeHandler(handler)
# Create formatters
detailed_formatter = logging.Formatter(
'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
simple_formatter = logging.Formatter(
'%(levelname)s: %(message)s'
)
# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(simple_formatter)
logger.addHandler(console_handler)
# File handler
log_file = self.results_dir / 'logs' / 'pipeline.log'
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(detailed_formatter)
logger.addHandler(file_handler)
return logger
def save_pipeline_state(self):
"""Save current pipeline state"""
state_file = self.results_dir / 'pipeline_state.json'
with open(state_file, 'w') as f:
json.dump(self.pipeline_state, f, indent=2, default=str)
def run_complete_pipeline(self) -> Dict[str, Any]:
"""
Run the complete medical classification pipeline
Returns:
Dictionary containing complete pipeline results
"""
self.pipeline_state['start_time'] = datetime.now()
self.logger.info(" Starting Medical Image Classification Pipeline")
self.logger.info("=" * 80)
try:
# Step 1: Preprocessing
step1_results = self.run_step1_preprocessing()
# Step 2: Model Training
step2_results = self.run_step2_model_training(step1_results)
# Step 3: HBO Optimization step3_results = self.run_step3_optimization(step1_results, step2_results)
# Step 4: Cross Validation
step4_results = self.run_step4_validation(step1_results, step3_results)
# Step 5: Performance Evaluation
step5_results = self.run_step5_evaluation(step4_results)
# Step 6: Report Generation
step6_results = self.run_step6_reporting(step1_results, step2_results, step3_results, step4_results, step5_results)
# Compile final results
final_results = self._compile_final_results(
step1_results, step2_results, step3_results, step4_results, step5_results, step6_results
)
self.pipeline_state['end_time'] = datetime.now()
self.pipeline_state['results'] = final_results
# Save final state
self.save_pipeline_state()
# Print summary
self._print_pipeline_summary(final_results)
return final_results
except Exception as e:
self.logger.error(f"Pipeline failed at step {self.pipeline_state['current_step']}: {str(e)}")
self.pipeline_state['failed_steps'].append(self.pipeline_state['current_step'])
self.save_pipeline_state()
raise
def run_step1_preprocessing(self) -> Dict[str, Any]:
"""Step 1: Medical Image Preprocessing"""
self.pipeline_state['current_step'] = 1
self.logger.info("STEP 1: Medical Image Preprocessing")
self.logger.info("-" * 50)
try:
# Initialize preprocessor
self.preprocessor = MedicalImagePreprocessor(
config=self.config['preprocessing'],
output_dir=self.results_dir / 'step1-preprocessing'
)
# Run preprocessing
step_start = time.time()
preprocessed_data = self.preprocessor.preprocess_dataset(
input_dir=self.config['data']['input_directory'],
labels_file=self.config['data']['labels_file'],
validation_split=self.config['data']['validation_split'],
test_split=self.config['data']['test_split']
)
step_time = time.time() - step_start
# Save intermediate results
if self.config['pipeline']['save_intermediate']:
self.preprocessor.save_preprocessed_data(
self.results_dir / 'intermediate' / 'preprocessed_data.pkl'
)
results = {
'status': 'completed',
'execution_time': step_time,
'data_summary': preprocessed_data['summary'],
'output_paths': preprocessed_data['output_paths'],
'quality_metrics': preprocessed_data.get('quality_metrics', {})
}
self.pipeline_state['completed_steps'].append(1)
self.logger.info(f"Step 1 completed in {step_time:.2f} seconds")
self.logger.info(f"Processed {preprocessed_data['summary']['total_images']} images")
return results
except Exception as e:
self.logger.error(f"Step 1 failed: {str(e)}")
raise
def run_step2_model_training(self, step1_results: Dict[str, Any]) -> Dict[str, Any]:
"""Step 2: CNN Model Training"""
self.pipeline_state['current_step'] = 2
self.logger.info(" STEP 2: CNN Model Training")
self.logger.info("-" * 50)
try:
# Initialize trainer
self.trainer = CNNModelTrainer(
model_config=self.config['model'],
training_config=self.config['training'],
output_dir=self.results_dir / 'step2-model-training'
)
# Run initial training
step_start = time.time()
training_results = self.trainer.train_baseline_model(
data_paths=step1_results['output_paths']
)
step_time = time.time() - step_start
# Save model
model_path = self.results_dir / 'models' / 'baseline_model.pkl'
self.trainer.save_model(model_path)
results = {
'status': 'completed',
'execution_time': step_time,
'training_metrics': training_results['training_history'],
'validation_metrics': training_results['validation_metrics'],
'model_path': str(model_path),
'model_architecture': training_results['model_summary']
}
self.pipeline_state['completed_steps'].append(2)
self.logger.info(f"Step 2 completed in {step_time:.2f} seconds")
self.logger.info(f"Best validation accuracy: {training_results['validation_metrics'].get('accuracy', 'N/A'):.4f}")
return results
except Exception as e:
self.logger.error(f"Step 2 failed: {str(e)}")
raise
def run_step3_optimization(self, step1_results: Dict[str, Any], step2_results: Dict[str, Any]) -> Dict[str, Any]:
"""Step 3: HBO Hyperparameter Optimization"""
self.pipeline_state['current_step'] = 3
self.logger.info("STEP 3: HBO Hyperparameter Optimization")
self.logger.info("-" * 50)
try:
# Initialize optimizer
self.optimizer = HBOOptimizer(
config=self.config['optimization'],
output_dir=self.results_dir / 'step3-optimization'
)
# Run HBO optimization
step_start = time.time()
optimization_results = self.optimizer.optimize_hyperparameters(
data_paths=step1_results['output_paths'],
baseline_model=step2_results['model_path'],
model_config=self.config['model'],
training_config=self.config['training']
)
step_time = time.time() - step_start
# Save optimized model
optimized_model_path = self.results_dir / 'models' / 'optimized_model.pkl'
self.optimizer.save_best_model(optimized_model_path)
results = {
'status': 'completed',
'execution_time': step_time,
'best_hyperparameters': optimization_results['best_params'],
'optimization_history': optimization_results['convergence_history'],
'best_fitness': optimization_results['best_fitness'],
'optimized_model_path': str(optimized_model_path),
'improvement': optimization_results.get('improvement', 0.0)
}
self.pipeline_state['completed_steps'].append(3)
self.logger.info(f"Step 3 completed in {step_time:.2f} seconds")
self.logger.info(f"HBO achieved {optimization_results.get('improvement', 0.0):.2%} improvement")
return results
except Exception as e:
self.logger.error(f"Step 3 failed: {str(e)}")
raise
def run_step4_validation(self, step1_results: Dict[str, Any],
step3_results: Dict[str, Any]) -> Dict[str, Any]:
"""Step 4: Cross-Validation"""
self.pipeline_state['current_step'] = 4
self.logger.info("STEP 4: Cross-Validation")
self.logger.info("-" * 50)
try:
# Initialize validator
self.validator = CrossValidator(
config=self.config['validation'],
output_dir=self.results_dir / 'step4-validation'
)
# Run cross-validation
step_start = time.time()
cv_results = self.validator.run_cross_validation(
data_paths=step1_results['output_paths'],
optimized_params=step3_results['best_hyperparameters'],
model_config=self.config['model'],
training_config=self.config['training']
)
step_time = time.time() - step_start
results = {
'status': 'completed',
'execution_time': step_time,
'cv_scores': cv_results['fold_scores'],
'mean_metrics': cv_results['mean_metrics'],
'std_metrics': cv_results['std_metrics'],
'confidence_intervals': cv_results['confidence_intervals'],
'statistical_significance': cv_results['statistical_tests'],
'fold_details': cv_results['fold_details']
}
self.pipeline_state['completed_steps'].append(4)
self.logger.info(f"Step 4 completed in {step_time:.2f} seconds")
self.logger.info(f"CV Accuracy: {cv_results['mean_metrics'].get('accuracy', 0):.4f} ± {cv_results['std_metrics'].get('accuracy', 0):.4f}")
return results
except Exception as e:
self.logger.error(f"Step 4 failed: {str(e)}")
raise
def run_step5_evaluation(self, step4_results: Dict[str, Any]) -> Dict[str, Any]:
"""Step 5: Performance Evaluation"""
self.pipeline_state['current_step'] = 5
self.logger.info("STEP 5: Performance Evaluation")
self.logger.info("-" * 50)
try:
# Initialize evaluator
self.evaluator = PerformanceEvaluator(
config=self.config['evaluation'],
output_dir=self.results_dir / 'step5-evaluation'
)
# Run comprehensive evaluation
step_start = time.time()
evaluation_results = self.evaluator.evaluate_model_performance(
cv_results=step4_results,
generate_plots=self.config['evaluation']['visualizations'],
plots_dir=self.results_dir / 'plots'
)
step_time = time.time() - step_start
results = {
'status': 'completed',
'execution_time': step_time,
'clinical_metrics': evaluation_results['clinical_metrics'],
'diagnostic_performance': evaluation_results['diagnostic_performance'],
'statistical_analysis': evaluation_results['statistical_analysis'],
'clinical_grade': evaluation_results['clinical_assessment']['grade'],
'recommendations': evaluation_results['clinical_assessment']['recommendations']
}
self.pipeline_state['completed_steps'].append(5)
self.logger.info(f"Step 5 completed in {step_time:.2f} seconds")
self.logger.info(f" Clinical Grade: {evaluation_results['clinical_assessment']['grade']}")
return results
except Exception as e:
self.logger.error(f"Step 5 failed: {str(e)}")
raise
def run_step6_reporting(self, step1_results: Dict[str, Any], step2_results: Dict[str, Any],
step3_results: Dict[str, Any], step4_results: Dict[str, Any],
step5_results: Dict[str, Any]) -> Dict[str, Any]:
"""Step 6: Report Generation"""
self.pipeline_state['current_step'] = 6
self.logger.info("STEP 6: Report Generation")
self.logger.info("-" * 50)
try:
# Initialize reporter
self.reporter = ReportGenerator(
config=self.config['reporting'],
output_dir=self.results_dir / 'step6-reporting'
)
# Compile all results
all_results = {
'preprocessing': step1_results,
'training': step2_results,
'optimization': step3_results,
'validation': step4_results,
'evaluation': step5_results,
'config': self.config,
'pipeline_info': {
'id': self.pipeline_id,
'start_time': self.pipeline_state['start_time'],
'completed_steps': self.pipeline_state['completed_steps']
}
}
# Generate reports
step_start = time.time()
report_results = self.reporter.generate_comprehensive_report(
results=all_results,
plots_dir=self.results_dir / 'plots'
)
step_time = time.time() - step_start
results = {
'status': 'completed',
'execution_time': step_time,
'report_paths': report_results['generated_reports'],
'summary_dashboard': report_results['dashboard_path'],
'clinical_summary': report_results['clinical_summary']
}
self.pipeline_state['completed_steps'].append(6)
self.logger.info(f"Step 6 completed in {step_time:.2f} seconds")
self.logger.info(f"Reports generated: {len(report_results['generated_reports'])}")
return results
except Exception as e:
self.logger.error(f"Step 6 failed: {str(e)}")
raise
def _compile_final_results(self, *step_results) -> Dict[str, Any]:
"""Compile final pipeline results"""
total_time = (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
final_results = {
'pipeline_summary': {
'pipeline_id': self.pipeline_id,
'experiment_name': self.config['experiment']['name'],
'total_execution_time': total_time,
'completed_steps': len(self.pipeline_state['completed_steps']),
'failed_steps': len(self.pipeline_state['failed_steps']),
'success_rate': len(self.pipeline_state['completed_steps']) / 6 * 100
},
'step_results': {
f'step{i+1}': result for i, result in enumerate(step_results)
},
'key_metrics': {
'final_accuracy': step_results[4].get('clinical_metrics', {}).get('accuracy', 0),
'clinical_grade': step_results[4].get('clinical_grade', 'Unknown'),
'hbo_improvement': step_results[2].get('improvement', 0),
'cv_stability': step_results[3].get('std_metrics', {}).get('accuracy', 0)
},
'output_paths': {
'results_directory': str(self.results_dir),
'models_directory': str(self.results_dir / 'models'),
'reports_directory': str(self.results_dir / 'step6-reporting'),
'plots_directory': str(self.results_dir / 'plots')
}
}
return final_results
def _print_pipeline_summary(self, final_results: Dict[str, Any]):
"""Print pipeline execution summary"""
self.logger.info("=" * 80)
self.logger.info("MEDICAL CLASSIFICATION PIPELINE COMPLETED")
self.logger.info("=" * 80)
summary = final_results['pipeline_summary']
metrics = final_results['key_metrics']
self.logger.info(f"Pipeline ID: {summary['pipeline_id']}")
self.logger.info(f"Total Time: {summary['total_execution_time']:.2f} seconds")
self.logger.info(f"Completed Steps: {summary['completed_steps']}/6")
self.logger.info(f"Success Rate: {summary['success_rate']:.1f}%")
self.logger.info("")
self.logger.info("KEY RESULTS:")
self.logger.info(f" • Final Accuracy: {metrics['final_accuracy']:.4f}")
self.logger.info(f" • Clinical Grade: {metrics['clinical_grade']}")
self.logger.info(f" • HBO Improvement: {metrics['hbo_improvement']:.2%}")
self.logger.info(f" • CV Stability: ±{metrics['cv_stability']:.4f}")
self.logger.info("")
self.logger.info(f" Results saved to: {final_results['output_paths']['results_directory']}")
self.logger.info("=" * 80)
def main():
"""Main function to run the pipeline"""
import argparse
parser = argparse.ArgumentParser(description='Medical Image Classification Pipeline')
parser.add_argument('--config', type=str, help='Path to configuration file')
parser.add_argument('--data-dir', type=str, help='Path to medical images directory')
parser.add_argument('--labels-file', type=str, help='Path to labels CSV file')
parser.add_argument('--output-dir', type=str, default='./results', help='Output directory')
parser.add_argument('--experiment-name', type=str, default='medical_hbo_pipeline', help='Experiment name')
args = parser.parse_args()
# Create pipeline
pipeline = MedicalClassificationPipeline(config_path=args.config)
# Override config with command line arguments if provided
if args.data_dir:
pipeline.config['data']['input_directory'] = args.data_dir
if args.labels_file:
pipeline.config['data']['labels_file'] = args.labels_file
if args.output_dir:
pipeline.config['data']['output_directory'] = args.output_dir
if args.experiment_name:
pipeline.config['experiment']['name'] = args.experiment_name
# Run pipeline
try:
results = pipeline.run_complete_pipeline()
print(f"\nPipeline completed successfully!")
print(f" Results: {results['output_paths']['results_directory']}")
return 0
except Exception as e:
print(f"\nPipeline failed: {str(e)}")
return 1
if __name__ == "__main__":
exit(main())