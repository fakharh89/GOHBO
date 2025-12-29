#!/usr/bin/env python3
"""
Example Script: Medical Image Classification Pipeline
===================================================
This script demonstrates how to use the complete medical image classification
pipeline with Improved HBO optimization.
Usage:
python run_example.py --help
python run_example.py --quick-test
python run_example.py --config configs/default_config.yaml
Author: Medical AI Research Team
Version: 1.0.0
"""
import sys
import argparse
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')
# Add pipeline modules to path
sys.path.insert(0, str(Path(__file__).parent))
from pipeline_runner.medical_classification_pipeline import MedicalClassificationPipeline
def create_sample_dataset(output_dir: Path, n_images: int = 20) -> dict:
"""
Create a sample medical image dataset for demonstration
Args:
output_dir: Directory to create sample data
n_images: Number of sample images to create
Returns:
Dictionary with dataset information
"""
print(" Creating sample medical dataset...")
# Create directories
images_dir = output_dir / "medical_images"
images_dir.mkdir(parents=True, exist_ok=True)
# Create sample images and metadata
image_data = []
for i in range(n_images):
# Determine class and patient
class_name = "normal" if i < n_images // 2 else "abnormal"
patient_id = f"patient_{(i // 2) + 1:03d}" # 2 images per patient
# Create filename
filename = f"{class_name}_{i:03d}.png"
# Generate synthetic medical-like image
# Simulate different intensity patterns for normal vs abnormal
if class_name == "normal":
# Normal: more uniform intensity
image = np.random.normal(128, 30, (128, 128, 3))
image += np.random.normal(0, 10, (128, 128, 3)) # Less noise
else:
# Abnormal: more varied intensity patterns
image = np.random.normal(100, 50, (128, 128, 3))
image += np.random.normal(0, 20, (128, 128, 3)) # More noise
# Add some "abnormal" bright spots
for _ in range(np.random.randint(1, 4)):
x, y = np.random.randint(20, 108, 2)
size = np.random.randint(5, 15)
image[x:x+size, y:y+size] += np.random.uniform(50, 100)
# Clip to valid range and convert to uint8
image = np.clip(image, 0, 255).astype(np.uint8)
# Save image
image_path = images_dir / filename
Image.fromarray(image).save(image_path)
# Create metadata
image_data.append({
'image_path': filename,
'label': class_name,
'patient_id': patient_id,
'age': np.random.randint(25, 85),
'gender': np.random.choice(['M', 'F']),
'acquisition_date': f"2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
'severity': np.random.choice(['mild', 'moderate', 'severe']) if class_name == 'abnormal' else None
})
# Create labels CSV
labels_df = pd.DataFrame(image_data)
labels_file = output_dir / "labels.csv"
labels_df.to_csv(labels_file, index=False)
dataset_info = {
'images_dir': str(images_dir),
'labels_file': str(labels_file),
'n_images': n_images,
'n_classes': 2,
'class_distribution': labels_df['label'].value_counts().to_dict()
}
print(f"Created {n_images} sample images")
print(f"Class distribution: {dataset_info['class_distribution']}")
print(f" Images directory: {images_dir}")
print(f"Labels file: {labels_file}")
return dataset_info
def run_quick_test():
"""Run a quick test of the pipeline with minimal data"""
print("Starting Quick Test of Medical Image Classification Pipeline")
print("=" * 80)
# Create temporary directory for test data
with tempfile.TemporaryDirectory() as temp_dir:
temp_path = Path(temp_dir)
# Create sample dataset
dataset_info = create_sample_dataset(temp_path, n_images=12)
# Create quick test configuration
quick_config = {
'experiment': {
'name': 'quick_test_pipeline',
'description': 'Quick test of medical classification pipeline'
},
'data': {
'input_directory': dataset_info['images_dir'],
'labels_file': dataset_info['labels_file'],
'output_directory': str(temp_path / 'results'),
'validation_split': 0.3,
'test_split': 0.2,
'patient_level_split': True
},
'preprocessing': {
'target_size': [64, 64], # Small for quick processing
'normalization': 'min_max',
'quality_check': False,
'augmentation': False
},
'model': {
'architecture': 'custom_cnn',
'input_shape': [64, 64, 3],
'num_classes': 2,
'conv_blocks': 2,
'base_filters': 16,
'dropout_rate': 0.3
},
'training': {
'epochs': 5, # Very few epochs for quick test
'batch_size': 4,
'learning_rate': 0.01,
'early_stopping': False
},
'optimization': {
'algorithm': 'improved_hbo',
'iterations': 10, # Quick optimization
'population_size': 6,
'adaptive_params': True
},
'validation': {
'method': 'k_fold',
'folds': 3,
'strategy': 'patient_level'
},
'evaluation': {
'clinical_metrics': True,
'statistical_tests': False, # Skip for speed
'visualizations': False
},
'reporting': {
'generate_plots': False,
'clinical_assessment': True,
'detailed_report': True,
'summary_dashboard': False
},
'pipeline': {
'save_intermediate': False,
'verbose': True
}
}
try:
# Create and run pipeline
print("\nInitializing pipeline...")
pipeline = MedicalClassificationPipeline()
pipeline.config = quick_config
pipeline.pipeline_id = "quick_test_demo"
pipeline.setup_directories()
pipeline.logger = pipeline._setup_logging()
print(" Running complete pipeline...")
results = pipeline.run_complete_pipeline()
# Display results
print("\n" + "=" * 80)
print("QUICK TEST COMPLETED SUCCESSFULLY!")
print("=" * 80)
print(f"\nPipeline Summary:")
print(f" • Experiment: {results['pipeline_summary']['pipeline_id']}")
print(f" • Total Time: {results['pipeline_summary']['total_execution_time']:.1f} seconds")
print(f" • Success Rate: {results['pipeline_summary']['success_rate']:.1f}%")
print(f" • Completed Steps: {results['pipeline_summary']['completed_steps']}/6")
print(f"\nKey Results:")
key_metrics = results['key_metrics']
print(f" • Final Accuracy: {key_metrics['final_accuracy']:.3f}")
print(f" • Clinical Grade: {key_metrics['clinical_grade']}")
print(f" • HBO Improvement: {key_metrics['hbo_improvement']:.3f}")
print(f" • CV Stability: ±{key_metrics['cv_stability']:.3f}")
print(f"\n Output Directory: {results['output_paths']['results_directory']}")
print("\nQuick test demonstrates that the pipeline is working correctly!")
print("For full experiments, use larger datasets and the default configuration.")
return True
except Exception as e:
print(f"\nQuick test failed: {str(e)}")
print("Check the error details above for troubleshooting.")
return False
def run_with_config(config_path: str, **overrides):
"""
Run pipeline with specified configuration file
Args:
config_path: Path to YAML configuration file
**overrides: Configuration overrides
"""
config_file = Path(config_path)
if not config_file.exists():
print(f"Configuration file not found: {config_path}")
return False
print(f"Running Medical Image Classification Pipeline")
print(f"Configuration: {config_path}")
print("=" * 80)
try:
# Create pipeline
pipeline = MedicalClassificationPipeline(config_path=config_path)
# Apply overrides
for key, value in overrides.items():
if '.' in key:
# Handle nested keys like 'data.input_directory'
parts = key.split('.')
current = pipeline.config
for part in parts[:-1]:
if part not in current:
current[part] = {}
current = current[part]
current[parts[-1]] = value
else:
pipeline.config[key] = value
# Run pipeline
results = pipeline.run_complete_pipeline()
print("\nPIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
# Display summary
summary = results['pipeline_summary']
key_metrics = results['key_metrics']
print(f"\nExecution Summary:")
print(f" • Pipeline ID: {summary['pipeline_id']}")
print(f" • Total Time: {summary['total_execution_time']:.2f} seconds")
print(f" • Success Rate: {summary['success_rate']:.1f}%")
print(f"\nPerformance Results:")
print(f" • Final Accuracy: {key_metrics['final_accuracy']:.4f}")
print(f" • Clinical Grade: {key_metrics['clinical_grade']}")
print(f" • HBO Improvement: {key_metrics['hbo_improvement']:.4f}")
print(f"\n Results Location:")
print(f" • Main Directory: {results['output_paths']['results_directory']}")
print(f" • Reports: {results['output_paths']['reports_directory']}")
print(f" • Models: {results['output_paths']['models_directory']}")
return True
except Exception as e:
print(f"\nPipeline failed: {str(e)}")
import traceback
traceback.print_exc()
return False
def main():
"""Main function with command-line interface"""
parser = argparse.ArgumentParser(
description='Medical Image Classification Pipeline with Improved HBO',
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
Examples:
%(prog)s --quick-test # Run quick test with sample data
%(prog)s --config configs/default_config.yaml # Run with default configuration
%(prog)s --config configs/quick_test_config.yaml # Run with quick test configuration
# Run with custom settings
%(prog)s --config configs/default_config.yaml \\
--data-dir /path/to/images \\
--labels-file /path/to/labels.csv \\
--experiment-name my_experiment
"""
)
# Configuration options
parser.add_argument('--config', type=str, help='Path to YAML configuration file')
parser.add_argument('--quick-test', action='store_true', help='Run quick test with generated sample data')
# Data options
parser.add_argument('--data-dir', type=str, help='Path to medical images directory')
parser.add_argument('--labels-file', type=str, help='Path to labels CSV file')
parser.add_argument('--output-dir', type=str, help='Output directory for results')
# Experiment options
parser.add_argument('--experiment-name', type=str, help='Name for the experiment')
parser.add_argument('--quick', action='store_true', help='Use quick settings (fewer epochs, iterations, etc.)')
args = parser.parse_args()
# Validate arguments
if not args.quick_test and not args.config:
parser.error("Must specify either --quick-test or --config")
if args.quick_test and args.config:
parser.error("Cannot specify both --quick-test and --config")
# Run based on arguments
if args.quick_test:
success = run_quick_test()
else:
# Build overrides dictionary
overrides = {}
if args.data_dir:
overrides['data.input_directory'] = args.data_dir
if args.labels_file:
overrides['data.labels_file'] = args.labels_file
if args.output_dir:
overrides['data.output_directory'] = args.output_dir
if args.experiment_name:
overrides['experiment.name'] = args.experiment_name
# Quick mode overrides
if args.quick:
overrides.update({
'training.epochs': 20,
'optimization.iterations': 50,
'validation.folds': 3,
'evaluation.visualizations': False,
'reporting.generate_plots': False
})
success = run_with_config(args.config, **overrides)
# Exit with appropriate code
sys.exit(0 if success else 1)
if __name__ == "__main__":
main()