"""
Medical Image Classification Framework with Improved HBO
=======================================================
A comprehensive, step-by-step pipeline for medical image classification
using an Improved Heap-Based Optimization (HBO) algorithm.
Author: Medical AI Research Team
Version: 1.0.0
Modules:
pipeline_runner: Main pipeline orchestrator
step1_preprocessing: Medical image preprocessing
step2_model_training: CNN model training
step3_optimization: HBO hyperparameter optimization
step4_validation: Cross-validation
step5_evaluation: Performance evaluation
step6_reporting: Report generation
"""
__version__ = "1.0.0"
__author__ = "Medical AI Research Team"
__description__ = "Medical Image Classification with Improved HBO"
# Import main pipeline class for convenience
try:
from pipeline_runner.medical_classification_pipeline import MedicalClassificationPipeline
__all__ = ['MedicalClassificationPipeline']
except ImportError:
# Handle case where dependencies are not installed
__all__ = []
# Version info
VERSION_INFO = {
'version': __version__,
'author': __author__, 'description': __description__,
'components': [
'preprocessing',
'model_training', 'hbo_optimization',
'cross_validation',
'performance_evaluation',
'report_generation'
]
}
def get_version_info():
"""Get version and component information"""
return VERSION_INFO
def print_banner():
"""Print framework banner"""
print("=" * 80)
print(" Medical Image Classification Framework with Improved HBO")
print(f"Version: {__version__}")
print(f" Author: {__author__}")
print("=" * 80)
print(" Complete pipeline for medical image classification research")
print("HBO optimization for hyperparameter tuning")
print("Clinical-grade evaluation and reporting")
print("=" * 80)