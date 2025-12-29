# Medical Image Classification Pipeline with Improved HBO

A comprehensive, step-by-step pipeline for medical image classification using an **Improved Heap-Based Optimization (HBO)** algorithm for hyperparameter optimization.

## 🏥 Pipeline Overview

This framework implements a complete medical image classification pipeline with six distinct steps:

```
📁 Data Input → 🔧 Preprocessing → 🧠 Model Training → 🎯 HBO Optimization → 🔄 Cross-Validation → 📈 Evaluation → 📋 Reporting
```

## 📁 Directory Structure

```
final-code/
├── pipeline-runner/           # Main pipeline orchestrator
│   └── medical_classification_pipeline.py
├── step1-preprocessing/       # Medical image preprocessing
│   └── preprocessing_step.py
├── step2-model-training/      # CNN model training
│   └── model_training_step.py
├── step3-optimization/        # HBO hyperparameter optimization
│   └── optimization_step.py
├── step4-validation/          # Cross-validation
│   └── validation_step.py
├── step5-evaluation/          # Performance evaluation
│   └── evaluation_step.py
├── step6-reporting/           # Report generation
│   └── reporting_step.py
├── configs/                   # Configuration files
│   ├── default_config.yaml
│   └── quick_test_config.yaml
├── data/                      # Input data directory
├── results/                   # Output results directory
└── README.md                  # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Navigate to the final-code directory
cd final-code

# Install required dependencies
pip install numpy pandas scikit-learn scipy matplotlib seaborn
pip install tensorflow  # or pytorch
pip install opencv-python pillow pyyaml
pip install pydicom nibabel  # for medical image formats (optional)
```

### 2. Prepare Your Data

Create your data directory structure:

```
data/
├── medical_images/
│   ├── normal_001.png
│   ├── normal_002.png
│   ├── abnormal_001.png
│   └── abnormal_002.png
└── labels.csv
```

Your `labels.csv` should contain:

```csv
image_path,label,patient_id,age,gender
normal_001.png,normal,patient_001,45,M
normal_002.png,normal,patient_002,67,F
abnormal_001.png,abnormal,patient_003,52,M
abnormal_002.png,abnormal,patient_004,38,F
```

### 3. Run the Complete Pipeline

#### Option A: Using Default Configuration

```bash
python pipeline-runner/medical_classification_pipeline.py \
    --config configs/default_config.yaml \
    --data-dir ./data/medical_images \
    --labels-file ./data/labels.csv \
    --output-dir ./results
```

#### Option B: Using Quick Test Configuration

```bash
python pipeline-runner/medical_classification_pipeline.py \
    --config configs/quick_test_config.yaml
```

#### Option C: Using Python API

```python
from pipeline_runner.medical_classification_pipeline import MedicalClassificationPipeline

# Create and configure pipeline
pipeline = MedicalClassificationPipeline(config_path='configs/default_config.yaml')

# Override specific settings if needed
pipeline.config['data']['input_directory'] = './data/medical_images'
pipeline.config['data']['labels_file'] = './data/labels.csv'
pipeline.config['experiment']['name'] = 'my_experiment'

# Run complete pipeline
results = pipeline.run_complete_pipeline()

# Access results
print(f"Clinical Grade: {results['step_results']['step5']['clinical_grade']}")
print(f"Final Accuracy: {results['key_metrics']['final_accuracy']:.4f}")
```

## 🔧 Pipeline Steps Explained

### Step 1: Medical Image Preprocessing (`step1-preprocessing/`)

**Purpose**: Prepare medical images for training

**Features**:
- Medical image format support (DICOM, NIfTI, PNG, JPEG)
- Quality assessment and filtering
- CLAHE enhancement for medical images
- Normalization techniques (z-score, min-max, percentile)
- Medical-appropriate augmentation
- Patient-level data splitting

**Key Functions**:
```python
from step1_preprocessing.preprocessing_step import MedicalImagePreprocessor

preprocessor = MedicalImagePreprocessor(config, output_dir)
results = preprocessor.preprocess_dataset(input_dir, labels_file)
```

### Step 2: CNN Model Training (`step2-model-training/`)

**Purpose**: Train baseline CNN models for medical image classification

**Features**:
- Custom CNN architectures optimized for medical images
- Transfer learning with pre-trained models (ResNet50, VGG16, DenseNet)
- Medical-specific training configurations
- Class imbalance handling
- Early stopping and model checkpointing

**Key Functions**:
```python
from step2_model_training.model_training_step import CNNModelTrainer

trainer = CNNModelTrainer(model_config, training_config, output_dir)
results = trainer.train_baseline_model(data_paths)
```

### Step 3: HBO Hyperparameter Optimization (`step3-optimization/`)

**Purpose**: Optimize CNN hyperparameters using Improved HBO algorithm

**Features**:
- Improved Heap-Based Optimization algorithm
- Adaptive parameter tuning
- Medical-specific hyperparameter bounds
- Convergence tracking and early stopping
- Parallel optimization support

**Key Functions**:
```python
from step3_optimization.optimization_step import HBOOptimizer

optimizer = HBOOptimizer(config, output_dir)
results = optimizer.optimize_hyperparameters(data_paths, baseline_model, 
                                            model_config, training_config)
```

### Step 4: Cross-Validation (`step4-validation/`)

**Purpose**: Robust validation with patient-level splitting

**Features**:
- Patient-level k-fold cross-validation
- Multiple validation strategies
- Medical-specific metrics calculation
- Statistical significance testing
- Comprehensive fold-wise analysis

**Key Functions**:
```python
from step4_validation.validation_step import CrossValidator

validator = CrossValidator(config, output_dir)
results = validator.run_cross_validation(data_paths, optimized_params,
                                        model_config, training_config)
```

### Step 5: Performance Evaluation (`step5-evaluation/`)

**Purpose**: Comprehensive medical performance evaluation

**Features**:
- Medical-specific performance metrics
- Clinical acceptability grading
- Statistical significance analysis
- ROC analysis and calibration metrics
- Performance visualization

**Key Functions**:
```python
from step5_evaluation.evaluation_step import PerformanceEvaluator

evaluator = PerformanceEvaluator(config, output_dir)
results = evaluator.evaluate_model_performance(cv_results)
```

### Step 6: Report Generation (`step6-reporting/`)

**Purpose**: Generate comprehensive reports and documentation

**Features**:
- Executive summaries for stakeholders
- Clinical reports for medical professionals
- Technical reports for researchers
- Interactive HTML dashboards
- Regulatory compliance documentation

**Key Functions**:
```python
from step6_reporting.reporting_step import ReportGenerator

reporter = ReportGenerator(config, output_dir)
results = reporter.generate_comprehensive_report(all_results, plots_dir)
```

## ⚙️ Configuration

### Main Configuration File (`configs/default_config.yaml`)

The pipeline uses YAML configuration files to control all aspects of the execution. Key sections:

```yaml
# Experiment metadata
experiment:
  name: "medical_classification_hbo"
  description: "Medical image classification with HBO optimization"

# Data settings
data:
  input_directory: "./data/medical_images"
  labels_file: "./data/labels.csv"
  patient_level_split: true

# Model architecture
model:
  architecture: "custom_cnn"
  input_shape: [224, 224, 3]
  num_classes: 2

# HBO optimization
optimization:
  algorithm: "improved_hbo"
  iterations: 100
  population_size: 30

# Validation settings
validation:
  method: "k_fold"
  folds: 5
  strategy: "patient_level"
```

### Custom Configuration

Create your own configuration file:

```yaml
# my_config.yaml
experiment:
  name: "my_medical_experiment"

data:
  input_directory: "/path/to/my/images"
  labels_file: "/path/to/my/labels.csv"
  num_classes: 3  # for multi-class classification

model:
  conv_blocks: 5
  base_filters: 128

optimization:
  iterations: 150
  population_size: 40
```

## 📊 Understanding Results

### Output Directory Structure

After running the pipeline, you'll get:

```
results/
├── medical_classification_hbo_20241128_120000/
│   ├── step1-preprocessing/
│   │   ├── train/
│   │   ├── validation/
│   │   ├── test/
│   │   └── preprocessing.log
│   ├── step2-model-training/
│   │   ├── best_model.h5
│   │   └── training_results.json
│   ├── step3-optimization/
│   │   ├── optimization_results.json
│   │   └── convergence_history.json
│   ├── step4-validation/
│   │   └── cross_validation_results.json
│   ├── step5-evaluation/
│   │   ├── performance_evaluation.json
│   │   └── evaluation_summary.txt
│   ├── step6-reporting/
│   │   ├── executive_summary.txt
│   │   ├── clinical_report.txt
│   │   ├── technical_report.txt
│   │   └── dashboard.html
│   ├── models/
│   ├── plots/
│   └── pipeline_state.json
```

### Key Result Files

1. **`step6-reporting/dashboard.html`**: Interactive HTML dashboard
2. **`step6-reporting/executive_summary.txt`**: High-level summary for stakeholders
3. **`step6-reporting/clinical_report.txt`**: Detailed clinical assessment
4. **`step5-evaluation/evaluation_summary.txt`**: Performance evaluation summary
5. **`models/optimized_model.pkl`**: Best optimized model

### Clinical Grading System

The pipeline provides clinical grading:

- **Excellent**: Ready for clinical validation studies
- **Good**: Promising for clinical validation with minor improvements
- **Acceptable**: Requires improvements before clinical validation
- **Limited**: Significant improvements needed for clinical use
- **Poor**: Not suitable for clinical use in current form

## 🎯 Advanced Usage

### Running Individual Steps

You can run individual pipeline steps:

```python
# Run only preprocessing
from step1_preprocessing.preprocessing_step import MedicalImagePreprocessor

config = {'target_size': [224, 224], 'normalization': 'z_score'}
preprocessor = MedicalImagePreprocessor(config, './preprocessing_output')
results = preprocessor.preprocess_dataset('./data/images', './data/labels.csv')
```

### Custom Hyperparameter Bounds

Modify HBO optimization bounds:

```python
# In step3_optimization/optimization_step.py
def set_custom_bounds(self):
    self.bounds_lower = np.array([2.0, 32.0, 1.5, 0.1, 1e-4, 1e-5])
    self.bounds_upper = np.array([8.0, 256.0, 4.0, 0.7, 1e-2, 1e-2])
```

### Adding Custom Metrics

Add your own evaluation metrics:

```python
# In step5_evaluation/evaluation_step.py
def calculate_custom_metrics(self, y_true, y_pred):
    # Your custom metric calculation
    custom_score = your_metric_function(y_true, y_pred)
    return {'custom_metric': custom_score}
```

## 🐛 Troubleshooting

### Common Issues

1. **Memory Issues**:
   ```yaml
   # In config file
   training:
     batch_size: 16  # Reduce batch size
   preprocessing:
     target_size: [128, 128]  # Reduce image size
   ```

2. **Slow Training**:
   ```yaml
   # Use quick test config
   training:
     epochs: 10
   optimization:
     iterations: 20
   ```

3. **Import Errors**:
   ```bash
   # Make sure you're in the final-code directory
   cd final-code
   # Add to Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)"
   ```

### Debug Mode

Enable detailed logging:

```python
pipeline = MedicalClassificationPipeline(config_path='configs/default_config.yaml')
pipeline.config['pipeline']['verbose'] = True
```

## 📚 Example Datasets

### Binary Classification (Normal vs Abnormal)

```csv
image_path,label,patient_id,age,gender,severity
chest_001.png,normal,P001,45,M,
chest_002.png,normal,P002,67,F,
chest_003.png,abnormal,P003,52,M,moderate
chest_004.png,abnormal,P004,38,F,severe
```

### Multi-class Classification

```csv
image_path,label,patient_id,diagnosis
brain_001.png,0,P001,normal
brain_002.png,1,P002,mild_cognitive_impairment
brain_003.png,2,P003,alzheimers
brain_004.png,1,P004,mild_cognitive_impairment
```

## 📝 Citation

If you use this framework in your research, please cite:

```bibtex
@article{medical_hbo_pipeline,
  title={Medical Image Classification Pipeline with Improved Heap-Based Optimization},
  author={Medical AI Research Team},
  journal={Medical AI Framework},
  year={2024},
  note={Step-by-step pipeline for medical image classification}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Add your improvements to the appropriate step directory
4. Test with provided configurations
5. Submit pull request

## 📄 License

This project is licensed under the MIT License. See LICENSE file for details.

## 🙏 Acknowledgments

- Medical domain experts for clinical validation guidance
- Open-source medical imaging community
- Contributors to optimization algorithms research

---

**⚠️ Important**: This framework is for research purposes. Clinical deployment requires proper validation, regulatory approval, and medical expert oversight.

**🔬 Research Ready**: Complete implementation for academic research and development in medical AI.

*Last updated: November 2024*