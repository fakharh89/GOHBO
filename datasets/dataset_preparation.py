#!/usr/bin/env python3
"""
Medical Dataset Preparation for GOHBORESNET18
==============================================
Automated download and preparation of the three medical datasets:
1. Colorectal Cancer Dataset
2. Brain Tumor MRI Dataset 3. Chest X-ray Pneumonia Dataset
Handles Kaggle API integration, data preprocessing, and dataset formatting
for the GOHBORESNET18 model evaluation.
Author: Medical AI Research Team
Version: 1.0.0
"""
import os
import sys
import shutil
import zipfile
import requests
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import time
import json
import logging
from datetime import datetime
try:
from PIL import Image
PIL_AVAILABLE = True
except ImportError:
PIL_AVAILABLE = False
print("PIL not available. Image processing will be limited.")
try:
import kaggle
KAGGLE_AVAILABLE = True
except ImportError:
KAGGLE_AVAILABLE = False
print("Kaggle API not available. Manual download instructions will be provided.")
class MedicalDatasetPreparer:
"""
Automated preparation of medical image datasets for research
Handles download, preprocessing, and formatting of medical image datasets
for machine learning research applications.
"""
def __init__(self, base_data_dir: str = "./medical_datasets",
target_image_size: Tuple[int, int] = (224, 224),
verbose: bool = True):
"""
Initialize dataset preparer
Args:
base_data_dir: Base directory for storing datasets
target_image_size: Target size for image preprocessing
verbose: Enable detailed logging
"""
self.base_data_dir = Path(base_data_dir)
self.target_image_size = target_image_size
self.verbose = verbose
# Create base directory
self.base_data_dir.mkdir(parents=True, exist_ok=True)
# Setup logging
if verbose:
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger(__name__)
else:
self.logger = logging.getLogger(__name__)
self.logger.addHandler(logging.NullHandler())
# Dataset configurations
self.datasets_config = {
'colorectal_cancer': {
'kaggle_dataset': 'ankushpanday2/colorectal-cancer-global-dataset-and-predictions',
'url': 'https://www.kaggle.com/datasets/ankushpanday2/colorectal-cancer-global-dataset-and-predictions',
'description': 'Colorectal Cancer Global Dataset',
'expected_classes': ['benign', 'malignant'],
'data_format': 'organized_folders'
},
'brain_tumor': {
'kaggle_dataset': 'masoudnickparvar/brain-tumor-mri-dataset',
'url': 'https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset',
'description': 'Brain Tumor MRI Dataset',
'expected_classes': ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor'],
'data_format': 'organized_folders'
},
'chest_xray': {
'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
'url': 'https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia',
'description': 'Chest X-ray Pneumonia Dataset',
'expected_classes': ['NORMAL', 'PNEUMONIA'],
'data_format': 'train_val_test_folders'
}
}
self.logger.info(f"Medical Dataset Preparer initialized")
self.logger.info(f"Base directory: {self.base_data_dir}")
self.logger.info(f"Target image size: {self.target_image_size}")
def check_kaggle_setup(self) -> bool:
"""Check if Kaggle API is properly configured"""
if not KAGGLE_AVAILABLE:
self.logger.warning("Kaggle API not available")
return False
try:
# Test Kaggle API authentication
kaggle.api.authenticate()
self.logger.info("Kaggle API authenticated successfully")
return True
except Exception as e:
self.logger.warning(f"Kaggle API authentication failed: {str(e)}")
return False
def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
"""
Download dataset from Kaggle
Args:
dataset_name: Name of dataset ('colorectal_cancer', 'brain_tumor', 'chest_xray')
force_redownload: Force redownload even if dataset exists
Returns:
True if successful, False otherwise
"""
if dataset_name not in self.datasets_config:
self.logger.error(f"Unknown dataset: {dataset_name}")
return False
config = self.datasets_config[dataset_name]
dataset_dir = self.base_data_dir / dataset_name
# Check if dataset already exists
if dataset_dir.exists() and not force_redownload:
self.logger.info(f"Dataset {dataset_name} already exists. Use force_redownload=True to redownload.")
return True
# Check Kaggle API availability
if not self.check_kaggle_setup():
self.logger.error("Kaggle API not available. Please install kaggle package and setup API credentials.")
self.logger.info("Instructions:")
self.logger.info("1. pip install kaggle")
self.logger.info("2. Get API key from https://www.kaggle.com/account")
self.logger.info("3. Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY")
self.logger.info(f"4. Manual download: {config['url']}")
return False
try:
self.logger.info(f"Downloading {config['description']}...")
# Create temporary download directory
temp_dir = self.base_data_dir / f"temp_{dataset_name}"
temp_dir.mkdir(exist_ok=True)
# Download dataset
kaggle.api.dataset_download_files(
config['kaggle_dataset'],
path=str(temp_dir),
unzip=True
)
# Move to final location
if dataset_dir.exists():
shutil.rmtree(dataset_dir)
# Find the downloaded content (may be in subdirectories)
downloaded_content = list(temp_dir.glob('*'))
if len(downloaded_content) == 1 and downloaded_content[0].is_dir():
# Single directory - move its contents
shutil.move(str(downloaded_content[0]), str(dataset_dir))
else:
# Multiple files/directories - move the temp dir
shutil.move(str(temp_dir), str(dataset_dir))
temp_dir = None
# Clean up temp directory if it still exists
if temp_dir and temp_dir.exists():
shutil.rmtree(temp_dir)
self.logger.info(f"Successfully downloaded {dataset_name} to {dataset_dir}")
return True
except Exception as e:
self.logger.error(f"Failed to download {dataset_name}: {str(e)}")
return False
def analyze_dataset_structure(self, dataset_name: str) -> Dict:
"""
Analyze the structure of a downloaded dataset
Args:
dataset_name: Name of dataset to analyze
Returns:
Dictionary with dataset structure information
"""
dataset_dir = self.base_data_dir / dataset_name
if not dataset_dir.exists():
self.logger.error(f"Dataset {dataset_name} not found at {dataset_dir}")
return {}
analysis = {
'dataset_name': dataset_name,
'dataset_path': str(dataset_dir),
'total_size_mb': sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()) / (1024*1024),
'directory_structure': {},
'file_counts': {},
'image_extensions': set(),
'total_files': 0,
'total_images': 0
}
# Walk through directory structure
for root, dirs, files in os.walk(dataset_dir):
rel_path = os.path.relpath(root, dataset_dir)
if rel_path == '.':
rel_path = 'root'
analysis['directory_structure'][rel_path] = {
'subdirectories': dirs,
'file_count': len(files),
'files': files[:10] if len(files) <= 10 else files[:10] + ['...'] # Limit to avoid huge output
}
# Count files by extension
for file in files:
file_path = Path(root) / file
ext = file_path.suffix.lower()
analysis['file_counts'][ext] = analysis['file_counts'].get(ext, 0) + 1
analysis['total_files'] += 1
# Check if it's an image
if ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
analysis['image_extensions'].add(ext)
analysis['total_images'] += 1
# Convert set to list for JSON serialization
analysis['image_extensions'] = list(analysis['image_extensions'])
# Try to identify class structure
analysis['potential_classes'] = self._identify_classes(dataset_dir)
self.logger.info(f"Dataset {dataset_name} analysis complete:")
self.logger.info(f" Total files: {analysis['total_files']}")
self.logger.info(f" Total images: {analysis['total_images']}")
self.logger.info(f" Size: {analysis['total_size_mb']:.1f} MB")
self.logger.info(f" Potential classes: {analysis['potential_classes']}")
return analysis
def _identify_classes(self, dataset_dir: Path) -> List[str]:
"""Identify potential class directories in dataset"""
potential_classes = []
# Look for common dataset organizations
for subdir in dataset_dir.iterdir():
if subdir.is_dir():
# Check if subdirectory contains images
image_count = sum(1 for f in subdir.rglob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
if image_count > 10: # Threshold for considering it a class directory
potential_classes.append(subdir.name)
# Also check for train/val/test structure
train_dir = dataset_dir / 'train'
if train_dir.exists():
for subdir in train_dir.iterdir():
if subdir.is_dir():
image_count = sum(1 for f in subdir.rglob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp'])
if image_count > 5:
potential_classes.append(f"train/{subdir.name}")
return sorted(potential_classes)
def prepare_dataset_for_training(self, dataset_name: str,
output_format: str = 'organized',
train_split: float = 0.7,
val_split: float = 0.15,
test_split: float = 0.15) -> Dict:
"""
Prepare dataset for training with proper splits and organization
Args:
dataset_name: Name of dataset to prepare
output_format: 'organized' for class folders or 'flat' for single directory
train_split: Proportion for training set
val_split: Proportion for validation set test_split: Proportion for test set
Returns:
Dictionary with preparation results
"""
dataset_dir = self.base_data_dir / dataset_name
prepared_dir = self.base_data_dir / f"{dataset_name}_prepared"
if not dataset_dir.exists():
self.logger.error(f"Dataset {dataset_name} not found. Download it first.")
return {}
# Analyze dataset first
analysis = self.analyze_dataset_structure(dataset_name)
# Create prepared dataset directory
prepared_dir.mkdir(exist_ok=True)
preparation_results = {
'dataset_name': dataset_name,
'prepared_path': str(prepared_dir),
'splits': {
'train': train_split,
'validation': val_split,
'test': test_split
},
'classes': {},
'total_samples': 0,
'preparation_time': 0
}
start_time = time.time()
try:
# Identify dataset structure and prepare accordingly
if dataset_name == 'colorectal_cancer':
preparation_results.update(
self._prepare_colorectal_dataset(dataset_dir, prepared_dir, train_split, val_split, test_split)
)
elif dataset_name == 'brain_tumor':
preparation_results.update(
self._prepare_brain_tumor_dataset(dataset_dir, prepared_dir, train_split, val_split, test_split)
)
elif dataset_name == 'chest_xray':
preparation_results.update(
self._prepare_chest_xray_dataset(dataset_dir, prepared_dir, train_split, val_split, test_split)
)
else:
# Generic preparation
preparation_results.update(
self._prepare_generic_dataset(dataset_dir, prepared_dir, train_split, val_split, test_split)
)
preparation_results['preparation_time'] = time.time() - start_time
# Save preparation metadata
metadata_file = prepared_dir / 'dataset_metadata.json'
with open(metadata_file, 'w') as f:
json.dump(preparation_results, f, indent=2)
# Create labels CSV file
self._create_labels_csv(prepared_dir, preparation_results)
self.logger.info(f"Dataset {dataset_name} prepared successfully")
self.logger.info(f"Prepared dataset location: {prepared_dir}")
self.logger.info(f"Total samples: {preparation_results['total_samples']}")
self.logger.info(f"Classes: {list(preparation_results['classes'].keys())}")
return preparation_results
except Exception as e:
self.logger.error(f"Failed to prepare dataset {dataset_name}: {str(e)}")
return {}
def _prepare_colorectal_dataset(self, dataset_dir: Path, prepared_dir: Path, train_split: float, val_split: float, test_split: float) -> Dict:
"""Prepare colorectal cancer dataset"""
self.logger.info("Preparing Colorectal Cancer dataset...")
# Implementation would depend on the actual dataset structure
# This is a template that should be customized based on the real dataset
results = {
'classes': {'benign': 0, 'malignant': 0},
'total_samples': 0,
'dataset_type': 'colorectal_cancer'
}
# Create train/val/test directories
for split in ['train', 'validation', 'test']:
(prepared_dir / split).mkdir(exist_ok=True)
for class_name in results['classes'].keys():
(prepared_dir / split / class_name).mkdir(exist_ok=True)
# Placeholder implementation
self.logger.warning("Colorectal dataset preparation needs to be customized based on actual dataset structure")
return results
def _prepare_brain_tumor_dataset(self, dataset_dir: Path, prepared_dir: Path,
train_split: float, val_split: float, test_split: float) -> Dict:
"""Prepare brain tumor MRI dataset"""
self.logger.info("Preparing Brain Tumor MRI dataset...")
results = {
'classes': {'glioma_tumor': 0, 'meningioma_tumor': 0, 'no_tumor': 0, 'pituitary_tumor': 0},
'total_samples': 0,
'dataset_type': 'brain_tumor'
}
# Create train/val/test directories
for split in ['train', 'validation', 'test']:
(prepared_dir / split).mkdir(exist_ok=True)
for class_name in results['classes'].keys():
(prepared_dir / split / class_name).mkdir(exist_ok=True)
# Placeholder implementation
self.logger.warning("Brain tumor dataset preparation needs to be customized based on actual dataset structure")
return results
def _prepare_chest_xray_dataset(self, dataset_dir: Path, prepared_dir: Path,
train_split: float, val_split: float, test_split: float) -> Dict:
"""Prepare chest X-ray pneumonia dataset"""
self.logger.info("Preparing Chest X-ray Pneumonia dataset...")
results = {
'classes': {'NORMAL': 0, 'PNEUMONIA': 0},
'total_samples': 0,
'dataset_type': 'chest_xray'
}
# This dataset typically comes pre-split, so we might need to reorganize
# Check for existing train/val/test structure
existing_train = dataset_dir / 'train'
existing_val = dataset_dir / 'val'
existing_test = dataset_dir / 'test'
# Create our organized structure
for split in ['train', 'validation', 'test']:
(prepared_dir / split).mkdir(exist_ok=True)
for class_name in results['classes'].keys():
(prepared_dir / split / class_name).mkdir(exist_ok=True)
# Placeholder implementation
self.logger.warning("Chest X-ray dataset preparation needs to be customized based on actual dataset structure")
return results
def _prepare_generic_dataset(self, dataset_dir: Path, prepared_dir: Path,
train_split: float, val_split: float, test_split: float) -> Dict:
"""Generic dataset preparation for unknown structure"""
self.logger.info("Preparing dataset with generic method...")
results = {
'classes': {},
'total_samples': 0,
'dataset_type': 'generic'
}
# Find all images and try to organize them
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']
all_images = []
for ext in image_extensions:
all_images.extend(dataset_dir.rglob(f'*{ext}'))
all_images.extend(dataset_dir.rglob(f'*{ext.upper()}'))
# Try to identify classes from directory structure
class_images = {}
for image_path in all_images:
# Use parent directory as class name
potential_class = image_path.parent.name
if potential_class not in class_images:
class_images[potential_class] = []
class_images[potential_class].append(image_path)
results['classes'] = {class_name: len(images) for class_name, images in class_images.items()}
results['total_samples'] = len(all_images)
self.logger.info(f"Found {len(all_images)} images in {len(class_images)} potential classes")
return results
def _create_labels_csv(self, prepared_dir: Path, preparation_results: Dict):
"""Create labels CSV file for the prepared dataset"""
labels_data = []
for split in ['train', 'validation', 'test']:
split_dir = prepared_dir / split
if split_dir.exists():
for class_dir in split_dir.iterdir():
if class_dir.is_dir():
class_name = class_dir.name
for image_file in class_dir.iterdir():
if image_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
labels_data.append({
'image_path': f"{split}/{class_name}/{image_file.name}",
'label': class_name,
'split': split,
'dataset': preparation_results['dataset_name']
})
# Create CSV file
if labels_data:
labels_df = pd.DataFrame(labels_data)
labels_file = prepared_dir / 'labels.csv'
labels_df.to_csv(labels_file, index=False)
self.logger.info(f"Created labels CSV with {len(labels_data)} entries")
else:
self.logger.warning("No label data found to create CSV file")
def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, bool]:
"""Download all three medical datasets"""
self.logger.info("Starting download of all medical datasets...")
results = {}
for dataset_name in self.datasets_config.keys():
self.logger.info(f"\nDownloading {dataset_name}...")
results[dataset_name] = self.download_dataset(dataset_name, force_redownload)
# Summary
successful = sum(1 for success in results.values() if success)
self.logger.info(f"\nDownload Summary: {successful}/{len(results)} datasets downloaded successfully")
for dataset_name, success in results.items():
status = "Success" if success else "Failed"
self.logger.info(f" {dataset_name}: {status}")
return results
def prepare_all_datasets(self) -> Dict[str, Dict]:
"""Prepare all downloaded datasets for training"""
self.logger.info("Starting preparation of all datasets...")
results = {}
for dataset_name in self.datasets_config.keys():
self.logger.info(f"\nPreparing {dataset_name}...")
results[dataset_name] = self.prepare_dataset_for_training(dataset_name)
# Summary
successful = sum(1 for result in results.values() if result)
self.logger.info(f"\nPreparation Summary: {successful}/{len(results)} datasets prepared successfully")
return results
def generate_dataset_report(self) -> str:
"""Generate a comprehensive report of all datasets"""
report_lines = [
"Medical Datasets Report",
"=" * 50,
f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
f"Base Directory: {self.base_data_dir}",
""
]
for dataset_name, config in self.datasets_config.items():
report_lines.append(f"Dataset: {dataset_name}")
report_lines.append("-" * 30)
report_lines.append(f"Description: {config['description']}")
report_lines.append(f"Kaggle URL: {config['url']}")
# Check if downloaded
dataset_dir = self.base_data_dir / dataset_name
if dataset_dir.exists():
analysis = self.analyze_dataset_structure(dataset_name)
report_lines.append(f"Status: Downloaded")
report_lines.append(f"Total Images: {analysis.get('total_images', 0)}")
report_lines.append(f"Size: {analysis.get('total_size_mb', 0):.1f} MB")
report_lines.append(f"Classes: {analysis.get('potential_classes', [])}")
else:
report_lines.append(f"Status: Not downloaded")
# Check if prepared
prepared_dir = self.base_data_dir / f"{dataset_name}_prepared"
if prepared_dir.exists():
metadata_file = prepared_dir / 'dataset_metadata.json'
if metadata_file.exists():
with open(metadata_file, 'r') as f:
metadata = json.load(f)
report_lines.append(f"Prepared: Yes ({metadata.get('total_samples', 0)} samples)")
else:
report_lines.append(f"Prepared: Yes (metadata missing)")
else:
report_lines.append(f"Prepared: No")
report_lines.append("")
report = "\n".join(report_lines)
# Save report
report_file = self.base_data_dir / "datasets_report.txt"
with open(report_file, 'w') as f:
f.write(report)
self.logger.info(f"Dataset report saved to: {report_file}")
return report
def main():
"""Main function for testing dataset preparation"""
print("Medical Dataset Preparation Tool")
print("=" * 50)
# Initialize preparer
preparer = MedicalDatasetPreparer(
base_data_dir="./medical_datasets",
target_image_size=(224, 224),
verbose=True
)
print("\nDataset configurations:")
for name, config in preparer.datasets_config.items():
print(f" {name}: {config['description']}")
# Check Kaggle setup
print(f"\nChecking Kaggle API setup...")
kaggle_available = preparer.check_kaggle_setup()
if not kaggle_available:
print("\nKaggle API not available.")
print("To download datasets automatically:")
print("1. pip install kaggle")
print("2. Get API credentials from https://www.kaggle.com/account")
print("3. Place kaggle.json in ~/.kaggle/")
print("\nDataset URLs for manual download:")
for name, config in preparer.datasets_config.items():
print(f" {name}: {config['url']}")
else:
print("Kaggle API is available and authenticated")
# For demonstration, we won't actually download to avoid long operations
print("\nTo download all datasets, run:")
print(" preparer.download_all_datasets()")
print(" preparer.prepare_all_datasets()")
# Generate report
print("\nGenerating dataset report...")
report = preparer.generate_dataset_report()
print(report)
if __name__ == "__main__":
main()