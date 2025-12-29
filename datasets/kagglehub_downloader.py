#!/usr/bin/env python3
"""
Enhanced Dataset Downloader using KaggleHub
===========================================
Modern approach to downloading medical datasets using the new kagglehub library
for the three required medical datasets:
1. Colorectal Cancer Global Dataset
2. Brain Tumor MRI Dataset
3. Chest X-ray Pneumonia Dataset
Author: Medical AI Research Team
Version: 1.0.0
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import logging
from datetime import datetime
import shutil
try:
import kagglehub
KAGGLEHUB_AVAILABLE = True
print(" KaggleHub available for dataset download")
except ImportError:
KAGGLEHUB_AVAILABLE = False
print(" KaggleHub not available. Install with: pip install kagglehub")
try:
from PIL import Image
PIL_AVAILABLE = True
except ImportError:
PIL_AVAILABLE = False
print(" PIL not available. Image processing will be limited.")
class KaggleHubDownloader:
"""
Enhanced dataset downloader using KaggleHub
"""
def __init__(self, base_data_dir: str = "./medical_datasets_kagglehub",
verbose: bool = True):
"""
Initialize KaggleHub downloader
Args:
base_data_dir: Directory to store downloaded datasets
verbose: Enable detailed logging
"""
self.base_data_dir = Path(base_data_dir)
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
# Dataset configurations for KaggleHub
self.datasets_config = {
'colorectal_cancer': {
'dataset_id': 'ankushpanday2/colorectal-cancer-global-dataset-and-predictions',
'description': 'Colorectal Cancer Global Dataset and Predictions',
'expected_files': ['train.csv', 'test.csv'], # Common file patterns
'data_type': 'mixed' # Images + CSV
},
'brain_tumor': {
'dataset_id': 'masoudnickparvar/brain-tumor-mri-dataset', 'description': 'Brain Tumor MRI Dataset',
'expected_files': [], # Will be images in folders
'data_type': 'images'
},
'chest_xray': {
'dataset_id': 'paultimothymooney/chest-xray-pneumonia',
'description': 'Chest X-ray Pneumonia Dataset', 'expected_files': [], # Will be images in train/val/test folders
'data_type': 'images'
}
}
self.logger.info(f"KaggleHub Downloader initialized")
self.logger.info(f"Base directory: {self.base_data_dir}")
def check_kagglehub_availability(self) -> bool:
"""Check if KaggleHub is available and working"""
if not KAGGLEHUB_AVAILABLE:
self.logger.error("KaggleHub not available. Install with: pip install kagglehub")
return False
try:
# Test basic kagglehub functionality
self.logger.info("KaggleHub is available and ready")
return True
except Exception as e:
self.logger.error(f"KaggleHub error: {str(e)}")
return False
def download_colorectal_cancer_dataset(self) -> Dict:
"""Download colorectal cancer dataset using KaggleHub"""
self.logger.info("Downloading Colorectal Cancer Dataset...")
if not self.check_kagglehub_availability():
return {'success': False, 'error': 'KaggleHub not available'}
try:
dataset_id = self.datasets_config['colorectal_cancer']['dataset_id']
output_dir = self.base_data_dir / 'colorectal_cancer'
self.logger.info(f"Downloading dataset: {dataset_id}")
# Download entire dataset
download_path = kagglehub.dataset_download(dataset_id)
self.logger.info(f"Dataset downloaded to: {download_path}")
# Copy to our organized location
if output_dir.exists():
shutil.rmtree(output_dir)
shutil.copytree(download_path, output_dir)
# Analyze downloaded content
analysis = self._analyze_downloaded_dataset(output_dir, 'colorectal_cancer')
result = {
'success': True,
'dataset_name': 'colorectal_cancer',
'download_path': str(download_path),
'organized_path': str(output_dir),
'analysis': analysis
}
self.logger.info(f" Colorectal cancer dataset downloaded successfully")
return result
except Exception as e:
self.logger.error(f"Failed to download colorectal cancer dataset: {str(e)}")
return {'success': False, 'error': str(e)}
def download_brain_tumor_dataset(self) -> Dict:
"""Download brain tumor MRI dataset using KaggleHub"""
self.logger.info("Downloading Brain Tumor MRI Dataset...")
if not self.check_kagglehub_availability():
return {'success': False, 'error': 'KaggleHub not available'}
try:
dataset_id = self.datasets_config['brain_tumor']['dataset_id']
output_dir = self.base_data_dir / 'brain_tumor'
self.logger.info(f"Downloading dataset: {dataset_id}")
# Download entire dataset
download_path = kagglehub.dataset_download(dataset_id)
self.logger.info(f"Dataset downloaded to: {download_path}")
# Copy to our organized location
if output_dir.exists():
shutil.rmtree(output_dir)
shutil.copytree(download_path, output_dir)
# Analyze downloaded content
analysis = self._analyze_downloaded_dataset(output_dir, 'brain_tumor')
result = {
'success': True,
'dataset_name': 'brain_tumor',
'download_path': str(download_path),
'organized_path': str(output_dir),
'analysis': analysis
}
self.logger.info(f" Brain tumor dataset downloaded successfully")
return result
except Exception as e:
self.logger.error(f"Failed to download brain tumor dataset: {str(e)}")
return {'success': False, 'error': str(e)}
def download_chest_xray_dataset(self) -> Dict:
"""Download chest X-ray pneumonia dataset using KaggleHub"""
self.logger.info("Downloading Chest X-ray Pneumonia Dataset...")
if not self.check_kagglehub_availability():
return {'success': False, 'error': 'KaggleHub not available'}
try:
dataset_id = self.datasets_config['chest_xray']['dataset_id']
output_dir = self.base_data_dir / 'chest_xray'
self.logger.info(f"Downloading dataset: {dataset_id}")
# Download entire dataset
download_path = kagglehub.dataset_download(dataset_id)
self.logger.info(f"Dataset downloaded to: {download_path}")
# Copy to our organized location
if output_dir.exists():
shutil.rmtree(output_dir)
shutil.copytree(download_path, output_dir)
# Analyze downloaded content
analysis = self._analyze_downloaded_dataset(output_dir, 'chest_xray')
result = {
'success': True,
'dataset_name': 'chest_xray',
'download_path': str(download_path),
'organized_path': str(output_dir),
'analysis': analysis
}
self.logger.info(f" Chest X-ray dataset downloaded successfully")
return result
except Exception as e:
self.logger.error(f"Failed to download chest X-ray dataset: {str(e)}")
return {'success': False, 'error': str(e)}
def _analyze_downloaded_dataset(self, dataset_dir: Path, dataset_name: str) -> Dict:
"""Analyze the structure and content of downloaded dataset"""
analysis = {
'dataset_name': dataset_name,
'total_files': 0,
'total_size_mb': 0,
'file_types': {},
'directory_structure': {},
'image_count': 0,
'csv_files': [],
'potential_classes': []
}
try:
# Walk through all files
for root, dirs, files in os.walk(dataset_dir):
rel_path = os.path.relpath(root, dataset_dir)
for file in files:
file_path = Path(root) / file
file_size = file_path.stat().st_size
file_ext = file_path.suffix.lower()
analysis['total_files'] += 1
analysis['total_size_mb'] += file_size / (1024 * 1024)
# Count file types
analysis['file_types'][file_ext] = analysis['file_types'].get(file_ext, 0) + 1
# Count images
if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
analysis['image_count'] += 1
# Note CSV files
if file_ext == '.csv':
analysis['csv_files'].append(str(file_path.relative_to(dataset_dir)))
# Record directory structure
if rel_path != '.':
analysis['directory_structure'][rel_path] = {
'subdirs': dirs,
'file_count': len(files)
}
# Potential class directories (containing images)
if any(f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')) for f in files):
if len(files) > 5: # Threshold for class directory
analysis['potential_classes'].append(rel_path)
self.logger.info(f"Dataset analysis complete:")
self.logger.info(f" Total files: {analysis['total_files']}")
self.logger.info(f" Total images: {analysis['image_count']}")
self.logger.info(f" Size: {analysis['total_size_mb']:.1f} MB")
self.logger.info(f" CSV files: {len(analysis['csv_files'])}")
self.logger.info(f" Potential classes: {analysis['potential_classes']}")
except Exception as e:
self.logger.error(f"Error analyzing dataset: {str(e)}")
analysis['analysis_error'] = str(e)
return analysis
def download_all_datasets(self) -> Dict:
"""Download all three medical datasets"""
self.logger.info("Starting download of all medical datasets using KaggleHub...")
results = {
'download_summary': {
'timestamp': datetime.now().isoformat(),
'total_datasets': 3,
'successful_downloads': 0,
'failed_downloads': 0
},
'dataset_results': {}
}
# Download each dataset
datasets = ['colorectal_cancer', 'brain_tumor', 'chest_xray']
download_functions = [
self.download_colorectal_cancer_dataset,
self.download_brain_tumor_dataset,
self.download_chest_xray_dataset
]
for dataset_name, download_func in zip(datasets, download_functions):
self.logger.info(f"\n{'='*60}")
self.logger.info(f"Downloading {dataset_name.upper()} Dataset")
self.logger.info(f"{'='*60}")
try:
result = download_func()
results['dataset_results'][dataset_name] = result
if result['success']:
results['download_summary']['successful_downloads'] += 1
self.logger.info(f" {dataset_name} download successful")
else:
results['download_summary']['failed_downloads'] += 1
self.logger.error(f" {dataset_name} download failed: {result.get('error', 'Unknown error')}")
except Exception as e:
results['dataset_results'][dataset_name] = {'success': False, 'error': str(e)}
results['download_summary']['failed_downloads'] += 1
self.logger.error(f" {dataset_name} download failed with exception: {str(e)}")
# Save download summary
summary_file = self.base_data_dir / 'download_summary.json'
with open(summary_file, 'w') as f:
json.dump(results, f, indent=2)
# Final summary
successful = results['download_summary']['successful_downloads']
total = results['download_summary']['total_datasets']
self.logger.info(f"\n{'='*60}")
self.logger.info(f"DOWNLOAD SUMMARY")
self.logger.info(f"{'='*60}")
self.logger.info(f"Successful: {successful}/{total} datasets")
self.logger.info(f"Failed: {results['download_summary']['failed_downloads']}/{total} datasets")
self.logger.info(f"Download summary saved: {summary_file}")
return results
def load_specific_file(self, dataset_name: str, file_path: str = "") -> Optional[pd.DataFrame]:
"""
Load a specific file from a dataset (especially useful for CSV files)
Args:
dataset_name: Name of the dataset
file_path: Specific file path within the dataset
Returns:
DataFrame if successful, None otherwise
"""
if dataset_name not in self.datasets_config:
self.logger.error(f"Unknown dataset: {dataset_name}")
return None
try:
dataset_id = self.datasets_config[dataset_name]['dataset_id']
self.logger.info(f"Loading file '{file_path}' from dataset: {dataset_id}")
# Use KaggleHub to load specific file
df = kagglehub.dataset_download(dataset_id)
# If file_path is specified, try to load that specific file
if file_path:
full_path = Path(df) / file_path
if full_path.exists() and full_path.suffix.lower() == '.csv':
return pd.read_csv(full_path)
else:
self.logger.warning(f"File {file_path} not found or not a CSV file")
return None
else:
# Return the download path for further processing
self.logger.info(f"Dataset downloaded to: {df}")
return None
except Exception as e:
self.logger.error(f"Failed to load file from {dataset_name}: {str(e)}")
return None
def test_colorectal_cancer_download():
"""Test the colorectal cancer dataset download as shown in your example"""
print("Testing Colorectal Cancer Dataset Download")
print("="*60)
# Initialize downloader
downloader = KaggleHubDownloader(
base_data_dir="./test_kagglehub_datasets",
verbose=True
)
# Download colorectal cancer dataset
result = downloader.download_colorectal_cancer_dataset()
if result['success']:
print(f" Download successful!")
print(f"Dataset path: {result['organized_path']}")
print(f"Analysis: {result['analysis']}")
# Try to find and load CSV files
dataset_dir = Path(result['organized_path'])
csv_files = list(dataset_dir.rglob('*.csv'))
if csv_files:
print(f"\nFound {len(csv_files)} CSV files:")
for csv_file in csv_files[:3]: # Show first 3
print(f" - {csv_file.relative_to(dataset_dir)}")
try:
df = pd.read_csv(csv_file)
print(f" Shape: {df.shape}")
print(f" Columns: {list(df.columns)[:5]}...") # First 5 columns
if len(df) > 0:
print(f" Sample record: {df.iloc[0].to_dict()}")
except Exception as e:
print(f" Error loading CSV: {str(e)}")
print()
else:
print("No CSV files found in the dataset")
else:
print(f" Download failed: {result['error']}")
def main():
"""Main function to test KaggleHub functionality"""
print("KaggleHub Medical Dataset Downloader")
print("="*60)
if not KAGGLEHUB_AVAILABLE:
print(" KaggleHub not available. Install with:")
print(" pip install kagglehub")
return
# Test single dataset download
test_colorectal_cancer_download()
print("\n To download all datasets, use:")
print(" downloader = KaggleHubDownloader()")
print(" results = downloader.download_all_datasets()")
if __name__ == "__main__":
main()