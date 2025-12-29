#!/usr/bin/env python3
"""
Kaggle Medical Datasets Downloader
=================================
Downloads and prepares three real medical datasets from Kaggle:
1. Colorectal Cancer Dataset 2. Brain Tumor MRI Dataset
3. Chest X-ray Pneumonia Dataset
Requirements:
- kaggle API: pip install kaggle
- Set up Kaggle API credentials: ~/.kaggle/kaggle.json
Author: Medical AI Research Team
Version: 1.0.0
"""
import os
import zipfile
import shutil
from pathlib import Path
import requests
import json
from typing import Dict, List, Tuple, Optional
import subprocess
import sys
class KaggleMedicalDownloader:
"""Downloads and prepares Kaggle medical datasets for GOHBORESNET18 evaluation"""
def __init__(self, download_dir: str = "medical_datasets"):
self.download_dir = Path(download_dir)
self.download_dir.mkdir(exist_ok=True)
# Dataset configurations
self.datasets = {
'colorectal_cancer': {
'name': 'Colorectal Cancer',
'kaggle_dataset': 'kmader/colorectal-histology-mnist',
'local_dir': self.download_dir / 'colorectal_cancer',
'image_size': (224, 224),
'num_classes': 8,
'expected_files': ['kather_texture_2016_image_tiles_5000.zip']
},
'brain_tumor': {
'name': 'Brain Tumor MRI',
'kaggle_dataset': 'masoudnickparvar/brain-tumor-mri-dataset',
'local_dir': self.download_dir / 'brain_tumor',
'image_size': (224, 224), 'num_classes': 4,
'expected_files': ['Training', 'Testing']
},
'chest_xray': {
'name': 'Chest X-ray Pneumonia',
'kaggle_dataset': 'paultimothymooney/chest-xray-pneumonia',
'local_dir': self.download_dir / 'chest_xray',
'image_size': (224, 224),
'num_classes': 2,
'expected_files': ['chest_xray.zip']
}
}
def check_kaggle_setup(self) -> bool:
"""Check if Kaggle API is properly configured"""
try:
# Check if kaggle is installed
subprocess.run(['kaggle', '--version'], capture_output=True, check=True)
# Check if API key exists
kaggle_dir = Path.home() / '.kaggle'
if not (kaggle_dir / 'kaggle.json').exists():
print("ERROR: Kaggle API key not found!")
print("Please set up your Kaggle API credentials:")
print("1. Go to https://www.kaggle.com/account")
print("2. Create API token and download kaggle.json")
print("3. Place it in ~/.kaggle/kaggle.json")
return False
return True
except (subprocess.CalledProcessError, FileNotFoundError):
print("ERROR: Kaggle CLI not installed!")
print("Install with: pip install kaggle")
return False
def download_dataset(self, dataset_key: str) -> bool:
"""Download a specific dataset from Kaggle"""
if dataset_key not in self.datasets:
print(f"ERROR: Unknown dataset: {dataset_key}")
return False
dataset_info = self.datasets[dataset_key]
dataset_name = dataset_info['name']
kaggle_dataset = dataset_info['kaggle_dataset']
local_dir = dataset_info['local_dir']
print(f"\nDownloading {dataset_name}...")
print(f" Kaggle dataset: {kaggle_dataset}")
print(f" Local directory: {local_dir}")
# Create dataset directory
local_dir.mkdir(exist_ok=True)
try:
# Download using kaggle CLI
cmd = ['kaggle', 'datasets', 'download', '-d', kaggle_dataset, '-p', str(local_dir), '--unzip']
result = subprocess.run(cmd, capture_output=True, text=True)
if result.returncode != 0:
print(f"ERROR: Download failed: {result.stderr}")
return False
print(f"SUCCESS: {dataset_name} downloaded successfully!")
# Verify expected files exist
missing_files = []
for expected_file in dataset_info['expected_files']:
if not (local_dir / expected_file).exists():
missing_files.append(expected_file)
if missing_files:
print(f"WARNING: Missing expected files: {missing_files}")
return True
except Exception as e:
print(f"ERROR: Error downloading {dataset_name}: {e}")
return False
def download_all_datasets(self) -> Dict[str, bool]:
"""Download all three medical datasets"""
print("Starting Kaggle Medical Datasets Download")
print("=" * 50)
if not self.check_kaggle_setup():
return {}
results = {}
for dataset_key in self.datasets.keys():
success = self.download_dataset(dataset_key)
results[dataset_key] = success
# Summary
print(f"\nDownload Summary:")
print("-" * 30)
successful = sum(results.values())
total = len(results)
for dataset_key, success in results.items():
dataset_name = self.datasets[dataset_key]['name']
status = "SUCCESS" if success else "FAILED"
print(f"{status} {dataset_name}")
print(f"\nSuccessful downloads: {successful}/{total}")
return results
def get_dataset_info(self, dataset_key: str) -> Optional[Dict]:
"""Get information about a downloaded dataset"""
if dataset_key not in self.datasets:
return None
dataset_info = self.datasets[dataset_key].copy()
local_dir = dataset_info['local_dir']
if not local_dir.exists():
return None
# Count files
image_files = []
for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
image_files.extend(list(local_dir.rglob(f'*{ext}')))
image_files.extend(list(local_dir.rglob(f'*{ext.upper()}')))
dataset_info['total_images'] = len(image_files)
dataset_info['available'] = True
dataset_info['local_path'] = str(local_dir)
return dataset_info
def create_dataset_index(self) -> Dict:
"""Create index of all available datasets"""
index = {
'datasets': {},
'summary': {
'total_datasets': len(self.datasets),
'available_datasets': 0,
'total_images': 0
}
}
for dataset_key in self.datasets.keys():
info = self.get_dataset_info(dataset_key)
if info:
index['datasets'][dataset_key] = info
index['summary']['available_datasets'] += 1
index['summary']['total_images'] += info.get('total_images', 0)
# Save index
index_file = self.download_dir / 'dataset_index.json'
with open(index_file, 'w') as f:
json.dump(index, f, indent=2)
return index
def main():
"""Main function for standalone usage"""
downloader = KaggleMedicalDownloader()
print("Kaggle Medical Datasets Downloader")
print("=" * 40)
# Download all datasets
results = downloader.download_all_datasets()
if any(results.values()):
# Create dataset index
index = downloader.create_dataset_index()
print(f"\nDataset Index Created:")
print(f" Available datasets: {index['summary']['available_datasets']}")
print(f" Total images: {index['summary']['total_images']}")
# Save results
results_file = downloader.download_dir / 'download_results.json'
with open(results_file, 'w') as f:
json.dump({'download_results': results, 'dataset_index': index}, f, indent=2)
print(f" Results saved to: {results_file}")
else:
print("\nERROR: No datasets were downloaded successfully")
if __name__ == "__main__":
main()