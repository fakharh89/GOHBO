#!/usr/bin/env python3
"""
Download All Three Medical Datasets using KaggleHub
===================================================
Downloads all required datasets using the kagglehub library as demonstrated:
1. Colorectal Cancer Global Dataset
2. Brain Tumor MRI Dataset
3. Chest X-ray Pneumonia Dataset
Author: Medical AI Research Team
Version: 1.0.0
"""
import kagglehub
import pandas as pd
from pathlib import Path
import os
import json
from datetime import datetime
def download_colorectal_cancer():
"""Download colorectal cancer dataset"""
print(" Downloading Colorectal Cancer Dataset...")
try:
# Download latest version
path = kagglehub.dataset_download("ankushpanday2/colorectal-cancer-global-dataset-and-predictions")
print(f" Colorectal Cancer dataset downloaded to: {path}")
# Analyze the dataset
dataset_path = Path(path)
csv_files = list(dataset_path.glob('*.csv'))
if csv_files:
df = pd.read_csv(csv_files[0])
print(f" Dataset shape: {df.shape}")
print(f" Columns: {df.columns.tolist()[:5]}...") print(f" Sample record keys: {list(df.iloc[0].keys())[:5]}...")
return {
'success': True,
'path': str(path),
'dataset': 'colorectal_cancer',
'files': [str(f) for f in dataset_path.iterdir()],
'analysis': {
'total_files': len(list(dataset_path.iterdir())),
'csv_files': len(csv_files),
'dataset_shape': df.shape if csv_files else None
}
}
except Exception as e:
print(f" Error downloading colorectal cancer dataset: {e}")
return {'success': False, 'error': str(e), 'dataset': 'colorectal_cancer'}
def download_brain_tumor():
"""Download brain tumor MRI dataset"""
print("\n Downloading Brain Tumor MRI Dataset...")
try:
# Download latest version
path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
print(f" Brain Tumor MRI dataset downloaded to: {path}")
# Analyze the dataset
dataset_path = Path(path)
# Count image files and directories
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = []
for ext in image_extensions:
image_files.extend(list(dataset_path.rglob(f'*{ext}')))
image_files.extend(list(dataset_path.rglob(f'*{ext.upper()}')))
# Find potential class directories
class_dirs = []
for item in dataset_path.iterdir():
if item.is_dir():
# Check if directory contains images
dir_images = sum(1 for f in item.rglob('*') if f.suffix.lower() in image_extensions)
if dir_images > 5: # Threshold for class directory
class_dirs.append(item.name)
print(f" Total images found: {len(image_files)}")
print(f" Potential classes: {class_dirs}")
return {
'success': True,
'path': str(path),
'dataset': 'brain_tumor',
'analysis': {
'total_files': len(list(dataset_path.rglob('*'))),
'image_files': len(image_files),
'potential_classes': class_dirs,
'class_count': len(class_dirs)
}
}
except Exception as e:
print(f" Error downloading brain tumor dataset: {e}")
return {'success': False, 'error': str(e), 'dataset': 'brain_tumor'}
def download_chest_xray():
"""Download chest X-ray pneumonia dataset"""
print("\n Downloading Chest X-ray Pneumonia Dataset...")
try:
# Download latest version path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
print(f" Chest X-ray Pneumonia dataset downloaded to: {path}")
# Analyze the dataset
dataset_path = Path(path)
# Count image files and analyze structure
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
image_files = []
for ext in image_extensions:
image_files.extend(list(dataset_path.rglob(f'*{ext}')))
image_files.extend(list(dataset_path.rglob(f'*{ext.upper()}')))
# Look for train/val/test structure
train_dir = dataset_path / 'train' if (dataset_path / 'train').exists() else None
val_dir = dataset_path / 'val' if (dataset_path / 'val').exists() else None
test_dir = dataset_path / 'test' if (dataset_path / 'test').exists() else None
structure_info = {}
if train_dir:
train_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
structure_info['train'] = {
'classes': train_classes,
'total_images': len(list(train_dir.rglob('*.jpeg'))) + len(list(train_dir.rglob('*.jpg')))
}
if test_dir:
test_classes = [d.name for d in test_dir.iterdir() if d.is_dir()]
structure_info['test'] = {
'classes': test_classes,
'total_images': len(list(test_dir.rglob('*.jpeg'))) + len(list(test_dir.rglob('*.jpg')))
}
print(f" Total images found: {len(image_files)}")
print(f" Structure: {list(structure_info.keys())}")
return {
'success': True,
'path': str(path),
'dataset': 'chest_xray',
'analysis': {
'total_files': len(list(dataset_path.rglob('*'))),
'image_files': len(image_files),
'structure': structure_info,
'has_train_test_split': bool(train_dir and test_dir)
}
}
except Exception as e:
print(f" Error downloading chest X-ray dataset: {e}")
return {'success': False, 'error': str(e), 'dataset': 'chest_xray'}
def main():
"""Download all three medical datasets"""
print(" Medical Datasets Downloader using KaggleHub")
print("=" * 80)
# Download all datasets
results = {
'download_info': {
'timestamp': datetime.now().isoformat(),
'library': 'kagglehub',
'total_datasets': 3
},
'datasets': {}
}
# Download each dataset
datasets = [
('colorectal_cancer', download_colorectal_cancer),
('brain_tumor', download_brain_tumor),
('chest_xray', download_chest_xray)
]
successful_downloads = 0
for dataset_name, download_func in datasets:
result = download_func()
results['datasets'][dataset_name] = result
if result['success']:
successful_downloads += 1
# Summary
print(f"\nDownload Summary")
print("=" * 40)
print(f"Successful downloads: {successful_downloads}/3")
for dataset_name, result in results['datasets'].items():
status = "yes" if result['success'] else "no"
print(f"{status} {dataset_name}: {'Success' if result['success'] else result.get('error', 'Failed')}")
if result['success'] and 'analysis' in result:
analysis = result['analysis']
if dataset_name == 'colorectal_cancer':
print(f" Shape: {analysis.get('dataset_shape', 'N/A')}")
elif dataset_name == 'brain_tumor':
print(f" Images: {analysis.get('image_files', 0)}, Classes: {analysis.get('class_count', 0)}")
elif dataset_name == 'chest_xray':
print(f" Images: {analysis.get('image_files', 0)}, Structure: {list(analysis.get('structure', {}).keys())}")
# Save results summary
results_file = Path('./datasets_download_summary.json')
with open(results_file, 'w') as f:
json.dump(results, f, indent=2)
print(f"\n Download summary saved to: {results_file}")
if successful_downloads == 3:
print(" All datasets downloaded successfully!")
print("Ready for GOHBORESNET18 evaluation!")
else:
print(" Some datasets failed to download. Check error messages above.")
return results
if __name__ == "__main__":
main()