"""
Step 1: Medical Image Preprocessing
===================================
This module handles all medical image preprocessing tasks including:
- Medical image loading (DICOM, PNG, JPEG, etc.)
- Quality assessment and filtering
- Normalization and enhancement
- Data augmentation for training
- Train/validation/test splitting
Author: Medical AI Research Team
Version: 1.0.0
"""
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import json
import pickle
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
# Optional medical imaging libraries
try:
import pydicom
DICOM_AVAILABLE = True
except ImportError:
DICOM_AVAILABLE = False
try:
import nibabel as nib
NIBABEL_AVAILABLE = True
except ImportError:
NIBABEL_AVAILABLE = False
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
class MedicalImagePreprocessor:
"""Medical image preprocessing pipeline"""
def __init__(self, config: Dict[str, Any], output_dir: str):
"""
Initialize medical image preprocessor
Args:
config: Preprocessing configuration dictionary
output_dir: Directory to save preprocessing outputs
"""
self.config = config
self.output_dir = Path(output_dir)
self.output_dir.mkdir(parents=True, exist_ok=True)
# Setup logging
self.logger = self._setup_logger()
# Preprocessing parameters
self.target_size = tuple(config.get('target_size', [224, 224]))
self.normalization = config.get('normalization', 'z_score')
self.augmentation_enabled = config.get('augmentation', True)
self.quality_check = config.get('quality_check', True)
self.clahe_enabled = config.get('clahe_enabled', True)
# Statistics tracking
self.stats = {
'total_processed': 0,
'quality_passed': 0,
'quality_failed': 0,
'processing_times': [],
'quality_scores': []
}
self.logger.info("Medical Image Preprocessor initialized")
self.logger.info(f" Target size: {self.target_size}")
self.logger.info(f"Normalization: {self.normalization}")
self.logger.info(f"Augmentation: {self.augmentation_enabled}")
def _setup_logger(self) -> logging.Logger:
"""Setup logging for preprocessing step"""
logger = logging.getLogger(f'preprocessing_step')
logger.setLevel(logging.INFO)
# Remove existing handlers
for handler in logger.handlers[:]:
logger.removeHandler(handler)
# Create handler
log_file = self.output_dir / 'preprocessing.log'
handler = logging.FileHandler(log_file)
# Create formatter
formatter = logging.Formatter(
'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
return logger
def load_medical_image(self, image_path: str) -> np.ndarray:
"""
Load medical image from various formats
Args:
image_path: Path to the medical image
Returns:
numpy array containing the image data
"""
image_path = Path(image_path)
if not image_path.exists():
raise FileNotFoundError(f"Image not found: {image_path}")
# Load based on file extension
if image_path.suffix.lower() == '.dcm' and DICOM_AVAILABLE:
# DICOM image
dicom_data = pydicom.dcmread(str(image_path))
image = dicom_data.pixel_array
# Handle different DICOM pixel representations
if hasattr(dicom_data, 'RescaleSlope') and hasattr(dicom_data, 'RescaleIntercept'):
image = image * dicom_data.RescaleSlope + dicom_data.RescaleIntercept
elif image_path.suffix.lower() in ['.nii', '.nii.gz'] and NIBABEL_AVAILABLE:
# NIfTI image
nii_data = nib.load(str(image_path))
image = nii_data.get_fdata()
# Take middle slice for 3D volumes
if len(image.shape) == 3:
middle_slice = image.shape[2] // 2
image = image[:, :, middle_slice]
else:
# Standard image formats
image = cv2.imread(str(image_path))
if image is None:
raise ValueError(f"Could not load image: {image_path}")
# Convert BGR to RGB for OpenCV images
if len(image.shape) == 3:
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# Ensure proper data type
if image.dtype != np.uint8:
# Normalize to 0-255 range
image = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
return image
def assess_image_quality(self, image: np.ndarray) -> Dict[str, float]:
"""
Assess medical image quality
Args:
image: Input image array
Returns:
Dictionary with quality metrics
"""
if len(image.shape) == 3:
# Convert to grayscale for quality assessment
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
else:
gray = image
# Calculate quality metrics
quality_metrics = {}
# Signal-to-noise ratio approximation
mean_signal = np.mean(gray)
noise_std = np.std(gray)
snr = mean_signal / (noise_std + 1e-8)
quality_metrics['snr'] = float(snr)
# Contrast measure (standard deviation)
contrast = np.std(gray)
quality_metrics['contrast'] = float(contrast)
# Sharpness measure (variance of Laplacian)
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
sharpness = np.var(laplacian)
quality_metrics['sharpness'] = float(sharpness)
# Brightness measure
brightness = np.mean(gray)
quality_metrics['brightness'] = float(brightness)
# Overall quality score (weighted combination)
quality_score = (
0.3 * min(snr / 10.0, 1.0) + # SNR component
0.3 * min(contrast / 50.0, 1.0) + # Contrast component
0.2 * min(sharpness / 1000.0, 1.0) + # Sharpness component
0.2 * (1.0 - abs(brightness - 127.5) / 127.5) # Brightness component
)
quality_metrics['overall_quality'] = float(quality_score)
return quality_metrics
def normalize_image(self, image: np.ndarray) -> np.ndarray:
"""
Normalize medical image
Args:
image: Input image array
Returns:
Normalized image array
"""
image_float = image.astype(np.float32)
if self.normalization == 'min_max':
# Min-max normalization to [0, 1]
normalized = (image_float - image_float.min()) / (image_float.max() - image_float.min())
elif self.normalization == 'z_score':
# Z-score normalization
mean = np.mean(image_float)
std = np.std(image_float)
normalized = (image_float - mean) / (std + 1e-8)
elif self.normalization == 'percentile':
# Percentile-based normalization
p1, p99 = np.percentile(image_float, [1, 99])
normalized = np.clip((image_float - p1) / (p99 - p1), 0, 1)
else:
# No normalization
normalized = image_float / 255.0
return normalized
def enhance_image(self, image: np.ndarray) -> np.ndarray:
"""
Apply medical image enhancement
Args:
image: Input image array
Returns:
Enhanced image array
"""
if not self.clahe_enabled:
return image
# Convert to uint8 for CLAHE
if image.dtype != np.uint8:
img_uint8 = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
else:
img_uint8 = image
if len(img_uint8.shape) == 3:
# Apply CLAHE to each channel
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced_channels = []
for channel in range(img_uint8.shape[2]):
enhanced_channels.append(clahe.apply(img_uint8[:, :, channel]))
enhanced = np.stack(enhanced_channels, axis=2)
else:
# Apply CLAHE to grayscale
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(img_uint8)
return enhanced
def resize_image(self, image: np.ndarray) -> np.ndarray:
"""
Resize image to target size
Args:
image: Input image array
Returns:
Resized image array
"""
if len(image.shape) == 2:
# Grayscale image - convert to 3-channel
image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
# Resize maintaining aspect ratio and padding if needed
h, w = image.shape[:2]
target_h, target_w = self.target_size
# Calculate scaling factor
scale = min(target_h / h, target_w / w)
# Resize
new_h = int(h * scale)
new_w = int(w * scale)
resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
# Pad to target size
if new_h != target_h or new_w != target_w:
# Calculate padding
pad_h = (target_h - new_h) // 2
pad_w = (target_w - new_w) // 2
# Create padded image
padded = np.zeros((target_h, target_w, 3), dtype=image.dtype)
padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
resized = padded
return resized
def augment_image(self, image: np.ndarray, augment_params: Dict[str, Any] = None) -> np.ndarray:
"""
Apply data augmentation to medical image
Args:
image: Input image array
augment_params: Augmentation parameters
Returns:
Augmented image array
"""
if not self.augmentation_enabled or augment_params is None:
return image
augmented = image.copy()
# Random rotation
if 'rotation' in augment_params and np.random.random() < 0.5:
angle = np.random.uniform(-augment_params['rotation'], augment_params['rotation'])
h, w = augmented.shape[:2]
center = (w // 2, h // 2)
matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
augmented = cv2.warpAffine(augmented, matrix, (w, h))
# Random horizontal flip (usually not recommended for medical images)
if 'horizontal_flip' in augment_params and augment_params['horizontal_flip'] and np.random.random() < 0.3:
augmented = cv2.flip(augmented, 1)
# Brightness adjustment
if 'brightness' in augment_params and np.random.random() < 0.5:
brightness_factor = np.random.uniform(*augment_params['brightness'])
augmented = np.clip(augmented * brightness_factor, 0, 255).astype(augmented.dtype)
# Gaussian noise
if 'noise' in augment_params and np.random.random() < 0.3:
noise = np.random.normal(0, augment_params['noise'], augmented.shape)
augmented = np.clip(augmented + noise, 0, 255).astype(augmented.dtype)
return augmented
def process_single_image(self, image_path: str, augmentation_params: Dict[str, Any] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
"""
Process a single medical image
Args:
image_path: Path to the image
augmentation_params: Parameters for data augmentation
Returns:
Tuple of (processed_image, metadata)
"""
import time
start_time = time.time()
try:
# Load image
image = self.load_medical_image(image_path)
# Assess quality
quality_metrics = {}
if self.quality_check:
quality_metrics = self.assess_image_quality(image)
self.stats['quality_scores'].append(quality_metrics['overall_quality'])
# Check if image passes quality threshold
quality_threshold = self.config.get('quality_threshold', 0.3)
if quality_metrics['overall_quality'] < quality_threshold:
self.stats['quality_failed'] += 1
self.logger.warning(f"Image failed quality check: {image_path}")
return None, {'status': 'quality_failed', 'quality_metrics': quality_metrics}
# Enhance image
enhanced = self.enhance_image(image)
# Resize image
resized = self.resize_image(enhanced)
# Normalize image
normalized = self.normalize_image(resized)
# Apply augmentation if specified
if augmentation_params is not None:
final_image = self.augment_image((normalized * 255).astype(np.uint8), augmentation_params)
final_image = final_image.astype(np.float32) / 255.0
else:
final_image = normalized
# Calculate processing time
processing_time = time.time() - start_time
self.stats['processing_times'].append(processing_time)
self.stats['total_processed'] += 1
self.stats['quality_passed'] += 1
# Create metadata
metadata = {
'status': 'success',
'original_path': image_path,
'original_shape': image.shape,
'final_shape': final_image.shape,
'processing_time': processing_time,
'quality_metrics': quality_metrics,
'augmentation_applied': augmentation_params is not None
}
return final_image, metadata
except Exception as e:
self.logger.error(f"Error processing {image_path}: {str(e)}")
return None, {'status': 'error', 'error': str(e)}
def load_dataset_info(self, labels_file: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
"""
Load dataset information from labels file
Args:
labels_file: Path to labels CSV file
Returns:
Tuple of (dataframe, dataset_info)
"""
labels_path = Path(labels_file)
if not labels_path.exists():
raise FileNotFoundError(f"Labels file not found: {labels_file}")
# Load labels
df = pd.read_csv(labels_path)
# Basic validation
required_columns = ['image_path', 'label']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
raise ValueError(f"Missing required columns: {missing_columns}")
# Encode labels if they are strings
label_encoder = LabelEncoder()
if df['label'].dtype == 'object':
df['label_encoded'] = label_encoder.fit_transform(df['label'])
class_names = list(label_encoder.classes_)
else:
df['label_encoded'] = df['label']
class_names = [f"class_{i}" for i in range(df['label'].nunique())]
# Dataset statistics
dataset_info = {
'total_images': len(df),
'num_classes': df['label'].nunique(),
'class_names': class_names,
'class_distribution': df['label'].value_counts().to_dict(),
'has_patient_id': 'patient_id' in df.columns,
'has_temporal_info': any(col in df.columns for col in ['date', 'acquisition_date', 'timestamp'])
}
self.logger.info(f"Dataset loaded: {dataset_info['total_images']} images, {dataset_info['num_classes']} classes")
self.logger.info(f"Class distribution: {dataset_info['class_distribution']}")
return df, dataset_info
def split_dataset(self, df: pd.DataFrame, validation_split: float = 0.2, test_split: float = 0.2, patient_level: bool = True) -> Dict[str, pd.DataFrame]:
"""
Split dataset into train/validation/test sets
Args:
df: Dataset dataframe
validation_split: Fraction for validation set
test_split: Fraction for test set
patient_level: Whether to split at patient level
Returns:
Dictionary with train/val/test dataframes
"""
if patient_level and 'patient_id' in df.columns:
# Patient-level splitting to prevent data leakage
unique_patients = df['patient_id'].unique()
# First split: train vs (val + test)
train_patients, temp_patients = train_test_split(
unique_patients, test_size=(validation_split + test_split),
stratify=None, # Cannot stratify on patients directly
random_state=42
)
# Second split: val vs test
if test_split > 0:
val_patients, test_patients = train_test_split(
temp_patients,
test_size=test_split / (validation_split + test_split),
random_state=42
)
else:
val_patients = temp_patients
test_patients = []
# Create splits
train_df = df[df['patient_id'].isin(train_patients)].copy()
val_df = df[df['patient_id'].isin(val_patients)].copy()
test_df = df[df['patient_id'].isin(test_patients)].copy() if test_patients else pd.DataFrame()
self.logger.info(f" Patient-level split: {len(train_patients)} train patients, {len(val_patients)} val patients, {len(test_patients)} test patients")
else:
# Image-level splitting
# First split: train vs (val + test)
train_df, temp_df = train_test_split(
df,
test_size=(validation_split + test_split),
stratify=df['label_encoded'],
random_state=42
)
# Second split: val vs test
if test_split > 0:
val_df, test_df = train_test_split(
temp_df,
test_size=test_split / (validation_split + test_split),
stratify=temp_df['label_encoded'],
random_state=42
)
else:
val_df = temp_df
test_df = pd.DataFrame()
self.logger.info(f"Image-level split: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")
# Log class distributions
for split_name, split_df in [('Train', train_df), ('Validation', val_df), ('Test', test_df)]:
if len(split_df) > 0:
distribution = split_df['label'].value_counts().to_dict()
self.logger.info(f" {split_name} distribution: {distribution}")
splits = {
'train': train_df,
'validation': val_df,
'test': test_df
}
return splits
def preprocess_dataset(self, input_dir: str, labels_file: str, validation_split: float = 0.2, test_split: float = 0.2) -> Dict[str, Any]:
"""
Preprocess complete dataset
Args:
input_dir: Directory containing medical images
labels_file: Path to labels CSV file
validation_split: Fraction for validation set
test_split: Fraction for test set
Returns:
Dictionary containing preprocessing results
"""
self.logger.info(f"Starting dataset preprocessing")
self.logger.info(f" Input directory: {input_dir}")
self.logger.info(f"Labels file: {labels_file}")
# Load dataset info
df, dataset_info = self.load_dataset_info(labels_file)
# Split dataset
splits = self.split_dataset(df, validation_split, test_split, patient_level=self.config.get('patient_level_split', True))
# Preprocess each split
processed_splits = {}
output_paths = {}
# Augmentation parameters for training set
aug_params = {
'rotation': 15,
'horizontal_flip': False, # Usually not recommended for medical
'brightness': [0.9, 1.1],
'noise': 2.0
} if self.augmentation_enabled else None
for split_name, split_df in splits.items():
if len(split_df) == 0:
continue
self.logger.info(f"Processing {split_name} set ({len(split_df)} images)")
processed_images = []
metadata_list = []
valid_indices = []
for idx, row in split_df.iterrows():
image_path = Path(input_dir) / row['image_path']
# Apply augmentation only to training set
current_aug_params = aug_params if split_name == 'train' else None
processed_image, metadata = self.process_single_image(str(image_path), current_aug_params)
if processed_image is not None:
processed_images.append(processed_image)
metadata_list.append(metadata)
valid_indices.append(idx)
if processed_images:
# Stack images
images_array = np.stack(processed_images, axis=0)
# Get corresponding labels
valid_split_df = split_df.loc[valid_indices].copy()
labels_array = valid_split_df['label_encoded'].values
# Save processed data
split_output_dir = self.output_dir / split_name
split_output_dir.mkdir(exist_ok=True)
images_path = split_output_dir / 'images.npy'
labels_path = split_output_dir / 'labels.npy'
metadata_path = split_output_dir / 'metadata.json'
df_path = split_output_dir / 'dataframe.csv'
np.save(images_path, images_array)
np.save(labels_path, labels_array)
valid_split_df.to_csv(df_path, index=False)
with open(metadata_path, 'w') as f:
json.dump(metadata_list, f, indent=2, default=str)
processed_splits[split_name] = {
'images': images_array,
'labels': labels_array,
'dataframe': valid_split_df,
'metadata': metadata_list
}
output_paths[split_name] = {
'images': str(images_path),
'labels': str(labels_path),
'metadata': str(metadata_path),
'dataframe': str(df_path)
}
self.logger.info(f"{split_name} set: {len(processed_images)}/{len(split_df)} images processed successfully")
# Save preprocessing statistics
stats_path = self.output_dir / 'preprocessing_stats.json'
with open(stats_path, 'w') as f:
json.dump(self.stats, f, indent=2)
# Create summary
summary = {
'total_images': dataset_info['total_images'],
'successfully_processed': self.stats['total_processed'],
'quality_failed': self.stats['quality_failed'],
'processing_success_rate': self.stats['total_processed'] / dataset_info['total_images'],
'average_processing_time': np.mean(self.stats['processing_times']),
'average_quality_score': np.mean(self.stats['quality_scores']) if self.stats['quality_scores'] else 0,
'dataset_info': dataset_info
}
self.logger.info(f"Preprocessing completed!")
self.logger.info(f"Success rate: {summary['processing_success_rate']:.2%}")
self.logger.info(f"Average time per image: {summary['average_processing_time']:.3f}s")
self.logger.info(f"Average quality score: {summary['average_quality_score']:.3f}")
return {
'processed_splits': processed_splits,
'output_paths': output_paths,
'summary': summary,
'quality_metrics': self.stats,
'dataset_info': dataset_info
}
def save_preprocessed_data(self, filepath: str):
"""Save preprocessed data to file"""
with open(filepath, 'wb') as f:
pickle.dump({
'config': self.config,
'stats': self.stats,
'output_dir': str(self.output_dir)
}, f)
if __name__ == "__main__":
# Example usage
config = {
'target_size': [224, 224],
'normalization': 'z_score',
'augmentation': True,
'quality_check': True,
'clahe_enabled': True,
'quality_threshold': 0.3
}
preprocessor = MedicalImagePreprocessor(config, './preprocessing_output')
print("Medical Image Preprocessor ready for use!")