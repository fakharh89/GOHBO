"""
Step 2: CNN Model Training
==========================
This module handles CNN model creation, training, and initial evaluation for
medical image classification tasks.
Features:
- Custom CNN architectures for medical images
- Transfer learning with pre-trained models
- Medical-specific training configurations
- Early stopping and model checkpointing
- Comprehensive training metrics tracking
Author: Medical AI Research Team
Version: 1.0.0
"""
import numpy as np
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')
# Deep learning framework imports with fallbacks
try:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50, VGG16, DenseNet121
TF_AVAILABLE = True
except ImportError:
TF_AVAILABLE = False
print("Warning: TensorFlow not available. Model training will be limited.")
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
class CNNModelTrainer:
"""CNN model trainer for medical image classification"""
def __init__(self, model_config: Dict[str, Any], training_config: Dict[str, Any], output_dir: str):
"""
Initialize CNN model trainer
Args:
model_config: Model architecture configuration
training_config: Training parameters configuration
output_dir: Directory to save training outputs
"""
self.model_config = model_config
self.training_config = training_config
self.output_dir = Path(output_dir)
self.output_dir.mkdir(parents=True, exist_ok=True)
# Setup logging
self.logger = self._setup_logger()
# Model parameters
self.input_shape = tuple(model_config.get('input_shape', [224, 224, 3]))
self.num_classes = model_config.get('num_classes', 2)
self.architecture = model_config.get('architecture', 'custom_cnn')
# Training parameters
self.epochs = training_config.get('epochs', 100)
self.batch_size = training_config.get('batch_size', 32)
self.learning_rate = training_config.get('learning_rate', 0.001)
self.early_stopping = training_config.get('early_stopping', True)
self.patience = training_config.get('patience', 10)
# Model storage
self.model = None
self.training_history = None
self.logger.info(" CNN Model Trainer initialized")
self.logger.info(f" Input shape: {self.input_shape}")
self.logger.info(f"️ Number of classes: {self.num_classes}")
self.logger.info(f"️ Architecture: {self.architecture}")
# Setup GPU if available
if TF_AVAILABLE:
self._setup_gpu()
def _setup_logger(self) -> logging.Logger:
"""Setup logging for model training step"""
logger = logging.getLogger('model_training_step')
logger.setLevel(logging.INFO)
# Remove existing handlers
for handler in logger.handlers[:]:
logger.removeHandler(handler)
# Create handler
log_file = self.output_dir / 'model_training.log'
handler = logging.FileHandler(log_file)
# Create formatter
formatter = logging.Formatter(
'%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
return logger
def _setup_gpu(self):
"""Setup GPU configuration for TensorFlow"""
if not TF_AVAILABLE:
return
# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
try:
for gpu in gpus:
tf.config.experimental.set_memory_growth(gpu, True)
self.logger.info(f"GPU setup completed: {len(gpus)} GPU(s) available")
except RuntimeError as e:
self.logger.warning(f"GPU setup failed: {e}")
else:
self.logger.info("No GPU found, using CPU")
def create_custom_cnn(self) -> keras.Model:
"""
Create custom CNN architecture for medical images
Returns:
Compiled Keras model
"""
# Model parameters from config
conv_blocks = self.model_config.get('conv_blocks', 4)
base_filters = self.model_config.get('base_filters', 64)
filter_multiplier = self.model_config.get('filter_multiplier', 2.0)
dropout_rate = self.model_config.get('dropout_rate', 0.3)
l2_reg = self.model_config.get('l2_regularization', 1e-4)
model = models.Sequential()
model.add(layers.Input(shape=self.input_shape))
# Convolutional blocks
current_filters = base_filters
for block in range(conv_blocks):
# Convolutional layers
model.add(layers.Conv2D(
filters=int(current_filters),
kernel_size=(3, 3),
activation='relu',
padding='same',
kernel_regularizer=keras.regularizers.l2(l2_reg),
name=f'conv_{block}_1'
))
model.add(layers.Conv2D(
filters=int(current_filters),
kernel_size=(3, 3),
activation='relu',
padding='same',
kernel_regularizer=keras.regularizers.l2(l2_reg),
name=f'conv_{block}_2'
))
# Batch normalization
model.add(layers.BatchNormalization(name=f'bn_{block}'))
# Max pooling
model.add(layers.MaxPooling2D(
pool_size=(2, 2),
name=f'maxpool_{block}'
))
# Dropout for regularization
model.add(layers.Dropout(dropout_rate, name=f'dropout_{block}'))
# Increase filters for next block
current_filters *= filter_multiplier
# Global average pooling
model.add(layers.GlobalAveragePooling2D(name='global_avg_pool'))
# Dense layers
model.add(layers.Dense(
256,
activation='relu',
kernel_regularizer=keras.regularizers.l2(l2_reg),
name='dense_1'
))
model.add(layers.Dropout(dropout_rate, name='dropout_final'))
# Output layer
if self.num_classes == 2:
# Binary classification
model.add(layers.Dense(1, activation='sigmoid', name='output'))
else:
# Multi-class classification
model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
self.logger.info(f"Custom CNN created with {conv_blocks} blocks")
return model
def create_transfer_learning_model(self, base_model_name: str = 'resnet50') -> keras.Model:
"""
Create transfer learning model
Args:
base_model_name: Name of the base model ('resnet50', 'vgg16', 'densenet121')
Returns:
Compiled Keras model
"""
# Load pre-trained base model
if base_model_name.lower() == 'resnet50':
base_model = ResNet50(
weights='imagenet',
include_top=False,
input_shape=self.input_shape
)
elif base_model_name.lower() == 'vgg16':
base_model = VGG16(
weights='imagenet',
include_top=False,
input_shape=self.input_shape
)
elif base_model_name.lower() == 'densenet121':
base_model = DenseNet121(
weights='imagenet',
include_top=False,
input_shape=self.input_shape
)
else:
raise ValueError(f"Unsupported base model: {base_model_name}")
# Freeze base model layers initially
base_model.trainable = False
# Add custom top layers
dropout_rate = self.model_config.get('dropout_rate', 0.3)
l2_reg = self.model_config.get('l2_regularization', 1e-4)
model = models.Sequential([
base_model,
layers.GlobalAveragePooling2D(name='global_avg_pool'),
layers.Dense(
512,
activation='relu',
kernel_regularizer=keras.regularizers.l2(l2_reg),
name='dense_1'
),
layers.Dropout(dropout_rate, name='dropout_1'),
layers.Dense(
256,
activation='relu',
kernel_regularizer=keras.regularizers.l2(l2_reg),
name='dense_2'
),
layers.Dropout(dropout_rate, name='dropout_2'),
])
# Output layer
if self.num_classes == 2:
model.add(layers.Dense(1, activation='sigmoid', name='output'))
else:
model.add(layers.Dense(self.num_classes, activation='softmax', name='output'))
self.logger.info(f"Transfer learning model created with {base_model_name}")
return model
def compile_model(self, model: keras.Model) -> keras.Model:
"""
Compile the model with appropriate loss and metrics
Args:
model: Keras model to compile
Returns:
Compiled model
"""
# Choose optimizer
optimizer_name = self.training_config.get('optimizer', 'adam')
if optimizer_name.lower() == 'adam':
optimizer = optimizers.Adam(learning_rate=self.learning_rate)
elif optimizer_name.lower() == 'sgd':
optimizer = optimizers.SGD(
learning_rate=self.learning_rate,
momentum=0.9,
nesterov=True
)
elif optimizer_name.lower() == 'rmsprop':
optimizer = optimizers.RMSprop(learning_rate=self.learning_rate)
else:
optimizer = optimizers.Adam(learning_rate=self.learning_rate)
# Choose loss function
if self.num_classes == 2:
loss = 'binary_crossentropy'
metrics = ['accuracy', 'precision', 'recall']
else:
loss = 'sparse_categorical_crossentropy'
metrics = ['accuracy', 'sparse_categorical_crossentropy']
# Compile model
model.compile(
optimizer=optimizer,
loss=loss,
metrics=metrics
)
self.logger.info(f"Model compiled with {optimizer_name} optimizer")
return model
def create_callbacks(self) -> List[keras.callbacks.Callback]:
"""
Create training callbacks
Returns:
List of Keras callbacks
"""
callbacks_list = []
# Model checkpoint
checkpoint_path = self.output_dir / 'best_model.h5'
checkpoint_callback = callbacks.ModelCheckpoint(
filepath=str(checkpoint_path),
monitor='val_accuracy',
mode='max',
save_best_only=True,
save_weights_only=False,
verbose=1
)
callbacks_list.append(checkpoint_callback)
# Early stopping
if self.early_stopping:
early_stop_callback = callbacks.EarlyStopping(
monitor='val_loss',
mode='min',
patience=self.patience,
restore_best_weights=True,
verbose=1
)
callbacks_list.append(early_stop_callback)
# Reduce learning rate on plateau
if self.training_config.get('reduce_lr_on_plateau', True):
reduce_lr_callback = callbacks.ReduceLROnPlateau(
monitor='val_loss',
mode='min',
factor=0.5,
patience=self.patience // 2,
min_lr=1e-7,
verbose=1
)
callbacks_list.append(reduce_lr_callback)
# CSV logger
csv_logger_path = self.output_dir / 'training_log.csv'
csv_logger = callbacks.CSVLogger(str(csv_logger_path), append=False)
callbacks_list.append(csv_logger)
self.logger.info(f"Created {len(callbacks_list)} training callbacks")
return callbacks_list
def calculate_class_weights(self, labels: np.ndarray) -> Dict[int, float]:
"""
Calculate class weights for handling imbalanced datasets
Args:
labels: Array of class labels
Returns:
Dictionary of class weights
"""
use_class_weights = self.training_config.get('class_weights', None)
if use_class_weights == 'balanced':
classes = np.unique(labels)
class_weights = compute_class_weight(
class_weight='balanced',
classes=classes,
y=labels
)
class_weight_dict = {int(cls): float(weight) for cls, weight in zip(classes, class_weights)}
self.logger.info(f"Class weights calculated: {class_weight_dict}")
return class_weight_dict
return None
def load_preprocessed_data(self, data_paths: Dict[str, Dict[str, str]]) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
"""
Load preprocessed data from Step 1
Args:
data_paths: Dictionary containing paths to preprocessed data
Returns:
Tuple of (images_dict, labels_dict)
"""
images_dict = {}
labels_dict = {}
for split_name, paths in data_paths.items():
if split_name in ['train', 'validation', 'test']:
images = np.load(paths['images'])
labels = np.load(paths['labels'])
images_dict[split_name] = images
labels_dict[split_name] = labels
self.logger.info(f" Loaded {split_name}: {images.shape[0]} samples")
return images_dict, labels_dict
def train_baseline_model(self, data_paths: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
"""
Train baseline model on preprocessed data
Args:
data_paths: Dictionary containing paths to preprocessed data
Returns:
Dictionary containing training results
"""
if not TF_AVAILABLE:
raise RuntimeError("TensorFlow is required for model training")
self.logger.info("Starting baseline model training")
# Load data
images_dict, labels_dict = self.load_preprocessed_data(data_paths)
# Create model
if self.architecture == 'custom_cnn':
model = self.create_custom_cnn()
elif self.architecture.startswith('transfer_'):
base_model_name = self.architecture.split('_', 1)[1]
model = self.create_transfer_learning_model(base_model_name)
else:
model = self.create_custom_cnn()
# Compile model
model = self.compile_model(model)
# Print model summary
model.summary()
# Calculate class weights
class_weights = self.calculate_class_weights(labels_dict['train'])
# Create callbacks
callbacks_list = self.create_callbacks()
# Prepare validation data
validation_data = None
if 'validation' in images_dict:
validation_data = (images_dict['validation'], labels_dict['validation'])
# Train model
start_time = time.time()
history = model.fit(
images_dict['train'],
labels_dict['train'],
batch_size=self.batch_size,
epochs=self.epochs,
validation_data=validation_data,
class_weight=class_weights,
callbacks=callbacks_list,
verbose=1
)
training_time = time.time() - start_time
# Store model and history
self.model = model
self.training_history = history
# Evaluate on validation set
validation_metrics = {}
if 'validation' in images_dict:
val_predictions = model.predict(images_dict['validation'])
if self.num_classes == 2:
val_pred_classes = (val_predictions > 0.5).astype(int).flatten()
else:
val_pred_classes = np.argmax(val_predictions, axis=1)
val_report = classification_report(
labels_dict['validation'],
val_pred_classes,
output_dict=True
)
validation_metrics = {
'accuracy': val_report['accuracy'],
'precision': val_report['weighted avg']['precision'],
'recall': val_report['weighted avg']['recall'],
'f1_score': val_report['weighted avg']['f1-score']
}
# Save training results
results = {
'training_history': {
'loss': [float(x) for x in history.history['loss']],
'accuracy': [float(x) for x in history.history.get('accuracy', [])],
'val_loss': [float(x) for x in history.history.get('val_loss', [])],
'val_accuracy': [float(x) for x in history.history.get('val_accuracy', [])]
},
'validation_metrics': validation_metrics,
'training_time': training_time,
'total_epochs': len(history.history['loss']),
'model_summary': self._get_model_summary(model),
'model_config': self.model_config,
'training_config': self.training_config
}
# Save results to file
results_path = self.output_dir / 'training_results.json'
with open(results_path, 'w') as f:
json.dump(results, f, indent=2)
self.logger.info(f"Training completed in {training_time:.2f} seconds")
self.logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
if validation_metrics:
self.logger.info(f"Final validation accuracy: {validation_metrics['accuracy']:.4f}")
return results
def _get_model_summary(self, model: keras.Model) -> str:
"""Get string representation of model summary"""
from io import StringIO
import sys
# Capture model summary
old_stdout = sys.stdout
sys.stdout = buffer = StringIO()
model.summary()
sys.stdout = old_stdout
return buffer.getvalue()
def save_model(self, filepath: str):
"""
Save trained model
Args:
filepath: Path to save the model
"""
if self.model is None:
raise ValueError("No model to save. Train model first.")
# Save full model
self.model.save(filepath)
# Also save as pickle for consistency
pickle_path = str(filepath).replace('.h5', '.pkl')
with open(pickle_path, 'wb') as f:
pickle.dump({
'model_config': self.model_config,
'training_config': self.training_config,
'model_path': filepath
}, f)
self.logger.info(f" Model saved to {filepath}")
def load_model(self, filepath: str) -> keras.Model:
"""
Load trained model
Args:
filepath: Path to the saved model
Returns:
Loaded Keras model
"""
model = keras.models.load_model(filepath)
self.model = model
self.logger.info(f" Model loaded from {filepath}")
return model
def predict(self, images: np.ndarray) -> np.ndarray:
"""
Make predictions on new images
Args:
images: Array of images to predict
Returns:
Array of predictions
"""
if self.model is None:
raise ValueError("No model loaded. Train or load a model first.")
predictions = self.model.predict(images)
self.logger.info(f" Made predictions for {len(images)} images")
return predictions
if __name__ == "__main__":
# Example usage
model_config = {
'architecture': 'custom_cnn',
'input_shape': [224, 224, 3],
'num_classes': 2,
'conv_blocks': 4,
'base_filters': 64,
'dropout_rate': 0.3
}
training_config = {
'epochs': 50,
'batch_size': 32,
'learning_rate': 0.001,
'early_stopping': True,
'patience': 10
}
trainer = CNNModelTrainer(model_config, training_config, './model_training_output')
print(" CNN Model Trainer ready for use!")