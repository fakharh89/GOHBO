#!/usr/bin/env python3
import numpy as np
import time
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')
try:
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50 # Will modify to ResNet18
from tensorflow.keras.preprocessing.image import ImageDataGenerator
TF_AVAILABLE = True
print(f"TensorFlow {tf.__version__} available - Full GOHBORESNET18 functionality enabled")
except ImportError as e:
TF_AVAILABLE = False
print(f"ERROR: TensorFlow not available: {e}. GOHBORESNET18 will use simulation mode.")
try:
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
SKLEARN_AVAILABLE = True
except ImportError:
SKLEARN_AVAILABLE = False
print("Scikit-learn not available. Using simplified metrics.")
from gohbo_algorithm import GOHBO
@dataclass
class ResNet18Config:
"""Configuration for ResNet-18 architecture"""
input_shape: Tuple[int, int, int] = (224, 224, 3)
num_classes: int = 2
dropout_rate: float = 0.5
use_pretrained: bool = False
freeze_base: bool = False
activation: str = 'relu'
final_activation: str = 'softmax'
@dataclass
class TrainingConfig:
"""Configuration for model training"""
batch_size: int = 32
epochs: int = 100
learning_rate: float = 0.001
optimizer: str = 'adam'
loss_function: str = 'categorical_crossentropy'
metrics: List[str] = None
early_stopping: bool = True
patience: int = 10
reduce_lr: bool = True
validation_split: float = 0.2
def __post_init__(self):
if self.metrics is None:
self.metrics = ['accuracy']
@dataclass
class GOHBOConfig:
"""Configuration for GOHBO optimization"""
population_size: int = 30
max_iterations: int = 50
gwo_weight: float = 0.4
hbo_weight: float = 0.4
ol_weight: float = 0.2
ol_frequency: int = 10
# Hyperparameter bounds for optimization
param_bounds: Dict = None
def __post_init__(self):
if self.param_bounds is None:
self.param_bounds = {
'learning_rate': (1e-5, 1e-1),
'batch_size': (8, 128),
'dropout_rate': (0.0, 0.8),
'epochs': (20, 200)
}
class ResNet18Builder:
"""Builder class for ResNet-18 architecture"""
@staticmethod
def residual_block(x, filters: int, kernel_size: int = 3, stride: int = 1, activation: str = 'relu', use_bias: bool = False):
"""Create a residual block"""
# Shortcut connection
shortcut = x
# First convolution
x = layers.Conv2D(filters, kernel_size, strides=stride, padding='same', use_bias=use_bias)(x)
x = layers.BatchNormalization()(x)
x = layers.Activation(activation)(x)
# Second convolution
x = layers.Conv2D(filters, kernel_size, strides=1, padding='same', use_bias=use_bias)(x)
x = layers.BatchNormalization()(x)
# Adjust shortcut if needed
if stride != 1 or shortcut.shape[-1] != filters:
shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same', use_bias=use_bias)(shortcut)
shortcut = layers.BatchNormalization()(shortcut)
# Add shortcut and apply activation
x = layers.Add()([x, shortcut])
x = layers.Activation(activation)(x)
return x
@staticmethod
def build_resnet18(config: ResNet18Config) -> tf.keras.Model:
"""Build ResNet-18 model"""
if not TF_AVAILABLE:
raise ImportError("TensorFlow is required for ResNet-18 model")
inputs = layers.Input(shape=config.input_shape)
# Initial convolution
x = layers.Conv2D(64, 7, strides=2, padding='same', use_bias=False)(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation(config.activation)(x)
x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
# Residual blocks
# Layer 1: 2 blocks, 64 filters
x = ResNet18Builder.residual_block(x, 64, activation=config.activation)
x = ResNet18Builder.residual_block(x, 64, activation=config.activation)
# Layer 2: 2 blocks, 128 filters
x = ResNet18Builder.residual_block(x, 128, stride=2, activation=config.activation)
x = ResNet18Builder.residual_block(x, 128, activation=config.activation)
# Layer 3: 2 blocks, 256 filters
x = ResNet18Builder.residual_block(x, 256, stride=2, activation=config.activation)
x = ResNet18Builder.residual_block(x, 256, activation=config.activation)
# Layer 4: 2 blocks, 512 filters
x = ResNet18Builder.residual_block(x, 512, stride=2, activation=config.activation)
x = ResNet18Builder.residual_block(x, 512, activation=config.activation)
# Global average pooling and classification head
x = layers.GlobalAveragePooling2D()(x)
if config.dropout_rate > 0:
x = layers.Dropout(config.dropout_rate)(x)
outputs = layers.Dense(config.num_classes, activation=config.final_activation)(x)
model = tf.keras.Model(inputs, outputs, name='ResNet18')
return model
class GOHBORESNET18:
"""
GOHBORESNET18: Integrated GOHBO + ResNet-18 Model
Combines GOHBO optimization with ResNet-18 architecture for optimal
medical image classification performance.
"""
def __init__(self,
resnet_config: ResNet18Config = None,
training_config: TrainingConfig = None,
gohbo_config: GOHBOConfig = None,
output_dir: str = "./gohboresnet18_results",
verbose: bool = True):
"""
Initialize GOHBORESNET18
Args:
resnet_config: ResNet-18 model configuration
training_config: Training parameters
gohbo_config: GOHBO optimization configuration
output_dir: Directory for saving results
verbose: Enable detailed logging
"""
self.resnet_config = resnet_config or ResNet18Config()
self.training_config = training_config or TrainingConfig()
self.gohbo_config = gohbo_config or GOHBOConfig()
self.output_dir = Path(output_dir)
self.verbose = verbose
# Create output directory
self.output_dir.mkdir(parents=True, exist_ok=True)
# Setup logging
if verbose:
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger(__name__)
else:
self.logger = logging.getLogger(__name__)
self.logger.addHandler(logging.NullHandler())
# Model and optimization state
self.best_model = None
self.best_hyperparameters = None
self.best_accuracy = 0.0
self.optimization_history = []
# Check availability of dependencies
if not TF_AVAILABLE:
self.logger.warning("TensorFlow not available. Using simulation mode.")
self.logger.info("GOHBORESNET18 initialized")
self.logger.info(f"Output directory: {self.output_dir}")
def _encode_hyperparameters(self, params: Dict) -> np.ndarray:
"""Encode hyperparameters to GOHBO solution vector"""
encoded = []
param_order = ['learning_rate', 'batch_size', 'dropout_rate', 'epochs']
for param_name in param_order:
if param_name in params:
value = params[param_name]
bounds = self.gohbo_config.param_bounds[param_name]
# Normalize to [0, 1] for GOHBO
if param_name in ['batch_size', 'epochs']:
# Integer parameters - use log scale
log_min, log_max = np.log(bounds[0]), np.log(bounds[1])
normalized = (np.log(value) - log_min) / (log_max - log_min)
else:
# Float parameters - linear scale for learning_rate, dropout_rate
if param_name == 'learning_rate':
# Log scale for learning rate
log_min, log_max = np.log(bounds[0]), np.log(bounds[1])
normalized = (np.log(value) - log_min) / (log_max - log_min)
else:
# Linear scale for dropout_rate
normalized = (value - bounds[0]) / (bounds[1] - bounds[0])
encoded.append(np.clip(normalized, 0, 1))
return np.array(encoded)
def _decode_hyperparameters(self, solution: np.ndarray) -> Dict:
"""Decode GOHBO solution vector to hyperparameters"""
params = {}
param_order = ['learning_rate', 'batch_size', 'dropout_rate', 'epochs']
for i, param_name in enumerate(param_order):
if i < len(solution):
normalized_value = np.clip(solution[i], 0, 1)
bounds = self.gohbo_config.param_bounds[param_name]
if param_name in ['batch_size', 'epochs']:
# Integer parameters with log scale
log_min, log_max = np.log(bounds[0]), np.log(bounds[1])
value = np.exp(log_min + normalized_value * (log_max - log_min))
params[param_name] = int(round(value))
else:
# Float parameters
if param_name == 'learning_rate':
# Log scale for learning rate
log_min, log_max = np.log(bounds[0]), np.log(bounds[1])
params[param_name] = np.exp(log_min + normalized_value * (log_max - log_min))
else:
# Linear scale for dropout_rate
params[param_name] = bounds[0] + normalized_value * (bounds[1] - bounds[0])
return params
def _build_model(self, hyperparameters: Dict) -> tf.keras.Model:
"""Build ResNet-18 model with given hyperparameters"""
if not TF_AVAILABLE:
# Return mock model for simulation
return None
# Update ResNet config with hyperparameters
config = ResNet18Config(
input_shape=self.resnet_config.input_shape,
num_classes=self.resnet_config.num_classes,
dropout_rate=hyperparameters.get('dropout_rate', self.resnet_config.dropout_rate),
use_pretrained=self.resnet_config.use_pretrained,
freeze_base=self.resnet_config.freeze_base,
activation=self.resnet_config.activation,
final_activation=self.resnet_config.final_activation
)
# Build model
model = ResNet18Builder.build_resnet18(config)
# Compile model
optimizer_map = {
'adam': optimizers.Adam(learning_rate=hyperparameters.get('learning_rate', 0.001)),
'sgd': optimizers.SGD(learning_rate=hyperparameters.get('learning_rate', 0.001)),
'rmsprop': optimizers.RMSprop(learning_rate=hyperparameters.get('learning_rate', 0.001))
}
optimizer = optimizer_map.get(self.training_config.optimizer, optimizers.Adam(learning_rate=hyperparameters.get('learning_rate', 0.001)))
model.compile(
optimizer=optimizer,
loss=self.training_config.loss_function,
metrics=self.training_config.metrics
)
return model
def _evaluate_model(self, hyperparameters: Dict, X_train: np.ndarray, y_train: np.ndarray,
X_val: np.ndarray, y_val: np.ndarray) -> float:
"""Evaluate model with given hyperparameters"""
if not TF_AVAILABLE:
# Simulation mode - return random accuracy with bias towards better hyperparameters
base_accuracy = 0.7
lr_bonus = max(0, 0.1 * (1 - abs(np.log10(hyperparameters.get('learning_rate', 0.001)) + 3) / 3))
dropout_bonus = max(0, 0.05 * (1 - abs(hyperparameters.get('dropout_rate', 0.3) - 0.3) / 0.3))
random_noise = np.random.normal(0, 0.05)
simulated_accuracy = base_accuracy + lr_bonus + dropout_bonus + random_noise
return np.clip(simulated_accuracy, 0.5, 0.95)
try:
# Build model
model = self._build_model(hyperparameters)
# Setup callbacks
callbacks_list = []
if self.training_config.early_stopping:
early_stopping = callbacks.EarlyStopping(
monitor='val_accuracy',
patience=self.training_config.patience,
restore_best_weights=True,
verbose=0
)
callbacks_list.append(early_stopping)
if self.training_config.reduce_lr:
reduce_lr = callbacks.ReduceLROnPlateau(
monitor='val_accuracy',
factor=0.5,
patience=5,
verbose=0
)
callbacks_list.append(reduce_lr)
# Train model
history = model.fit(
X_train, y_train,
batch_size=hyperparameters.get('batch_size', self.training_config.batch_size),
epochs=min(hyperparameters.get('epochs', self.training_config.epochs), 50), # Limit for optimization
validation_data=(X_val, y_val),
callbacks=callbacks_list,
verbose=0
)
# Evaluate on validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val, verbose=0)
return val_accuracy
except Exception as e:
self.logger.warning(f"Error evaluating model: {str(e)}")
return 0.0 # Return low fitness for failed evaluations
def _objective_function(self, solution: np.ndarray) -> float:
"""Objective function for GOHBO optimization (to minimize)"""
# Decode hyperparameters
hyperparameters = self._decode_hyperparameters(solution)
# Evaluate model
accuracy = self._evaluate_model(
hyperparameters, self.X_train, self.y_train,
self.X_val, self.y_val
)
# Store optimization history
self.optimization_history.append({
'hyperparameters': hyperparameters.copy(),
'accuracy': accuracy,
'fitness': 1.0 - accuracy # Convert accuracy to fitness (minimize)
})
# Update best solution
if accuracy > self.best_accuracy:
self.best_accuracy = accuracy
self.best_hyperparameters = hyperparameters.copy()
self.logger.info(f"New best accuracy: {accuracy:.4f}")
self.logger.info(f"Best hyperparameters: {hyperparameters}")
# Return fitness (to minimize)
return 1.0 - accuracy
def optimize_hyperparameters(self,
X_train: np.ndarray, y_train: np.ndarray,
X_val: np.ndarray, y_val: np.ndarray) -> Dict:
"""
Optimize hyperparameters using GOHBO
Args:
X_train: Training images
y_train: Training labels
X_val: Validation images y_val: Validation labels
Returns:
Dictionary with optimization results
"""
self.logger.info("Starting GOHBO hyperparameter optimization for ResNet-18")
# Store data for objective function
self.X_train = X_train
self.y_train = y_train
self.X_val = X_val
self.y_val = y_val
# Initialize optimization history
self.optimization_history = []
self.best_accuracy = 0.0
self.best_hyperparameters = None
# Setup GOHBO optimizer
dimension = len(self.gohbo_config.param_bounds)
bounds = (np.zeros(dimension), np.ones(dimension)) # Normalized bounds [0, 1]
gohbo = GOHBO(
objective_function=self._objective_function,
dimension=dimension,
bounds=bounds,
population_size=self.gohbo_config.population_size,
max_iterations=self.gohbo_config.max_iterations,
gwo_weight=self.gohbo_config.gwo_weight,
hbo_weight=self.gohbo_config.hbo_weight,
ol_weight=self.gohbo_config.ol_weight,
ol_frequency=self.gohbo_config.ol_frequency,
verbose=self.verbose
)
# Run optimization
start_time = time.time()
optimization_results = gohbo.optimize()
optimization_time = time.time() - start_time
# Process results
best_solution = optimization_results['best_position']
best_hyperparameters = self._decode_hyperparameters(best_solution)
self.logger.info(f"GOHBO optimization complete in {optimization_time:.2f} seconds")
self.logger.info(f"Best accuracy: {self.best_accuracy:.4f}")
self.logger.info(f"Optimal hyperparameters: {best_hyperparameters}")
# Save optimization results
results = {
'best_hyperparameters': best_hyperparameters,
'best_accuracy': self.best_accuracy,
'optimization_time': optimization_time,
'gohbo_results': optimization_results,
'optimization_history': self.optimization_history,
'algorithm': 'GOHBORESNET18'
}
# Save to file
results_file = self.output_dir / 'gohbo_optimization_results.json'
# Convert numpy arrays to lists for JSON serialization
serializable_results = {
'best_hyperparameters': best_hyperparameters,
'best_accuracy': float(self.best_accuracy),
'optimization_time': float(optimization_time),
'optimization_history': [
{
'hyperparameters': entry['hyperparameters'],
'accuracy': float(entry['accuracy']),
'fitness': float(entry['fitness'])
} for entry in self.optimization_history
],
'algorithm': 'GOHBORESNET18',
'config': {
'resnet_config': {
'input_shape': self.resnet_config.input_shape,
'num_classes': self.resnet_config.num_classes,
'dropout_rate': self.resnet_config.dropout_rate
},
'gohbo_config': {
'population_size': self.gohbo_config.population_size,
'max_iterations': self.gohbo_config.max_iterations,
'param_bounds': self.gohbo_config.param_bounds
}
}
}
with open(results_file, 'w') as f:
json.dump(serializable_results, f, indent=2)
self.logger.info(f"Results saved to: {results_file}")
return results
def train_final_model(self,
X_train: np.ndarray, y_train: np.ndarray,
X_val: np.ndarray, y_val: np.ndarray,
X_test: Optional[np.ndarray] = None,
y_test: Optional[np.ndarray] = None) -> Dict:
"""
Train final model with optimized hyperparameters
Args:
X_train: Training images
y_train: Training labels
X_val: Validation images
y_val: Validation labels
X_test: Test images (optional)
y_test: Test labels (optional)
Returns:
Dictionary with final model results
"""
if self.best_hyperparameters is None:
raise ValueError("No optimized hyperparameters found. Run optimize_hyperparameters first.")
self.logger.info("Training final GOHBORESNET18 model with optimized hyperparameters")
if not TF_AVAILABLE:
# Simulation mode
self.logger.info("TensorFlow not available. Using simulation mode.")
final_results = {
'final_accuracy': self.best_accuracy,
'test_accuracy': self.best_accuracy * np.random.uniform(0.95, 1.05),
'hyperparameters': self.best_hyperparameters,
'training_time': np.random.uniform(300, 600),
'model_saved': False,
'simulation_mode': True
}
return final_results
try:
# Build final model
model = self._build_model(self.best_hyperparameters)
# Setup callbacks for final training
callbacks_list = []
# Model checkpoint
checkpoint = callbacks.ModelCheckpoint(
self.output_dir / 'best_gohboresnet18_model.h5',
monitor='val_accuracy',
save_best_only=True,
verbose=1
)
callbacks_list.append(checkpoint)
if self.training_config.early_stopping:
early_stopping = callbacks.EarlyStopping(
monitor='val_accuracy',
patience=self.training_config.patience,
restore_best_weights=True,
verbose=1
)
callbacks_list.append(early_stopping)
# Train final model
start_time = time.time()
history = model.fit(
X_train, y_train,
batch_size=self.best_hyperparameters['batch_size'],
epochs=self.best_hyperparameters['epochs'],
validation_data=(X_val, y_val),
callbacks=callbacks_list,
verbose=1
)
training_time = time.time() - start_time
# Evaluate final model
final_val_loss, final_val_accuracy = model.evaluate(X_val, y_val, verbose=0)
results = {
'final_accuracy': final_val_accuracy,
'hyperparameters': self.best_hyperparameters,
'training_time': training_time,
'training_history': history.history,
'model_saved': True,
'simulation_mode': False
}
# Test evaluation if provided
if X_test is not None and y_test is not None:
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
results['test_accuracy'] = test_accuracy
# Generate predictions for detailed evaluation
if SKLEARN_AVAILABLE:
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1) if y_test.ndim > 1 else y_test
# Classification report
results['classification_report'] = classification_report(
y_test_classes, y_pred_classes, output_dict=True
)
# Confusion matrix
results['confusion_matrix'] = confusion_matrix(
y_test_classes, y_pred_classes
).tolist()
self.best_model = model
# Save results
results_file = self.output_dir / 'final_model_results.json'
# Convert numpy arrays to lists for JSON serialization
serializable_results = {}
for key, value in results.items():
if isinstance(value, (np.ndarray, np.floating, np.integer)):
serializable_results[key] = float(value) if np.isscalar(value) else value.tolist()
elif key == 'training_history':
# Convert history arrays to lists
serializable_results[key] = {
metric: [float(val) for val in values] for metric, values in value.items()
}
else:
serializable_results[key] = value
with open(results_file, 'w') as f:
json.dump(serializable_results, f, indent=2)
self.logger.info(f"Final model training complete")
self.logger.info(f"Final validation accuracy: {final_val_accuracy:.4f}")
if 'test_accuracy' in results:
self.logger.info(f"Final test accuracy: {results['test_accuracy']:.4f}")
self.logger.info(f"Results saved to: {results_file}")
return results
except Exception as e:
self.logger.error(f"Error training final model: {str(e)}")
raise
# Example usage and testing functions
def create_synthetic_medical_data(num_samples: int = 1000, input_shape: Tuple = (224, 224, 3)) -> Tuple:
"""Create synthetic medical image data for testing"""
# Generate synthetic images
X = np.random.rand(num_samples, *input_shape).astype(np.float32)
# Generate synthetic labels (binary classification)
y = np.random.randint(0, 2, size=num_samples)
# Convert to categorical if needed
if TF_AVAILABLE:
y = tf.keras.utils.to_categorical(y, 2)
return X, y
def test_gohboresnet18():
"""Test GOHBORESNET18 implementation"""
print("Testing GOHBORESNET18 Implementation")
print("=" * 60)
# Create synthetic data
print("Creating synthetic medical data...")
X, y = create_synthetic_medical_data(num_samples=200, input_shape=(64, 64, 3))
# Split data
if SKLEARN_AVAILABLE:
from sklearn.model_selection import train_test_split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
else:
# Manual split
train_size = int(0.6 * len(X))
val_size = int(0.2 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_val, y_val = X[train_size:train_size+val_size], y[train_size:train_size+val_size]
X_test, y_test = X[train_size+val_size:], y[train_size+val_size:]
print(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
# Configure GOHBORESNET18 for quick testing
resnet_config = ResNet18Config(
input_shape=(64, 64, 3),
num_classes=2,
dropout_rate=0.3
)
training_config = TrainingConfig(
batch_size=16,
epochs=10, # Reduced for testing
learning_rate=0.001
)
gohbo_config = GOHBOConfig(
population_size=10, # Reduced for testing
max_iterations=5 # Reduced for testing
)
# Initialize GOHBORESNET18
gohboresnet18 = GOHBORESNET18(
resnet_config=resnet_config,
training_config=training_config,
gohbo_config=gohbo_config,
output_dir="./test_gohboresnet18",
verbose=True
)
print("Initialized GOHBORESNET18")
# Run hyperparameter optimization
print("Running GOHBO hyperparameter optimization...")
optimization_results = gohboresnet18.optimize_hyperparameters(
X_train, y_train, X_val, y_val
)
print(f"Optimization complete. Best accuracy: {optimization_results['best_accuracy']:.4f}")
# Train final model
print("Training final model with optimized hyperparameters...")
final_results = gohboresnet18.train_final_model(
X_train, y_train, X_val, y_val, X_test, y_test
)
print(f"Final model training complete.")
print(f"Final validation accuracy: {final_results['final_accuracy']:.4f}")
if 'test_accuracy' in final_results:
print(f"Final test accuracy: {final_results['test_accuracy']:.4f}")
print("GOHBORESNET18 test completed successfully!")
if __name__ == "__main__":
test_gohboresnet18()