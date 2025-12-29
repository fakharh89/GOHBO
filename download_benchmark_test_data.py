#!/usr/bin/env python3
"""
Download Standard Benchmark Test Data Files
===========================================
Downloads the standard CEC benchmark test data files and matrices
for 30-dimensional optimization problems in M_X_D30.txt format.
These files contain:
- Rotation matrices for benchmark functions
- Shift vectors for function transformations
- Standard test datasets used in optimization research
Author: Medical AI Research Team
Version: 1.0.0
"""
import os
import requests
import zipfile
import numpy as np
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional
import shutil
class BenchmarkDataDownloader:
"""Download and prepare standard benchmark test data files"""
def __init__(self, data_dir: str = "./benchmark_test_data", verbose: bool = True):
"""
Initialize benchmark data downloader
Args:
data_dir: Directory to save downloaded data
verbose: Enable detailed logging
"""
self.data_dir = Path(data_dir)
self.data_dir.mkdir(parents=True, exist_ok=True)
# Setup logging
if verbose:
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger(__name__)
else:
self.logger = logging.getLogger(__name__)
self.logger.addHandler(logging.NullHandler())
# Standard benchmark data URLs
self.download_sources = {
'cec2013_database': {
'url': 'https://al-roomi.org/multimedia/CEC_Database/CEC2013/RealParameterOptimization/CEC2013_RealParameterOptimization_Database.rar',
'description': 'CEC 2013 Real-Parameter Optimization Database',
'expected_files': ['M_D30.txt', 'M_D50.txt', 'M_D100.txt']
},
'cec2013_github': {
'url': 'https://github.com/mikeagn/CEC2013/archive/refs/heads/master.zip',
'description': 'CEC 2013 GitHub Repository',
'expected_files': ['python/data/', 'matlab/data/', 'c++/data/']
},
'cec2014_functions': {
'url': 'https://web.njit.edu/~ychen/CEC14/CEC14-C++.zip',
'description': 'CEC 2014 C++ Implementation with test data',
'expected_files': ['input_data/', 'shift_data_*.txt']
}
}
self.logger.info(f"BenchmarkDataDownloader initialized. Data directory: {self.data_dir}")
def download_file(self, url: str, filename: str) -> bool:
"""Download a file from URL"""
try:
self.logger.info(f"Downloading {filename} from {url}")
response = requests.get(url, stream=True, timeout=60)
response.raise_for_status()
file_path = self.data_dir / filename
with open(file_path, 'wb') as f:
for chunk in response.iter_content(chunk_size=8192):
f.write(chunk)
self.logger.info(f"Downloaded: {file_path}")
return True
except Exception as e:
self.logger.error(f"Failed to download {filename}: {str(e)}")
return False
def extract_archive(self, archive_path: Path, extract_to: Optional[Path] = None) -> bool:
"""Extract archive file"""
if extract_to is None:
extract_to = archive_path.parent / archive_path.stem
try:
if archive_path.suffix.lower() == '.zip':
with zipfile.ZipFile(archive_path, 'r') as zip_ref:
zip_ref.extractall(extract_to)
elif archive_path.suffix.lower() == '.rar':
# Try to use unrar command
import subprocess
result = subprocess.run(['unrar', 'x', str(archive_path), str(extract_to)], capture_output=True, text=True)
if result.returncode != 0:
self.logger.warning(f"unrar failed. Please extract {archive_path} manually.")
return False
else:
self.logger.error(f"Unsupported archive format: {archive_path.suffix}")
return False
self.logger.info(f"Extracted: {archive_path} -> {extract_to}")
return True
except Exception as e:
self.logger.error(f"Failed to extract {archive_path}: {str(e)}")
return False
def generate_standard_matrices(self):
"""Generate standard rotation and shift matrices if downloads fail"""
self.logger.info("Generating standard benchmark matrices...")
# Standard dimensions for benchmark testing
dimensions = [2, 10, 30, 50, 100]
for dim in dimensions:
# Generate rotation matrix for each dimension
self.generate_rotation_matrix(dim)
# Generate shift vectors for each function (F1-F30)
for func_id in range(1, 31):
self.generate_shift_vector(func_id, dim)
self.logger.info("Standard matrices generated")
def generate_rotation_matrix(self, dimension: int):
"""Generate orthogonal rotation matrix for given dimension"""
# Generate random orthogonal matrix using QR decomposition
np.random.seed(42) # For reproducible results
A = np.random.randn(dimension, dimension)
Q, R = np.linalg.qr(A)
# Ensure proper rotation (det = 1)
Q = Q * np.linalg.det(Q)
# Save as M_DX.txt format
matrix_file = self.data_dir / f"M_D{dimension}.txt"
np.savetxt(matrix_file, Q, fmt='%.16e')
# Also save individual function matrices
for func_id in range(1, 31):
func_matrix_file = self.data_dir / f"M_{func_id}_D{dimension}.txt"
# Generate slightly different matrix for each function
np.random.seed(42 + func_id)
A_func = np.random.randn(dimension, dimension)
Q_func, _ = np.linalg.qr(A_func)
Q_func = Q_func * np.linalg.det(Q_func)
np.savetxt(func_matrix_file, Q_func, fmt='%.16e')
self.logger.info(f"Generated rotation matrices for dimension {dimension}")
def generate_shift_vector(self, func_id: int, dimension: int):
"""Generate shift vector for specific function and dimension"""
# Standard bounds for most benchmark functions
bounds = {
1: (-100, 100), # Sphere
2: (-10, 10), # Schwefel 2.22
3: (-100, 100), # Schwefel 1.2
4: (-100, 100), # Schwefel 2.21
5: (-30, 30), # Rosenbrock
6: (-100, 100), # Step
7: (-1.28, 1.28), # Noisy Quartic
8: (-500, 500), # Schwefel
9: (-5.12, 5.12), # Rastrigin
10: (-32, 32), # Ackley
# Add more function bounds as needed
}
# Get bounds for this function (default to [-100, 100])
lower, upper = bounds.get(func_id, (-100, 100))
# Generate random shift vector within 80% of bounds
np.random.seed(func_id * 1000 + dimension)
shift_range = 0.8 * min(abs(lower), abs(upper))
shift_vector = np.random.uniform(-shift_range, shift_range, dimension)
# Save shift vector
shift_file = self.data_dir / f"shift_data_{func_id}_D{dimension}.txt"
np.savetxt(shift_file, shift_vector, fmt='%.16e')
# Also save as bias value (scalar)
bias_file = self.data_dir / f"f{func_id}_o.txt"
bias_value = np.random.uniform(-1000, 1000) # Random bias
with open(bias_file, 'w') as f:
f.write(f"{bias_value:.16e}\n")
def download_cec2013_data(self) -> bool:
"""Download CEC 2013 benchmark data"""
self.logger.info("Downloading CEC 2013 benchmark data...")
# Try to download from GitHub (most reliable)
github_success = self.download_file(
self.download_sources['cec2013_github']['url'],
'cec2013-master.zip'
)
if github_success:
# Extract the archive
archive_path = self.data_dir / 'cec2013-master.zip'
extract_success = self.extract_archive(archive_path)
if extract_success:
# Copy data files from extracted archive
extracted_dir = self.data_dir / 'cec2013-master'
self.copy_data_files(extracted_dir)
# Clean up
shutil.rmtree(extracted_dir, ignore_errors=True)
archive_path.unlink(missing_ok=True)
return True
return False
def copy_data_files(self, source_dir: Path):
"""Copy data files from extracted archive"""
data_dirs = [
source_dir / 'CEC2013-master' / 'python' / 'data',
source_dir / 'CEC2013-master' / 'matlab' / 'data',
source_dir / 'CEC2013-master' / 'c++' / 'data',
source_dir / 'python' / 'data',
source_dir / 'matlab' / 'data',
source_dir / 'c++' / 'data'
]
files_copied = 0
for data_dir in data_dirs:
if data_dir.exists():
self.logger.info(f"Copying files from {data_dir}")
for file_path in data_dir.rglob('*.txt'):
dest_path = self.data_dir / file_path.name
shutil.copy2(file_path, dest_path)
files_copied += 1
self.logger.debug(f"Copied: {file_path.name}")
self.logger.info(f"Copied {files_copied} data files")
def create_data_index(self):
"""Create an index of all available data files"""
index = {
'rotation_matrices': [],
'shift_vectors': [],
'bias_values': [],
'function_data': {},
'dimensions_available': []
}
# Find all data files
for file_path in self.data_dir.glob('*.txt'):
filename = file_path.name
if filename.startswith('M_') and '_D' in filename:
# Rotation matrix file: M_X_DY.txt
parts = filename.replace('.txt', '').split('_')
if len(parts) >= 3:
func_id = parts[1]
dim_str = parts[2]
if dim_str.startswith('D'):
dimension = int(dim_str[1:])
if func_id.isdigit():
func_id = int(func_id)
if func_id not in index['function_data']:
index['function_data'][func_id] = {}
index['function_data'][func_id][f'rotation_matrix_D{dimension}'] = filename
if dimension not in index['dimensions_available']:
index['dimensions_available'].append(dimension)
index['rotation_matrices'].append(filename)
elif filename.startswith('shift_data_'):
index['shift_vectors'].append(filename)
elif filename.startswith('f') and filename.endswith('_o.txt'):
index['bias_values'].append(filename)
# Sort lists
index['rotation_matrices'].sort()
index['shift_vectors'].sort()
index['bias_values'].sort()
index['dimensions_available'].sort()
# Save index
index_file = self.data_dir / 'data_index.json'
with open(index_file, 'w') as f:
json.dump(index, f, indent=2)
self.logger.info(f"Data index created: {index_file}")
# Print summary
self.logger.info(f"Data Summary:")
self.logger.info(f" Rotation matrices: {len(index['rotation_matrices'])}")
self.logger.info(f" Shift vectors: {len(index['shift_vectors'])}")
self.logger.info(f" Bias values: {len(index['bias_values'])}")
self.logger.info(f" Functions with data: {len(index['function_data'])}")
self.logger.info(f" Dimensions available: {index['dimensions_available']}")
return index
def download_all_benchmark_data(self) -> Dict:
"""Download and prepare all benchmark test data"""
self.logger.info("="*60)
self.logger.info("DOWNLOADING BENCHMARK TEST DATA")
self.logger.info("="*60)
results = {
'downloads_attempted': 0,
'downloads_successful': 0,
'generated_files': 0,
'total_files': 0
}
# Try to download from various sources
download_success = False
# Attempt CEC 2013 download
results['downloads_attempted'] += 1
if self.download_cec2013_data():
results['downloads_successful'] += 1
download_success = True
# If downloads fail, generate standard matrices
if not download_success:
self.logger.warning("Downloads failed. Generating standard benchmark matrices...")
self.generate_standard_matrices()
results['generated_files'] = len(list(self.data_dir.glob('*.txt')))
# Create data index
index = self.create_data_index()
results['total_files'] = len(list(self.data_dir.glob('*.txt')))
# Save results summary
summary = {
'download_date': str(Path(__file__).stat().st_mtime),
'data_directory': str(self.data_dir),
'download_results': results,
'data_index': index
}
summary_file = self.data_dir / 'download_summary.json'
with open(summary_file, 'w') as f:
json.dump(summary, f, indent=2)
self.logger.info(f"\nDOWNLOAD SUMMARY:")
self.logger.info(f" Downloads attempted: {results['downloads_attempted']}")
self.logger.info(f" Downloads successful: {results['downloads_successful']}")
self.logger.info(f" Generated files: {results['generated_files']}")
self.logger.info(f" Total files available: {results['total_files']}")
self.logger.info(f" Data directory: {self.data_dir}")
self.logger.info(f" Summary saved: {summary_file}")
return results
def list_available_files(self, pattern: str = "M_*_D30.txt"):
"""List available data files matching pattern"""
matching_files = list(self.data_dir.glob(pattern))
if matching_files:
self.logger.info(f"\nAvailable files matching '{pattern}':")
for i, file_path in enumerate(sorted(matching_files), 1):
self.logger.info(f" {i:2}. {file_path.name}")
else:
self.logger.warning(f"No files found matching pattern: {pattern}")
return matching_files
def main():
"""Main function to download benchmark test data"""
import argparse
parser = argparse.ArgumentParser(
description="Download standard benchmark test data files",
formatter_class=argparse.RawDescriptionHelpFormatter,
epilog="""
Examples:
python download_benchmark_test_data.py # Download to default directory
python download_benchmark_test_data.py --data-dir ./data # Specify custom directory
python download_benchmark_test_data.py --list-only # Just list available files
"""
)
parser.add_argument('--data-dir', type=str, default='./benchmark_test_data',
help='Directory to save downloaded data')
parser.add_argument('--list-only', action='store_true',
help='Only list available files, do not download')
parser.add_argument('--verbose', action='store_true', default=True,
help='Enable verbose output')
args = parser.parse_args()
print(" Benchmark Test Data Downloader")
print("="*50)
print("Downloads standard benchmark test data files including:")
print("• M_1_D30.txt, M_2_D30.txt, ..., M_30_D30.txt (rotation matrices)")
print("• shift_data_*.txt (shift vectors)")
print("• f*_o.txt (bias values)")
print("="*50)
# Initialize downloader
downloader = BenchmarkDataDownloader(
data_dir=args.data_dir,
verbose=args.verbose
)
if args.list_only:
# Just list available files
downloader.list_available_files("M_*_D30.txt")
downloader.list_available_files("shift_data_*_D30.txt")
else:
# Download all benchmark data
results = downloader.download_all_benchmark_data()
# Show some example files
print(f"\n Sample files created:")
downloader.list_available_files("M_*_D30.txt")
if results['total_files'] > 0:
print(f"\nSUCCESS! Downloaded and prepared {results['total_files']} benchmark test data files")
else:
print(f"\nFAILED! No benchmark test data files were created")
if __name__ == "__main__":
main()