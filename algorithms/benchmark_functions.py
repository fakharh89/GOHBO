#!/usr/bin/env python3
import numpy as np
# Import the CEC implementation
from benchmark_functions_cec import CECBenchmarkSuite, BenchmarkFunction
# Main benchmark suite class - uses CEC implementation
class BenchmarkSuite(CECBenchmarkSuite):
def __init__(self, dimension: int = 30):
"""
Initialize CEC benchmark suite
Args:
dimension: Legacy parameter for compatibility (individual functions use their specified dimensions)
"""
super().__init__()
self.dimension = dimension
# Print information about the updated suite
if hasattr(self, '_print_info') and not self._print_info:
return
print("Loaded CEC Benchmark Functions Suite:")
print("- F1-F10: Basic functions (optimum: 100-1000)") print("- F11-F20: Hybrid functions (optimum: 1100-2000)")
print("- F21-F30: Composition functions (optimum: 2100-3000)")
print("- Exact dimensions: F1-F10 mostly D=30, F11-F20 mixed, F21-F30 varied")
# Prevent repeated printing
self._print_info = False
def main():
"""Test the CEC benchmark functions"""
print("CEC Benchmark Functions Test")
print("=" * 50)
# Create suite
suite = BenchmarkSuite()
# Test a few functions
test_functions = [1, 5, 10, 15, 20, 25, 30]
for fid in test_functions:
func = suite.get_function(fid)
if func:
# Create test point dim = func.dimension
test_x = np.zeros(dim) # Test at zero point
try:
result = func.function(test_x)
print(f"F{fid:2d}: {func.name:25s} | Dim: {dim:2d} | f(0): {result:8.2f} | Optimum: {func.global_optimum:7.1f}")
except Exception as e:
print(f"F{fid:2d}: {func.name:25s} | Dim: {dim:2d} | ERROR: {e}")
print("\nFunction List:")
suite.list_functions()
if __name__ == "__main__":
main()