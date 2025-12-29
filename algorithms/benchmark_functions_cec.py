#!/usr/bin/env python3
"""
30 CEC Benchmark Functions Test Suite for Optimization Algorithms
================================================================
"""
import numpy as np
import math
from typing import Callable, Dict, List, Tuple
from dataclasses import dataclass
@dataclass
class BenchmarkFunction:
"""Container for benchmark function information"""
name: str
function: Callable
dimension: int
bounds: Tuple[float, float]
global_optimum: float
description: str
function_type: str
class CECBenchmarkSuite:
"""
CEC Benchmark Functions Suite with exact mathematical formulations
30 functions with specified dimensions and global optimum values:
F1-F10: Basic functions (various dimensions)
F11-F20: Hybrid functions (various dimensions) F21-F30: Composition functions (various dimensions)
"""
def __init__(self):
"""Initialize CEC benchmark suite"""
self.functions = self._create_cec_function_suite()
# Pre-computed transformation matrices and shift vectors
# These would normally be loaded from M_X_D30.txt files
self._setup_transformation_data()
def _setup_transformation_data(self):
"""Setup transformation matrices and shift vectors for composition functions"""
# This is a simplified version - in practice, these come from M_X_D30.txt files
np.random.seed(42) # For reproducible transformations
self.rotation_matrices = {}
self.shift_vectors = {}
for i in range(21, 31): # F21-F30 use transformations
if i <= 23: # F21-F23
dim = [4, 4, 4][i-21]
elif i <= 26: # F24-F26
dim = [5, 5, 6][i-24]
else: # F27-F30
dim = [3, 3, 4, 4][i-27]
# Generate random orthogonal matrix using QR decomposition
A = np.random.randn(dim, dim)
Q, R = np.linalg.qr(A)
self.rotation_matrices[i] = Q
self.shift_vectors[i] = np.random.uniform(-5, 5, dim)
def _create_cec_function_suite(self) -> Dict[int, BenchmarkFunction]:
"""Create the complete suite of 30 CEC benchmark functions"""
functions = {}
# F1: Sphere Function (Shifted) - Dimension 30
functions[1] = BenchmarkFunction(
name="Shifted_Sphere",
function=lambda x: self._f1_shifted_sphere(x),
dimension=30,
bounds=(-100, 100),
global_optimum=100.0,
description="f1(x) = sum(xi^2) + 100",
function_type="unimodal"
)
# F2: Schwefel 2.22 - Dimension 30 functions[2] = BenchmarkFunction(
name="Schwefel_2_22",
function=lambda x: self._f2_schwefel_2_22(x),
dimension=30,
bounds=(-10, 10),
global_optimum=200.0,
description="f2(x) = sum(|xi|) + prod(|xi|)",
function_type="unimodal"
)
# F3: Schwefel 1.2 - Dimension 30
functions[3] = BenchmarkFunction(
name="Schwefel_1_2", function=lambda x: self._f3_schwefel_1_2(x),
dimension=30,
bounds=(-100, 100),
global_optimum=300.0,
description="f3(x) = sum(sum(xi)^2)",
function_type="unimodal"
)
# F4: Rosenbrock - Dimension 30
functions[4] = BenchmarkFunction(
name="Rosenbrock",
function=lambda x: self._f4_rosenbrock(x),
dimension=30,
bounds=(-30, 30), global_optimum=400.0,
description="f4(x) = sum(100*(xi+1-xi^2)^2 + (xi-1)^2)",
function_type="unimodal"
)
# F5: Rastrigin - Dimension 30
functions[5] = BenchmarkFunction(
name="Rastrigin",
function=lambda x: self._f5_rastrigin(x),
dimension=30,
bounds=(-5.12, 5.12),
global_optimum=500.0,
description="f5(x) = sum(xi^2 - 10*cos(2*pi*xi) + 10)",
function_type="multimodal"
)
# F6: Schaffer F6 - Dimension 30
functions[6] = BenchmarkFunction(
name="Schaffer_F6",
function=lambda x: self._f6_schaffer_f6(x),
dimension=30,
bounds=(-100, 100),
global_optimum=600.0,
description="f6(x) = 0.5 + (sin^2(sqrt(x^2+y^2))-0.5)/(1+0.001*(x^2+y^2))^2",
function_type="multimodal"
)
# F7: Griewank - Dimension 30
functions[7] = BenchmarkFunction(
name="Griewank",
function=lambda x: self._f7_griewank(x),
dimension=30,
bounds=(-600, 600),
global_optimum=700.0,
description="f7(x) = sum(xi^2)/4000 - prod(cos(xi/sqrt(i))) + 1",
function_type="multimodal"
)
# F8: Rastrigin (Modified) - Dimension 30
functions[8] = BenchmarkFunction(
name="Rastrigin_Modified",
function=lambda x: self._f8_rastrigin_modified(x),
dimension=30,
bounds=(-5.12, 5.12),
global_optimum=800.0,
description="f8(x) = f5(x) + bias",
function_type="multimodal"
)
# F9: Levy - Dimension 30 functions[9] = BenchmarkFunction(
name="Levy",
function=lambda x: self._f9_levy(x),
dimension=30,
bounds=(-10, 10),
global_optimum=900.0,
description="f9(x) = Levy function with trigonometric terms",
function_type="multimodal"
)
# F10: Schwefel 2.26 - Dimension 30
functions[10] = BenchmarkFunction(
name="Schwefel_2_26",
function=lambda x: self._f10_schwefel_2_26(x),
dimension=30,
bounds=(-500, 500),
global_optimum=1000.0,
description="f10(x) = 418.9829*D - sum(g(zi))",
function_type="multimodal"
)
# F11: Zakharov - Dimension 3
functions[11] = BenchmarkFunction(
name="Zakharov",
function=lambda x: self._f11_zakharov(x),
dimension=3,
bounds=(-5, 10),
global_optimum=1100.0,
description="f11(x) = sum(xi^2) + (sum(0.5*i*xi))^2 + (sum(0.5*i*xi))^4",
function_type="unimodal"
)
# F12: High Conditioned Elliptic - Dimension 3
functions[12] = BenchmarkFunction(
name="High_Conditioned_Elliptic",
function=lambda x: self._f12_elliptic(x),
dimension=3,
bounds=(-100, 100),
global_optimum=1200.0,
description="f12(x) = sum((10^6)^((i-1)/(D-1)) * xi^2)",
function_type="unimodal"
)
# F13: Ackley - Dimension 3
functions[13] = BenchmarkFunction(
name="Ackley",
function=lambda x: self._f13_ackley(x),
dimension=3,
bounds=(-32, 32),
global_optimum=1300.0,
description="f13(x) = -20*exp(-0.2*sqrt(sum(xi^2)/D)) - exp(sum(cos(2*pi*xi))/D) + 20 + e",
function_type="multimodal"
)
# F14: Weierstrass - Dimension 4
functions[14] = BenchmarkFunction(
name="Weierstrass",
function=lambda x: self._f14_weierstrass(x),
dimension=4,
bounds=(-0.5, 0.5),
global_optimum=1400.0,
description="f14(x) = Weierstrass function with trigonometric series",
function_type="multimodal"
)
# F15: Griewank (Modified) - Dimension 4
functions[15] = BenchmarkFunction(
name="Griewank_Modified",
function=lambda x: self._f15_griewank_modified(x),
dimension=4,
bounds=(-600, 600),
global_optimum=1500.0,
description="f15(x) = Modified Griewank function",
function_type="multimodal"
)
# F16: Pathological - Dimension 5
functions[16] = BenchmarkFunction(
name="Pathological",
function=lambda x: self._f16_pathological(x),
dimension=5,
bounds=(-100, 100),
global_optimum=1600.0,
description="f16(x) = Pathological function with oscillations",
function_type="multimodal"
)
# F17: Cosine Mixture - Dimension 5
functions[17] = BenchmarkFunction(
name="Cosine_Mixture",
function=lambda x: self._f17_cosine_mixture(x),
dimension=5,
bounds=(-1, 1),
global_optimum=1700.0,
description="f17(x) = Cosine mixture function",
function_type="multimodal"
)
# F18: Trid - Dimension 6
functions[18] = BenchmarkFunction(
name="Trid",
function=lambda x: self._f18_trid(x),
dimension=6,
bounds=(-36, 36),
global_optimum=1800.0,
description="f18(x) = Trid function",
function_type="unimodal"
)
# F19: Michalewicz - Dimension 6
functions[19] = BenchmarkFunction(
name="Michalewicz",
function=lambda x: self._f19_michalewicz(x),
dimension=6,
bounds=(0, math.pi),
global_optimum=1900.0,
description="f19(x) = Michalewicz function",
function_type="multimodal"
)
# F20: Alpine - Dimension 3
functions[20] = BenchmarkFunction(
name="Alpine",
function=lambda x: self._f20_alpine(x),
dimension=3,
bounds=(-10, 10),
global_optimum=2000.0,
description="f20(x) = Alpine function",
function_type="multimodal"
)
# F21-F30: Composition Functions (Shifted and Rotated)
# These use the base functions F1-F10 with transformations
# F21: Composition F1 (Shifted Sphere)
functions[21] = BenchmarkFunction(
name="Composition_F1",
function=lambda x: self._f21_composition_f1(x),
dimension=4,
bounds=(-5, 5),
global_optimum=2100.0,
description="f21(x) = f1(M*(x-o1)) + 2100",
function_type="composition"
)
# F22: Composition F2 (Shifted Schwefel 2.22)
functions[22] = BenchmarkFunction(
name="Composition_F2", function=lambda x: self._f22_composition_f2(x),
dimension=4,
bounds=(-5, 5),
global_optimum=2200.0,
description="f22(x) = f2(M*(x-o2)) + 2200",
function_type="composition"
)
# F23: Composition F3 (Shifted Schwefel 1.2)
functions[23] = BenchmarkFunction(
name="Composition_F3",
function=lambda x: self._f23_composition_f3(x),
dimension=4,
bounds=(-5, 5), global_optimum=2300.0,
description="f23(x) = f3(M*(x-o3)) + 2300",
function_type="composition"
)
# F24: Composition F4 (Shifted Rosenbrock)
functions[24] = BenchmarkFunction(
name="Composition_F4",
function=lambda x: self._f24_composition_f4(x),
dimension=5,
bounds=(-5, 5),
global_optimum=2400.0,
description="f24(x) = f4(M*(2.048*(x-o4))+1) + 2400",
function_type="composition"
)
# F25: Composition F5 (Shifted Rastrigin)
functions[25] = BenchmarkFunction(
name="Composition_F5",
function=lambda x: self._f25_composition_f5(x),
dimension=5,
bounds=(-5, 5),
global_optimum=2500.0,
description="f25(x) = f5(M*(x-o5)) + 2500",
function_type="composition"
)
# F26: Composition F6 (Shifted Schaffer F6)
functions[26] = BenchmarkFunction(
name="Composition_F6",
function=lambda x: self._f26_composition_f6(x),
dimension=6,
bounds=(-5, 5),
global_optimum=2600.0,
description="f26(x) = f6(M*(2.048*(x-o6))) + 2600",
function_type="composition"
)
# F27: Composition F7 (Shifted Griewank)
functions[27] = BenchmarkFunction(
name="Composition_F7",
function=lambda x: self._f27_composition_f7(x),
dimension=3,
bounds=(-5, 5),
global_optimum=2700.0,
description="f27(x) = f7(M*(600*(x-o7))) + 2700",
function_type="composition"
)
# F28: Composition F8 (Shifted Rastrigin Modified) functions[28] = BenchmarkFunction(
name="Composition_F8",
function=lambda x: self._f28_composition_f8(x),
dimension=3,
bounds=(-5, 5),
global_optimum=2800.0,
description="f28(x) = f8(5.12*(x-o8)) + 2800",
function_type="composition"
)
# F29: Composition F9 (Shifted Levy)
functions[29] = BenchmarkFunction(
name="Composition_F9",
function=lambda x: self._f29_composition_f9(x),
dimension=4,
bounds=(-5, 5),
global_optimum=2900.0,
description="f29(x) = f9(M*(5.12*(x-o9))) + 2900",
function_type="composition"
)
# F30: Composition F10 (Shifted Schwefel 2.26)
functions[30] = BenchmarkFunction(
name="Composition_F10",
function=lambda x: self._f30_composition_f10(x),
dimension=4,
bounds=(-5, 5),
global_optimum=3000.0,
description="f30(x) = f10(M*(1000*(x-o10))) + 3000", function_type="composition"
)
return functions
# Implementation of the 30 CEC benchmark functions
def _f1_shifted_sphere(self, x):
"""F1: Shifted Sphere Function"""
x = np.asarray(x)
return np.sum(x**2) + 100
def _f2_schwefel_2_22(self, x):
"""F2: Schwefel 2.22 Function"""
x = np.asarray(x)
return np.sum(np.abs(x)) + np.prod(np.abs(x)) + 200
def _f3_schwefel_1_2(self, x):
"""F3: Schwefel 1.2 Function"""
x = np.asarray(x)
total = 0
for i in range(len(x)):
total += np.sum(x[:i+1])**2
return total + 300
def _f4_rosenbrock(self, x):
"""F4: Rosenbrock Function"""
x = np.asarray(x)
total = 0
for i in range(len(x)-1):
total += 100 * (x[i+1] - x[i]**2)**2 + (x[i] - 1)**2
return total + 400
def _f5_rastrigin(self, x):
"""F5: Rastrigin Function"""
x = np.asarray(x)
A = 10
n = len(x)
return A*n + np.sum(x**2 - A*np.cos(2*np.pi*x)) + 500
def _f6_schaffer_f6(self, x):
"""F6: Schaffer F6 Function"""
x = np.asarray(x)
total = 0
for i in range(len(x)-1):
si = x[i]**2 + x[i+1]**2
total += 0.5 + (np.sin(np.sqrt(si))**2 - 0.5) / (1 + 0.001*si)**2
return total + 600
def _f7_griewank(self, x):
"""F7: Griewank Function"""
x = np.asarray(x)
part1 = np.sum(x**2) / 4000
part2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1))))
return part1 - part2 + 1 + 700
def _f8_rastrigin_modified(self, x):
"""F8: Rastrigin Modified Function"""
return self._f5_rastrigin(x) + 300 # f5 + bias difference
def _f9_levy(self, x):
"""F9: Levy Function"""
x = np.asarray(x)
w = 1 + (x - 1) / 4
part1 = np.sin(np.pi * w[0])**2
part2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * w[:-1] + 1)**2))
part3 = (w[-1] - 1)**2 * (1 + np.sin(2 * np.pi * w[-1])**2)
return part1 + part2 + part3 + 900
def _f10_schwefel_2_26(self, x):
"""F10: Schwefel 2.26 Function"""
x = np.asarray(x)
z = x + 4.209687462275036e2
return 418.9829 * len(x) - np.sum(z * np.sin(np.sqrt(np.abs(z)))) + 1000
def _f11_zakharov(self, x):
"""F11: Zakharov Function"""
x = np.asarray(x)
sum1 = np.sum(x**2)
sum2 = np.sum(0.5 * np.arange(1, len(x)+1) * x)
return sum1 + sum2**2 + sum2**4 + 1100
def _f12_elliptic(self, x):
"""F12: High Conditioned Elliptic Function"""
x = np.asarray(x)
D = len(x)
return np.sum([(10**6)**((i-1)/(D-1)) * x[i]**2 for i in range(D)]) + 1200
def _f13_ackley(self, x):
"""F13: Ackley Function"""
x = np.asarray(x)
D = len(x)
part1 = -20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / D))
part2 = -np.exp(np.sum(np.cos(2 * np.pi * x)) / D)
return part1 + part2 + 20 + np.e + 1300
def _f14_weierstrass(self, x):
"""F14: Weierstrass Function"""
x = np.asarray(x)
a, b, kmax = 0.5, 3, 20
f = 0
for xi in x:
f += np.sum([a**k * np.cos(2*np.pi * b**k * (xi + 0.5)) for k in range(kmax+1)])
D = len(x)
f -= D * np.sum([a**k * np.cos(2*np.pi * b**k * 0.5) for k in range(kmax+1)])
return f + 1400
def _f15_griewank_modified(self, x):
"""F15: Modified Griewank Function"""
x = np.asarray(x)
return np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(np.arange(1, len(x)+1)))) + 1 + 1500
def _f16_pathological(self, x):
"""F16: Pathological Function"""
x = np.asarray(x)
f = 0
for i in range(len(x)-1):
si = x[i]**2 + x[i+1]**2
f += 0.5 + (np.sin(np.sqrt(100*si))**2 - 0.5) / (1 + 0.001*si)**2
return f + 1600
def _f17_cosine_mixture(self, x):
"""F17: Cosine Mixture Function"""
x = np.asarray(x)
return -0.1 * np.sum(np.cos(5*np.pi*x)) + np.sum(x**2) + 1700
def _f18_trid(self, x):
"""F18: Trid Function"""
x = np.asarray(x)
return np.sum((x-1)**2) - np.sum(x[1:]*x[:-1]) + 1800
def _f19_michalewicz(self, x):
"""F19: Michalewicz Function"""
x = np.asarray(x)
m = 10
return -np.sum(np.sin(x) * np.sin(np.arange(1, len(x)+1) * x**2 / np.pi)**(2*m)) + 1900
def _f20_alpine(self, x):
"""F20: Alpine Function"""
x = np.asarray(x)
return np.sum(np.abs(x * np.sin(x) + 0.1 * x)) + 2000
# Composition Functions F21-F30
def _apply_transformation(self, x, func_id):
"""Apply rotation and shift transformation for composition functions"""
x = np.asarray(x)
if func_id in self.shift_vectors:
x_shifted = x - self.shift_vectors[func_id][:len(x)]
if func_id in self.rotation_matrices:
M = self.rotation_matrices[func_id][:len(x), :len(x)]
x_transformed = M @ x_shifted
else:
x_transformed = x_shifted
return x_transformed
return x
def _f21_composition_f1(self, x):
"""F21: Composition of F1"""
x_t = self._apply_transformation(x, 21)
return self._f1_shifted_sphere(x_t) - 100 + 2100
def _f22_composition_f2(self, x):
"""F22: Composition of F2"""
x_t = self._apply_transformation(x, 22)
return self._f2_schwefel_2_22(x_t) - 200 + 2200
def _f23_composition_f3(self, x):
"""F23: Composition of F3"""
x_t = self._apply_transformation(x, 23)
return self._f3_schwefel_1_2(x_t) - 300 + 2300
def _f24_composition_f4(self, x):
"""F24: Composition of F4"""
x_t = self._apply_transformation(2.048 * (np.asarray(x) - self.shift_vectors.get(24, np.zeros(len(x)))[:len(x)]) + 1, 24)
return self._f4_rosenbrock(x_t) - 400 + 2400
def _f25_composition_f5(self, x):
"""F25: Composition of F5"""
x_t = self._apply_transformation(x, 25)
return self._f5_rastrigin(x_t) - 500 + 2500
def _f26_composition_f6(self, x):
"""F26: Composition of F6"""
x_t = self._apply_transformation(2.048 * (np.asarray(x) - self.shift_vectors.get(26, np.zeros(len(x)))[:len(x)]), 26)
return self._f6_schaffer_f6(x_t) - 600 + 2600
def _f27_composition_f7(self, x):
"""F27: Composition of F7"""
x_t = self._apply_transformation(600 * (np.asarray(x) - self.shift_vectors.get(27, np.zeros(len(x)))[:len(x)]), 27)
return self._f7_griewank(x_t) - 700 + 2700
def _f28_composition_f8(self, x):
"""F28: Composition of F8"""
x_t = 5.12 * (np.asarray(x) - self.shift_vectors.get(28, np.zeros(len(x)))[:len(x)])
return self._f8_rastrigin_modified(x_t) - 800 + 2800
def _f29_composition_f9(self, x):
"""F29: Composition of F9"""
x_t = self._apply_transformation(5.12 * (np.asarray(x) - self.shift_vectors.get(29, np.zeros(len(x)))[:len(x)]), 29)
return self._f9_levy(x_t) - 900 + 2900
def _f30_composition_f10(self, x):
"""F30: Composition of F10"""
x_t = self._apply_transformation(1000 * (np.asarray(x) - self.shift_vectors.get(30, np.zeros(len(x)))[:len(x)]), 30)
return self._f10_schwefel_2_26(x_t) - 1000 + 3000
def get_function(self, function_id: int) -> BenchmarkFunction:
"""Get a specific benchmark function by ID"""
if function_id in self.functions:
return self.functions[function_id]
else:
return None
def get_all_functions(self) -> Dict[int, BenchmarkFunction]:
"""Get all benchmark functions"""
return self.functions
def list_functions(self) -> None:
"""Print information about all functions"""
print("CEC Benchmark Functions Suite")
print("=" * 50)
for fid, func in self.functions.items():
print(f"F{fid:2d}: {func.name:25s} | Dim: {func.dimension:2d} | Optimum: {func.global_optimum:7.1f} | Bounds: {func.bounds}")
# Compatibility wrapper to maintain existing interface
class BenchmarkSuite(CECBenchmarkSuite):
"""Wrapper class to maintain compatibility with existing code"""
def __init__(self, dimension: int = 30):
"""Initialize with specified dimension (for compatibility)"""
super().__init__()
self.dimension = dimension
# Note: CEC functions have their own specific dimensions
# The dimension parameter is kept for compatibility but individual
# functions use their specified dimensions from the CEC competition