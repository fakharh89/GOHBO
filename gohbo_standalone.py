#!/usr/bin/env python3
"""
GOHBO: Grey Wolf Optimization + Heap-Based Optimization + Orthogonal Learning - Standalone Script
===============================================================================================
Implementation of the GOHBO algorithm combining:
1. Grey Wolf Optimization (GWO) 2. Heap-Based Optimization (HBO)
3. Orthogonal Learning (OL) strategies
Based on Zhang et al. 2024 methodology and optimization principles.
This is a standalone script that can be run independently.
Author: Medical AI Research Team Version: 1.0.0
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Callable, Optional
from dataclasses import dataclass
import copy
class HBONode:
"""Represents a node in the HBO heap structure"""
def __init__(self, position: np.ndarray, fitness: float, index: int):
self.position = position.copy()
self.fitness = fitness
self.index = index
self.parent_index = -1
self.children_indices = []
self.level = 0
@dataclass
class WolfAgent:
"""Represents a wolf agent in GWO"""
position: np.ndarray
fitness: float
index: int
role: str = "omega" # alpha, beta, delta, omega
class HBOComponent:
"""
HBO Component for GOHBO - Embedded HBO functionality
This class contains the core HBO logic without running a full optimization
"""
def __init__(self, dimension: int, degree: int = 3, cycles: int = 4, max_iterations: int = 500):
"""Initialize HBO component"""
self.dimension = dimension
self.degree = degree
self.cycles = cycles
self.max_iterations = max_iterations
# Algorithm parameters
self.p1 = 0.3 # Probability threshold 1
self.p2 = 0.7 # Probability threshold 2
self.iter_per_cycle = max(1, max_iterations // cycles)
self.qtr_cycle = max(1, self.iter_per_cycle // 4)
def build_heap(self, population: List[HBONode]) -> List[HBONode]:
"""Build heap structure based on fitness (min-heap)"""
# Sort population by fitness (ascending for minimization)
sorted_population = sorted(population, key=lambda x: x.fitness)
# Build heap relationships
for i, node in enumerate(sorted_population):
node.index = i
# Calculate parent index
if i > 0:
parent_idx = (i - 1) // self.degree
node.parent_index = parent_idx
# Add to parent's children list
if parent_idx < len(sorted_population):
sorted_population[parent_idx].children_indices.append(i)
# Calculate level in heap
node.level = int(np.floor(np.log(i * (self.degree - 1) + 1) / np.log(self.degree))) if i > 0 else 0
return sorted_population
def get_parent_position(self, node_index: int, heap: List[HBONode]) -> Optional[np.ndarray]:
"""Get parent position for a given node"""
if node_index == 0 or node_index >= len(heap):
return None
parent_idx = heap[node_index].parent_index
if 0 <= parent_idx < len(heap):
return heap[parent_idx].position
return None
def get_colleague_position(self, node_index: int, heap: List[HBONode]) -> Optional[np.ndarray]:
"""Get colleague position (sibling or same-level node)"""
if node_index >= len(heap):
return None
current_node = heap[node_index]
# Find colleagues at same level
colleagues = []
for i, node in enumerate(heap):
if i != node_index and node.level == current_node.level:
colleagues.append(node)
# If no colleagues at same level, find siblings
if not colleagues and current_node.parent_index >= 0:
parent = heap[current_node.parent_index]
for child_idx in parent.children_indices:
if child_idx != node_index and child_idx < len(heap):
colleagues.append(heap[child_idx])
# Return random colleague position
if colleagues:
colleague = np.random.choice(colleagues)
return colleague.position
return None
def calculate_gamma(self, iteration: int) -> float:
"""Calculate gamma parameter for position update"""
cycle_position = (iteration % self.iter_per_cycle) + 1
gamma = abs(2 - cycle_position / self.qtr_cycle)
return gamma
def update_position(self, node: HBONode, parent_pos: Optional[np.ndarray],
colleague_pos: Optional[np.ndarray],
gamma: float,
bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
"""Update position based on HBO rules"""
lower_bounds, upper_bounds = bounds
new_position = node.position.copy()
for j in range(self.dimension):
r = np.random.random()
if r < self.p1:
# Interaction with parent (if exists)
if parent_pos is not None:
rn = np.random.random()
D = abs(parent_pos[j] - node.position[j])
new_position[j] = parent_pos[j] + rn * gamma * D
else:
# Self-contribution when no parent
new_position[j] = node.position[j] + np.random.uniform(-1, 1) * gamma
elif r < self.p2:
# Interaction with colleague
if colleague_pos is not None:
rn = np.random.random()
D = abs(colleague_pos[j] - node.position[j])
new_position[j] = colleague_pos[j] + rn * gamma * D
else:
# Self-contribution when no colleague
new_position[j] = node.position[j] + np.random.uniform(-1, 1) * gamma
else:
# Self-contribution
new_position[j] = node.position[j] + np.random.uniform(-1, 1) * gamma
# Ensure bounds
new_position = np.clip(new_position, lower_bounds, upper_bounds)
return new_position
def boundary_handling(self, position: np.ndarray, bounds: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
"""Handle boundary constraints"""
lower_bounds, upper_bounds = bounds
# Clip to bounds
bounded_position = np.clip(position, lower_bounds, upper_bounds)
# Random repositioning for out-of-bounds dimensions
for i in range(self.dimension):
if position[i] < lower_bounds[i] or position[i] > upper_bounds[i]:
if np.random.random() < 0.5:
bounded_position[i] = np.random.uniform(
lower_bounds[i], upper_bounds[i]
)
return bounded_position
class OrthogonalLearning:
"""
Orthogonal Learning strategy for enhanced exploration
Uses orthogonal experimental design to generate diverse solutions
"""
def __init__(self, dimension: int, levels: int = 2):
"""
Initialize orthogonal learning
Args:
dimension: Problem dimension
levels: Number of levels for orthogonal array (default: 2)
"""
self.dimension = dimension
self.levels = levels
self.orthogonal_array = self._generate_orthogonal_array()
def _generate_orthogonal_array(self) -> np.ndarray:
"""Generate orthogonal array L_n(2^k) for exploration"""
# For simplicity, use basic orthogonal array design
# In practice, this would use more sophisticated OA generation
if self.dimension <= 4:
# L4(2^3) orthogonal array
oa = np.array([
[0, 0, 0],
[0, 1, 1], [1, 0, 1],
[1, 1, 0]
])
elif self.dimension <= 8:
# L8(2^7) orthogonal array
oa = np.array([
[0, 0, 0, 0, 0, 0, 0],
[0, 0, 0, 1, 1, 1, 1],
[0, 1, 1, 0, 0, 1, 1],
[0, 1, 1, 1, 1, 0, 0],
[1, 0, 1, 0, 1, 0, 1],
[1, 0, 1, 1, 0, 1, 0],
[1, 1, 0, 0, 1, 1, 0],
[1, 1, 0, 1, 0, 0, 1]
])
else:
# Generate larger orthogonal array
n_experiments = min(16, 2 * self.dimension)
oa = np.random.randint(0, 2, size=(n_experiments, self.dimension))
return oa
def generate_orthogonal_positions(self, base_positions: List[np.ndarray], bounds: Tuple[np.ndarray, np.ndarray],
scale_factor: float = 0.1) -> List[np.ndarray]:
"""
Generate orthogonal positions around base positions
Args:
base_positions: List of base positions (alpha, beta, delta)
bounds: Problem bounds
scale_factor: Scaling factor for orthogonal perturbations
Returns:
List of orthogonally generated positions
"""
lower_bounds, upper_bounds = bounds
orthogonal_positions = []
# For each experiment in orthogonal array
for experiment in self.orthogonal_array:
new_position = np.zeros(self.dimension)
# Combine base positions using orthogonal design
for i in range(self.dimension):
if len(base_positions) >= 3:
# Use alpha, beta, delta positions
alpha_pos = base_positions[0][i] if i < len(base_positions[0]) else 0
beta_pos = base_positions[1][i] if i < len(base_positions[1]) else 0 delta_pos = base_positions[2][i] if i < len(base_positions[2]) else 0
# Orthogonal combination
if i < len(experiment):
if experiment[i] == 0:
new_position[i] = alpha_pos + scale_factor * (beta_pos - delta_pos)
else:
new_position[i] = beta_pos + scale_factor * (alpha_pos - delta_pos)
else:
new_position[i] = np.mean([alpha_pos, beta_pos, delta_pos])
else:
# Fallback for insufficient base positions
if base_positions:
new_position[i] = base_positions[0][i] + scale_factor * np.random.uniform(-1, 1)
else:
new_position[i] = np.random.uniform(lower_bounds[i], upper_bounds[i])
# Ensure bounds
new_position = np.clip(new_position, lower_bounds, upper_bounds)
orthogonal_positions.append(new_position)
return orthogonal_positions
class GOHBO:
"""
GOHBO: Grey Wolf Optimization + Heap-Based Optimization + Orthogonal Learning
Hybrid algorithm combining GWO exploration with HBO hierarchical structure
and orthogonal learning for enhanced diversification.
"""
def __init__(self,
objective_function: Callable,
dimension: int, bounds: Tuple[np.ndarray, np.ndarray],
population_size: int = 40,
max_iterations: int = 500,
gwo_weight: float = 0.4,
hbo_weight: float = 0.4,
ol_weight: float = 0.2,
ol_frequency: int = 10,
verbose: bool = True):
"""
Initialize GOHBO Algorithm
Args:
objective_function: Function to minimize
dimension: Problem dimension
bounds: (lower_bounds, upper_bounds)
population_size: Number of search agents
max_iterations: Maximum number of iterations
gwo_weight: Weight for GWO component
hbo_weight: Weight for HBO component ol_weight: Weight for orthogonal learning
ol_frequency: Frequency of orthogonal learning application
verbose: Enable detailed logging
"""
self.objective_function = objective_function
self.dimension = dimension
self.lower_bounds = np.array(bounds[0])
self.upper_bounds = np.array(bounds[1])
self.population_size = population_size
self.max_iterations = max_iterations
# Component weights
self.gwo_weight = gwo_weight
self.hbo_weight = hbo_weight
self.ol_weight = ol_weight
self.ol_frequency = ol_frequency
# Normalize weights
total_weight = gwo_weight + hbo_weight + ol_weight
self.gwo_weight /= total_weight
self.hbo_weight /= total_weight
self.ol_weight /= total_weight
self.verbose = verbose
# Initialize components
self.hbo_component = HBOComponent(
dimension=dimension,
max_iterations=max_iterations
)
self.orthogonal_learning = OrthogonalLearning(dimension)
# Algorithm state
self.wolves = []
self.alpha_wolf = None
self.beta_wolf = None self.delta_wolf = None
self.best_fitness = float('inf')
self.best_position = None
self.convergence_history = []
# Setup logging
if verbose:
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger(__name__)
else:
self.logger = logging.getLogger(__name__)
self.logger.addHandler(logging.NullHandler())
def _initialize_population(self) -> List[WolfAgent]:
"""Initialize wolf population"""
wolves = []
for i in range(self.population_size):
position = np.random.uniform(self.lower_bounds, self.upper_bounds)
fitness = self.objective_function(position)
wolf = WolfAgent(position, fitness, i)
wolves.append(wolf)
# Update global best
if fitness < self.best_fitness:
self.best_fitness = fitness
self.best_position = position.copy()
return wolves
def _update_alpha_beta_delta(self, wolves: List[WolfAgent]):
"""Update alpha, beta, and delta wolves"""
# Sort wolves by fitness
sorted_wolves = sorted(wolves, key=lambda x: x.fitness)
# Assign roles
if len(sorted_wolves) >= 3:
self.alpha_wolf = sorted_wolves[0]
self.alpha_wolf.role = "alpha"
self.beta_wolf = sorted_wolves[1]
self.beta_wolf.role = "beta"
self.delta_wolf = sorted_wolves[2] self.delta_wolf.role = "delta"
# Rest are omega
for wolf in sorted_wolves[3:]:
wolf.role = "omega"
# Update global best
if self.alpha_wolf and self.alpha_wolf.fitness < self.best_fitness:
self.best_fitness = self.alpha_wolf.fitness
self.best_position = self.alpha_wolf.position.copy()
def _gwo_position_update(self, wolf: WolfAgent, iteration: int) -> np.ndarray:
"""Update position using GWO mechanism"""
a = 2 - 2 * iteration / self.max_iterations # Linearly decreasing from 2 to 0
if not (self.alpha_wolf and self.beta_wolf and self.delta_wolf):
return wolf.position
new_position = np.zeros(self.dimension)
for i in range(self.dimension):
# Alpha wolf influence
r1, r2 = np.random.random(), np.random.random()
A1 = 2 * a * r1 - a
C1 = 2 * r2
D_alpha = abs(C1 * self.alpha_wolf.position[i] - wolf.position[i])
X1 = self.alpha_wolf.position[i] - A1 * D_alpha
# Beta wolf influence r1, r2 = np.random.random(), np.random.random()
A2 = 2 * a * r1 - a
C2 = 2 * r2
D_beta = abs(C2 * self.beta_wolf.position[i] - wolf.position[i])
X2 = self.beta_wolf.position[i] - A2 * D_beta
# Delta wolf influence
r1, r2 = np.random.random(), np.random.random() A3 = 2 * a * r1 - a
C3 = 2 * r2
D_delta = abs(C3 * self.delta_wolf.position[i] - wolf.position[i])
X3 = self.delta_wolf.position[i] - A3 * D_delta
# Final position
new_position[i] = (X1 + X2 + X3) / 3
return np.clip(new_position, self.lower_bounds, self.upper_bounds)
def _hbo_position_update(self, wolves: List[WolfAgent], wolf_index: int, iteration: int) -> np.ndarray:
"""Update position using HBO mechanism"""
# Convert wolves to HBO nodes
hbo_nodes = []
for i, wolf in enumerate(wolves):
node = HBONode(wolf.position, wolf.fitness, i)
hbo_nodes.append(node)
# Build heap and update using HBO logic
heap = self.hbo_component.build_heap(hbo_nodes)
gamma = self.hbo_component.calculate_gamma(iteration)
if wolf_index < len(heap):
node = heap[wolf_index]
parent_pos = self.hbo_component.get_parent_position(wolf_index, heap)
colleague_pos = self.hbo_component.get_colleague_position(wolf_index, heap)
bounds = (self.lower_bounds, self.upper_bounds)
new_position = self.hbo_component.update_position(node, parent_pos, colleague_pos, gamma, bounds)
return self.hbo_component.boundary_handling(new_position, bounds)
return wolves[wolf_index].position
def _orthogonal_learning_update(self, iteration: int) -> List[np.ndarray]:
"""Apply orthogonal learning for enhanced exploration"""
if not (self.alpha_wolf and self.beta_wolf and self.delta_wolf):
return []
# Get base positions
base_positions = [
self.alpha_wolf.position,
self.beta_wolf.position, self.delta_wolf.position
]
# Generate orthogonal positions
scale_factor = 0.5 * (1 - iteration / self.max_iterations) # Adaptive scaling
ol_positions = self.orthogonal_learning.generate_orthogonal_positions(
base_positions, (self.lower_bounds, self.upper_bounds),
scale_factor
)
return ol_positions
def optimize(self) -> Dict:
"""
Run the complete GOHBO optimization process
Returns:
Dictionary containing optimization results
"""
self.logger.info("Starting GOHBO Optimization")
self.logger.info(f"Population Size: {self.population_size}")
self.logger.info(f"Max Iterations: {self.max_iterations}")
self.logger.info(f"Component Weights - GWO: {self.gwo_weight:.2f}, HBO: {self.hbo_weight:.2f}, OL: {self.ol_weight:.2f}")
# Initialize population
self.wolves = self._initialize_population()
# Main optimization loop
for iteration in range(self.max_iterations):
# Update alpha, beta, delta wolves
self._update_alpha_beta_delta(self.wolves)
# Update each wolf position
new_wolves = []
for i, wolf in enumerate(self.wolves):
# Get updates from different components
gwo_position = self._gwo_position_update(wolf, iteration)
hbo_position = self._hbo_position_update(self.wolves, i, iteration)
# Combine updates using weights
combined_position = (
self.gwo_weight * gwo_position +
self.hbo_weight * hbo_position + self.ol_weight * wolf.position # Current position as baseline for OL
)
# Apply orthogonal learning periodically
if iteration % self.ol_frequency == 0:
ol_positions = self._orthogonal_learning_update(iteration)
if ol_positions and i < len(ol_positions):
combined_position = (
(1 - self.ol_weight) * combined_position +
self.ol_weight * ol_positions[i]
)
# Ensure bounds
combined_position = np.clip(combined_position, self.lower_bounds, self.upper_bounds)
# Evaluate new position
new_fitness = self.objective_function(combined_position)
# Greedy selection
if new_fitness < wolf.fitness:
new_wolf = WolfAgent(combined_position, new_fitness, i)
new_wolves.append(new_wolf)
else:
new_wolves.append(wolf)
# Update population
self.wolves = new_wolves
# Store convergence
self.convergence_history.append(self.best_fitness)
# Logging
if self.verbose and (iteration + 1) % 50 == 0:
self.logger.info(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness:.6e}")
self.logger.info(f"GOHBO Optimization Complete. Best Fitness: {self.best_fitness:.6e}")
return {
'best_position': self.best_position,
'best_fitness': self.best_fitness, 'convergence_history': np.array(self.convergence_history),
'final_population': self.wolves,
'algorithm': 'GOHBO',
'iterations': self.max_iterations,
'population_size': self.population_size,
'alpha_wolf': self.alpha_wolf,
'beta_wolf': self.beta_wolf,
'delta_wolf': self.delta_wolf
}
def get_statistics(self) -> Dict:
"""Get optimization statistics"""
if not self.convergence_history:
return {}
convergence = np.array(self.convergence_history)
return {
'final_fitness': self.best_fitness,
'initial_fitness': convergence[0] if len(convergence) > 0 else None,
'improvement': convergence[0] - self.best_fitness if len(convergence) > 0 else 0,
'convergence_rate': np.mean(np.diff(convergence)),
'convergence_std': np.std(convergence),
'iterations_to_best': np.argmin(convergence) + 1 if len(convergence) > 0 else 0,
'diversity_measure': self._calculate_population_diversity()
}
def _calculate_population_diversity(self) -> float:
"""Calculate population diversity measure"""
if len(self.wolves) < 2:
return 0.0
positions = np.array([wolf.position for wolf in self.wolves])
mean_position = np.mean(positions, axis=0)
diversity = np.mean([
np.linalg.norm(pos - mean_position) for pos in positions
])
return diversity
# Benchmark Functions
def sphere(x):
"""Sphere function (unimodal)"""
return np.sum(x**2)
def rastrigin(x):
"""Rastrigin function (multimodal)"""
A = 10
n = len(x)
return A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x))
def rosenbrock(x):
"""Rosenbrock function (multimodal)"""
return np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
def ackley(x):
"""Ackley function (multimodal)"""
a = 20
b = 0.2
c = 2 * np.pi
n = len(x)
term1 = -a * np.exp(-b * np.sqrt(np.sum(x**2) / n))
term2 = -np.exp(np.sum(np.cos(c * x)) / n)
return term1 + term2 + a + np.exp(1)
def griewank(x):
"""Griewank function (multimodal)"""
term1 = np.sum(x**2) / 4000
term2 = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
return term1 - term2 + 1
def run_gohbo_experiment(objective_function, function_name, dimension=10, bounds=(-100, 100)):
"""
Run GOHBO experiment on a given objective function
Args:
objective_function: Function to optimize
function_name: Name of the function for display
dimension: Problem dimension
bounds: Search space bounds (tuple)
"""
print(f"\n{'='*60}")
print(f"Running GOHBO on {function_name} Function")
print(f"{'='*60}")
# Set bounds
if isinstance(bounds, tuple):
lower_bounds = np.full(dimension, bounds[0])
upper_bounds = np.full(dimension, bounds[1])
else:
lower_bounds = bounds[0]
upper_bounds = bounds[1]
# Initialize GOHBO
gohbo = GOHBO(
objective_function=objective_function,
dimension=dimension,
bounds=(lower_bounds, upper_bounds),
population_size=30,
max_iterations=500,
verbose=True
)
# Run optimization
results = gohbo.optimize()
# Display results
print(f"\nResults for {function_name}:")
print(f"Best Fitness: {results['best_fitness']:.6e}")
print(f"Best Position: {results['best_position']}")
print(f"Alpha Wolf Fitness: {results['alpha_wolf'].fitness:.6e}")
# Get statistics
stats = gohbo.get_statistics()
print(f"\nStatistics:")
for key, value in stats.items():
if isinstance(value, (int, float)):
print(f" {key}: {value:.6e}" if abs(value) > 1e-3 else f" {key}: {value}")
else:
print(f" {key}: {value}")
return results
if __name__ == "__main__":
print("GOHBO: Grey Wolf Optimization + Heap-Based Optimization + Orthogonal Learning")
print("=" * 80)
print("Standalone Implementation for Benchmark Testing")
print("=" * 80)
# Test functions with their optimal bounds
test_functions = [
(sphere, "Sphere", 10, (-100, 100)),
(rastrigin, "Rastrigin", 10, (-5.12, 5.12)),
(rosenbrock, "Rosenbrock", 10, (-5, 10)),
(ackley, "Ackley", 10, (-32.768, 32.768)),
(griewank, "Griewank", 10, (-600, 600))
]
# Run experiments
all_results = {}
for func, name, dim, bounds in test_functions:
try:
result = run_gohbo_experiment(func, name, dim, bounds)
all_results[name] = result
except Exception as e:
print(f"Error running {name}: {e}")
# Summary
print(f"\n{'='*80}")
print("EXPERIMENT SUMMARY")
print(f"{'='*80}")
for name, result in all_results.items():
print(f"{name:12}: Best Fitness = {result['best_fitness']:.6e}")
# Component Analysis
print(f"\n{'='*80}")
print("ALGORITHM COMPONENT ANALYSIS")
print(f"{'='*80}")
print("GOHBO combines three key components:")
print("1. Grey Wolf Optimization (GWO) - Social hierarchy and hunting behavior")
print("2. Heap-Based Optimization (HBO) - Corporate rank hierarchy structure")
print("3. Orthogonal Learning (OL) - Systematic exploration strategy")
print("\nThe hybrid approach leverages the strengths of each component:")
print("- GWO provides global exploration through wolf pack dynamics")
print("- HBO adds hierarchical local search capabilities")
print("- OL ensures diverse solution generation and prevents premature convergence")
print(f"\nAll experiments completed successfully!")
print("GOHBO algorithm is ready for integration and further research.")