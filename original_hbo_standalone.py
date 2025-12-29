#!/usr/bin/env python3
"""
Original Heap-Based Optimization (HBO) Algorithm - Standalone Script
===================================================================
Implementation of the original HBO algorithm from:
Askari Q, Saeed M, Younas I. Heap-based optimizer inspired by corporate rank hierarchy for global optimization. Expert Systems with Applications. 2020.
This is a standalone script that can be run independently.
Author: Medical AI Research Team
Version: 1.0.0
"""
import numpy as np
import copy
from typing import Dict, List, Tuple, Callable, Optional
import logging
from abc import ABC, abstractmethod
class HBONode:
"""Represents a node in the HBO heap structure"""
def __init__(self, position: np.ndarray, fitness: float, index: int):
self.position = position.copy()
self.fitness = fitness
self.index = index
self.parent_index = -1
self.children_indices = []
self.level = 0
class OriginalHBO:
"""
Original Heap-Based Optimization (HBO) Algorithm
Inspired by corporate rank hierarchy for global optimization.
Uses heap data structure to model hierarchical relationships.
"""
def __init__(self, objective_function: Callable,
dimension: int,
bounds: Tuple[np.ndarray, np.ndarray],
population_size: int = 40,
max_iterations: int = 500,
degree: int = 3,
cycles: int = 4,
verbose: bool = True):
"""
Initialize HBO Algorithm
Args:
objective_function: Function to minimize
dimension: Problem dimension
bounds: (lower_bounds, upper_bounds)
population_size: Number of search agents
max_iterations: Maximum number of iterations
degree: Heap degree (default: 3-ary heap)
cycles: Number of cycles for gamma calculation
verbose: Enable detailed logging
"""
self.objective_function = objective_function
self.dimension = dimension
self.lower_bounds = np.array(bounds[0])
self.upper_bounds = np.array(bounds[1])
self.population_size = population_size
self.max_iterations = max_iterations
self.degree = degree
self.cycles = cycles
self.verbose = verbose
# Algorithm parameters
self.p1 = 0.3 # Probability threshold 1
self.p2 = 0.7 # Probability threshold 2
self.iter_per_cycle = max(1, max_iterations // cycles) # Prevent division by zero
self.qtr_cycle = max(1, self.iter_per_cycle // 4) # Prevent division by zero
# Initialize storage
self.heap = []
self.best_position = None
self.best_fitness = float('inf')
self.convergence_history = []
# Setup logging
if verbose:
logging.basicConfig(level=logging.INFO)
self.logger = logging.getLogger(__name__)
else:
self.logger = logging.getLogger(__name__)
self.logger.addHandler(logging.NullHandler())
def _initialize_population(self) -> List[HBONode]:
"""Initialize the HBO population with random positions"""
population = []
for i in range(self.population_size):
# Random position within bounds
position = np.random.uniform(self.lower_bounds, self.upper_bounds)
# Evaluate fitness
fitness = self.objective_function(position)
# Create HBO node
node = HBONode(position, fitness, i)
population.append(node)
# Update global best
if fitness < self.best_fitness:
self.best_fitness = fitness
self.best_position = position.copy()
return population
def _build_heap(self, population: List[HBONode]) -> List[HBONode]:
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
def _get_parent_position(self, node_index: int, heap: List[HBONode]) -> Optional[np.ndarray]:
"""Get parent position for a given node"""
if node_index == 0 or node_index >= len(heap):
return None
parent_idx = heap[node_index].parent_index
if 0 <= parent_idx < len(heap):
return heap[parent_idx].position
return None
def _get_colleague_position(self, node_index: int, heap: List[HBONode]) -> Optional[np.ndarray]:
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
def _calculate_gamma(self, iteration: int) -> float:
"""Calculate gamma parameter for position update"""
cycle_position = (iteration % self.iter_per_cycle) + 1
gamma = abs(2 - cycle_position / self.qtr_cycle)
return gamma
def _update_position(self, node: HBONode, parent_pos: Optional[np.ndarray],
colleague_pos: Optional[np.ndarray],
gamma: float) -> np.ndarray:
"""Update position based on HBO rules"""
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
new_position = np.clip(new_position, self.lower_bounds, self.upper_bounds)
return new_position
def _boundary_handling(self, position: np.ndarray) -> np.ndarray:
"""Handle boundary constraints"""
# Clip to bounds
bounded_position = np.clip(position, self.lower_bounds, self.upper_bounds)
# Random repositioning for out-of-bounds dimensions
for i in range(self.dimension):
if position[i] < self.lower_bounds[i] or position[i] > self.upper_bounds[i]:
if np.random.random() < 0.5:
bounded_position[i] = np.random.uniform(
self.lower_bounds[i], self.upper_bounds[i]
)
return bounded_position
def optimize(self) -> Dict:
"""
Run the complete HBO optimization process
Returns:
Dictionary containing optimization results
"""
self.logger.info("Starting Original HBO Optimization")
self.logger.info(f"Population Size: {self.population_size}")
self.logger.info(f"Max Iterations: {self.max_iterations}")
self.logger.info(f"Problem Dimension: {self.dimension}")
# Initialize population
population = self._initialize_population()
# Main optimization loop
for iteration in range(self.max_iterations):
# Build heap structure
heap = self._build_heap(population)
# Calculate gamma for this iteration
gamma = self._calculate_gamma(iteration)
# Update each agent
new_population = []
for i, node in enumerate(heap):
# Get parent and colleague positions
parent_pos = self._get_parent_position(i, heap)
colleague_pos = self._get_colleague_position(i, heap)
# Update position
new_position = self._update_position(node, parent_pos, colleague_pos, gamma)
# Handle boundaries
new_position = self._boundary_handling(new_position)
# Evaluate new fitness
new_fitness = self.objective_function(new_position)
# Greedy selection
if new_fitness < node.fitness:
# Accept new position
new_node = HBONode(new_position, new_fitness, node.index)
new_population.append(new_node)
# Update global best
if new_fitness < self.best_fitness:
self.best_fitness = new_fitness
self.best_position = new_position.copy()
else:
# Keep old position
new_population.append(node)
# Update population
population = new_population
# Store convergence
self.convergence_history.append(self.best_fitness)
# Logging
if self.verbose and (iteration + 1) % 50 == 0:
self.logger.info(f"Iteration {iteration + 1}: Best Fitness = {self.best_fitness:.6e}")
self.logger.info(f"HBO Optimization Complete. Best Fitness: {self.best_fitness:.6e}")
return {
'best_position': self.best_position,
'best_fitness': self.best_fitness,
'convergence_history': np.array(self.convergence_history),
'final_population': population,
'algorithm': 'Original_HBO',
'iterations': self.max_iterations,
'population_size': self.population_size
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
'iterations_to_best': np.argmin(convergence) + 1 if len(convergence) > 0 else 0
}
# Example benchmark functions for testing
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
def run_hbo_experiment(objective_function, function_name, dimension=10, bounds=(-100, 100)):
"""
Run HBO experiment on a given objective function
Args:
objective_function: Function to optimize
function_name: Name of the function for display
dimension: Problem dimension
bounds: Search space bounds (tuple)
"""
print(f"\n{'='*60}")
print(f"Running HBO on {function_name} Function")
print(f"{'='*60}")
# Set bounds
if isinstance(bounds, tuple):
lower_bounds = np.full(dimension, bounds[0])
upper_bounds = np.full(dimension, bounds[1])
else:
lower_bounds = bounds[0]
upper_bounds = bounds[1]
# Initialize HBO
hbo = OriginalHBO(
objective_function=objective_function,
dimension=dimension,
bounds=(lower_bounds, upper_bounds),
population_size=30,
max_iterations=500,
verbose=True
)
# Run optimization
results = hbo.optimize()
# Display results
print(f"\nResults for {function_name}:")
print(f"Best Fitness: {results['best_fitness']:.6e}")
print(f"Best Position: {results['best_position']}")
print(f"Convergence in {len(results['convergence_history'])} iterations")
# Get statistics
stats = hbo.get_statistics()
print(f"\nStatistics:")
for key, value in stats.items():
if isinstance(value, float):
print(f" {key}: {value:.6e}" if abs(value) > 1e-3 else f" {key}: {value}")
else:
print(f" {key}: {value}")
return results
if __name__ == "__main__":
print("Original Heap-Based Optimization (HBO) Algorithm")
print("=" * 60)
print("Standalone Implementation for Benchmark Testing")
print("=" * 60)
# Test functions with their optimal bounds
test_functions = [
(sphere, "Sphere", 10, (-100, 100)),
(rastrigin, "Rastrigin", 10, (-5.12, 5.12)),
(rosenbrock, "Rosenbrock", 10, (-5, 10)),
(ackley, "Ackley", 10, (-32.768, 32.768))
]
# Run experiments
all_results = {}
for func, name, dim, bounds in test_functions:
try:
result = run_hbo_experiment(func, name, dim, bounds)
all_results[name] = result
except Exception as e:
print(f"Error running {name}: {e}")
# Summary
print(f"\n{'='*60}")
print("EXPERIMENT SUMMARY")
print(f"{'='*60}")
for name, result in all_results.items():
print(f"{name:12}: Best Fitness = {result['best_fitness']:.6e}")
print(f"\nAll experiments completed successfully!")
print("Original HBO algorithm is ready for integration.")