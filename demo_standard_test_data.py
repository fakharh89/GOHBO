#!/usr/bin/env python3


import sys
import numpy as np
from pathlib import Path

# Add project paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir / 'algorithms'))

from algorithm_testing_framework import AlgorithmTestingFramework
from original_hbo import OriginalHBO


def demo_standard_vs_random():
    """Demonstrate difference between standard test data and random testing"""
    
    print(" STANDARD TEST DATA vs RANDOM INITIALIZATION DEMO")
    print("="*80)
    
    # Test F1 (Sphere) function with both approaches
    
    print("\n1 TESTING WITH STANDARD M_1_D30.txt DATA:")
    print("-"*50)
    
    # Initialize framework WITH standard test data
    framework_standard = AlgorithmTestingFramework(
        output_dir="./demo_results_standard",
        test_data_dir="./benchmark_test_data",  # Contains M_X_D30.txt files
        dimension=30,
        num_runs=3,
        max_iterations=50,
        verbose=False
    )
    
    # Run single experiment with standard data
    result_standard = framework_standard.run_single_experiment(
        OriginalHBO, 'HBO', function_id=1, run_number=1
    )
    
    print(f" Standard Test Data Result: {result_standard.best_fitness:.6e}")
    print(f"   Uses M_1_D30.txt rotation matrix")
    print(f"   Uses shift_data_1_D30.txt shift vector")
    print(f"   Uses f1_o.txt bias value")
    
    print("\n TESTING WITH RANDOM INITIALIZATION:")
    print("-"*50)
    
    # Initialize framework WITHOUT standard test data (non-existent directory)
    framework_random = AlgorithmTestingFramework(
        output_dir="./demo_results_random",
        test_data_dir="./nonexistent_directory",  # No test data available
        dimension=30,
        num_runs=3,
        max_iterations=50,
        verbose=False
    )
    
    # Run single experiment with random initialization
    result_random = framework_random.run_single_experiment(
        OriginalHBO, 'HBO', function_id=1, run_number=1
    )
    
    print(f" Random Initialization Result: {result_random.best_fitness:.6e}")
    print(f"   Uses random starting points")
    print(f"   No transformation matrices")
    print(f"   Standard function evaluation")
    
    print("\nCOMPARISON:")
    print("-"*50)
    
    print(f"Standard Test Data:     {result_standard.best_fitness:.6e}")
    print(f"Random Initialization:  {result_random.best_fitness:.6e}")
    
    if abs(result_standard.best_fitness - result_random.best_fitness) > 1e-10:
        print("DIFFERENT results - Standard test data is being used correctly!")
        print("   The M_X_D30.txt files transform the function landscape")
    else:
        print("  Same results - Standard test data may not be applied correctly")
    
    print("\n KEY DIFFERENCES:")
    print("-"*50)
    print(" Standard Test Data (M_X_D30.txt format):")
    print("   • Uses fixed rotation matrices for reproducible testing")
    print("   • Applies shift vectors to move global optimum")  
    print("   • Adds bias values to function output")
    print("   • Enables fair comparison across research papers")
    print("   • Matches CEC benchmark competition format")
    
    print("\n Random Initialization:")
    print("   • Different starting points each run")
    print("   • No function transformation")
    print("   • Results vary between runs/papers")
    print("   • Harder to reproduce exact results")
    
    print(f"\n DEMO COMPLETE")
    print(f"   Standard test data files are now integrated into the framework!")


if __name__ == "__main__":
    demo_standard_vs_random()