#!/usr/bin/env python3
"""
Quick demo of the categorical cut-cell system
"""

import numpy as np
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from verified_simulator import VerifiedCategoricalSimulator
    
    print("🚀 Starting Categorical Cut-Cell Demo")
    print("This demo will run a simplified simulation without Lean verification")
    
    # Create simulator
    sim = VerifiedCategoricalSimulator(resolution=30)
    
    # Run shorter simulation for demo
    print("\nRunning demo simulation (30 steps)...")
    
    # We'll manually run a few steps to demonstrate
    for step in range(30):
        # Simple evolution without full verification
        states = [cell.value for cell in sim.complex.cell_states]
        new_states = states.copy()
        
        # Simple diffusion
        for i in range(1, len(states)-1):
            new_states[i] = states[i] + 0.1 * (states[i+1] - 2*states[i] + states[i-1])
        
        for i, cell in enumerate(sim.complex.cell_states):
            cell.value = new_states[i]
        
        # Update knowledge graph occasionally
        if step % 5 == 0:
            sim.kg.update(sim.complex)
        
        if step % 10 == 0:
            print(f"  Step {step}: Pattern detection active")
    
    print("\n✅ Demo completed successfully!")
    print(f"Detected {len(sim.kg.patterns)} patterns in the simulation")
    
    # Test semantic search
    results = sim.search_patterns("gradient")
    print(f"Found {len(results)} patterns matching 'gradient'")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nTroubleshooting tips:")
    print("1. Make sure all Python files are in the same directory")
    print("2. Install required packages: pip install numpy")
    print("3. Check Python version (3.7+ required)")