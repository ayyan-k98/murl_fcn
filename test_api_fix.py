"""
Quick test to verify API fixes in fcn_agent.py
"""

import numpy as np
from data_structures import RobotState, WorldState
from fcn_agent import FCNAgent

print("="*70)
print("API FIX VALIDATION TEST")
print("="*70)

# Create agent
agent = FCNAgent(grid_size=20)
print("✓ FCN Agent created")

# Create test state with CORRECT attributes
robot_state = RobotState(
    position=(10, 10),
    orientation=0.0,
    visited_positions={(10, 10), (10, 11), (11, 10)}  # visited_positions, not visited_cells
)
print("✓ RobotState created with visited_positions")

world_state = WorldState(
    grid_size=20,
    graph=None,
    obstacles={(5, 5), (5, 6)},  # obstacles Set, not obstacle_map array
    coverage_map=np.zeros((20, 20), dtype=np.float32),
    map_type="empty"
)
print("✓ WorldState created with obstacles Set")

# Test encoding (this was failing before)
try:
    grid = agent._encode_state(robot_state, world_state)
    print(f"✓ State encoding successful: {grid.shape}")
    assert grid.shape == (1, 5, 20, 20), f"Expected (1,5,20,20), got {grid.shape}"
    print("✓ Shape correct: (1, 5, 20, 20)")
except AttributeError as e:
    print(f"✗ FAILED: {e}")
    exit(1)

# Test action selection
try:
    action = agent.select_action(robot_state, world_state, epsilon=0.0)
    print(f"✓ Action selection successful: action={action}")
    assert 0 <= action < 9, f"Invalid action: {action}"
    print("✓ Action valid: {0-8}")
except Exception as e:
    print(f"✗ FAILED: {e}")
    exit(1)

print("\n" + "="*70)
print("ALL TESTS PASSED - API FIX SUCCESSFUL!")
print("="*70)
print("\nYou can now run: python quick_test_fcn.py --episodes 50")
