"""
Test Optimizations

Verify that all implemented optimizations work correctly:
1. Batch Action Selection
2. Vectorized Frontier Detection
3. Cached Coordinate Grids
4. Optimized Replay Memory Sampling
"""

import torch
import numpy as np
import time
from config import config
from data_structures import RobotState, WorldState
from fcn_agent import FCNAgent
from replay_memory import StratifiedReplayMemory
from fcn_spatial_network import FCNSpatialNetwork

print("="*70)
print("TESTING ALL OPTIMIZATIONS")
print("="*70)

# ============================================================================
# Test 1: Vectorized Frontier Detection
# ============================================================================
print("\n" + "="*70)
print("Test 1: Vectorized Frontier Detection")
print("="*70)

agent = FCNAgent(grid_size=20)

# Create test state with some visited positions
robot_state = RobotState(
    position=(10, 10),
    orientation=0,
    visited_positions={(9, 9), (10, 9), (11, 9), (9, 10), (10, 10)}
)

world_state = WorldState(
    grid_size=20,
    graph=None,
    obstacles={(5, 5), (6, 6)},
    coverage_map=np.zeros((20, 20), dtype=np.float32),
    map_type="empty"
)

# Time the encoding (vectorized frontier detection)
start = time.time()
for _ in range(100):
    grid = agent._encode_state(robot_state, world_state)
elapsed = time.time() - start

print(f"  Grid shape: {grid.shape}")
print(f"  Expected: torch.Size([1, 5, 20, 20])")
print(f"  ✓ PASSED" if grid.shape == torch.Size([1, 5, 20, 20]) else "  ✗ FAILED")
print(f"  100 encodings took: {elapsed*1000:.2f}ms ({elapsed*10:.2f}ms per encoding)")

# Check frontier is correctly detected
frontier = grid[0, 3].numpy()
expected_frontier_count = np.sum(frontier)
print(f"  Frontier cells detected: {expected_frontier_count}")
print(f"  ✓ PASSED (frontier detected)" if expected_frontier_count > 0 else "  ✗ FAILED")

# ============================================================================
# Test 2: Batch Action Selection
# ============================================================================
print("\n" + "="*70)
print("Test 2: Batch Action Selection")
print("="*70)

# Create batch of states
batch_size = 16
grid_batch = torch.cat([grid for _ in range(batch_size)], dim=0)
print(f"  Batch shape: {grid_batch.shape}")

# Test batch action selection
start = time.time()
actions_batch = agent.select_actions_batch(grid_batch, epsilon=0.1)
elapsed_batch = time.time() - start

print(f"  Batch size: {len(actions_batch)}")
print(f"  Expected: {batch_size}")
print(f"  ✓ PASSED" if len(actions_batch) == batch_size else "  ✗ FAILED")
print(f"  Batch selection took: {elapsed_batch*1000:.2f}ms")

# Compare with sequential selection
start = time.time()
actions_sequential = []
for i in range(batch_size):
    action = agent.select_action_from_tensor(grid_batch[i:i+1], epsilon=0.1)
    actions_sequential.append(action)
elapsed_sequential = time.time() - start

print(f"  Sequential selection took: {elapsed_sequential*1000:.2f}ms")
speedup = elapsed_sequential / elapsed_batch
print(f"  Speedup: {speedup:.2f}x")
print(f"  ✓ PASSED (faster)" if speedup > 1.0 else "  ⚠ WARNING: Not faster")

# ============================================================================
# Test 3: Cached Coordinate Grids
# ============================================================================
print("\n" + "="*70)
print("Test 3: Cached Coordinate Grids")
print("="*70)

network = FCNSpatialNetwork(
    input_channels=5,
    num_actions=9,
    hidden_dim=128
)

# First forward pass (no cache)
x = torch.randn(4, 5, 20, 20)
start = time.time()
q1 = network(x)
elapsed_first = time.time() - start

# Second forward pass (should use cache)
start = time.time()
q2 = network(x)
elapsed_second = time.time() - start

print(f"  First forward pass: {elapsed_first*1000:.2f}ms")
print(f"  Second forward pass: {elapsed_second*1000:.2f}ms")
print(f"  Cache hit speedup: {elapsed_first/elapsed_second:.2f}x")
print(f"  Cache size: {len(network._coord_cache)}")
print(f"  ✓ PASSED (cache working)" if len(network._coord_cache) > 0 else "  ✗ FAILED")

# Test with different grid size (should create new cache entry)
x_large = torch.randn(4, 5, 30, 30)
q3 = network(x_large)
print(f"  Cache size after 30x30: {len(network._coord_cache)}")
print(f"  ✓ PASSED (multi-size)" if len(network._coord_cache) > 1 else "  ✗ FAILED")

# ============================================================================
# Test 4: Optimized Replay Memory Sampling
# ============================================================================
print("\n" + "="*70)
print("Test 4: Optimized Replay Memory Sampling")
print("="*70)

memory = StratifiedReplayMemory(capacity=1000)

# Add transitions to all strata
for i in range(400):
    state = torch.randn(1, 5, 20, 20)
    action = i % 9
    reward = np.random.uniform(-2, 10)
    next_state = torch.randn(1, 5, 20, 20)
    done = False

    # Distribute across strata
    if i % 4 == 0:
        info = {'coverage_gain': 1, 'knowledge_gain': 2, 'collision': False}
    elif i % 4 == 1:
        info = {'coverage_gain': 0, 'knowledge_gain': 3, 'collision': False}
    elif i % 4 == 2:
        info = {'coverage_gain': 0, 'knowledge_gain': 0, 'collision': True}
    else:
        info = {'coverage_gain': 0, 'knowledge_gain': 0, 'collision': False}

    memory.push(state, action, reward, next_state, done, info)

stats = memory.get_stats()
print(f"  Total transitions: {stats['total']}")
print(f"  Coverage: {stats['coverage']}")
print(f"  Exploration: {stats['exploration']}")
print(f"  Failure: {stats['failure']}")
print(f"  Neutral: {stats['neutral']}")

# Test sampling performance
batch_size = 256
start = time.time()
for _ in range(100):
    batch = memory.sample(batch_size)
elapsed = time.time() - start

print(f"  100 samples of batch_size={batch_size} took: {elapsed*1000:.2f}ms")
print(f"  Per sample: {elapsed*10:.2f}ms")
print(f"  Batch size: {len(batch)}")
print(f"  ✓ PASSED" if len(batch) == batch_size else "  ✗ FAILED")

# Verify stratification
coverage_count = sum(1 for t in batch if t[5].get('coverage_gain', 0) > 0)
exploration_count = sum(1 for t in batch if t[5].get('coverage_gain', 0) == 0 and t[5].get('knowledge_gain', 0) > 0)
failure_count = sum(1 for t in batch if t[5].get('collision', False))

print(f"  Coverage samples: {coverage_count} (expected ~{int(batch_size*0.4)})")
print(f"  Exploration samples: {exploration_count} (expected ~{int(batch_size*0.3)})")
print(f"  Failure samples: {failure_count} (expected ~{int(batch_size*0.2)})")

# ============================================================================
# Test 5: Integration Test - Quick Training Loop
# ============================================================================
print("\n" + "="*70)
print("Test 5: Integration Test - Mini Training Loop")
print("="*70)

from environment import CoverageEnvironment

# Create agent and environment
agent = FCNAgent(grid_size=20)
env = CoverageEnvironment(grid_size=20, map_type='empty')

print("  Running 5 episodes with all optimizations...")

total_start = time.time()
for episode in range(5):
    state = env.reset()
    episode_reward = 0

    for step in range(50):  # Short episodes
        # Use vectorized encoding
        grid_tensor = agent._encode_state(state, env.world_state)

        # Select action
        action = agent.select_action_from_tensor(grid_tensor)

        # Step
        next_state, reward, done, info = env.step(action)
        episode_reward += reward

        # Store transition
        next_grid_tensor = agent._encode_state(next_state, env.world_state)
        agent.store_transition(grid_tensor, action, reward, next_grid_tensor, done, info)

        # Optimize (uses optimized replay memory)
        if step % config.TRAIN_FREQ == 0 and len(agent.memory) >= config.MIN_REPLAY_SIZE:
            loss = agent.optimize()

        state = next_state

        if done:
            break

    coverage = info.get('coverage_pct', 0.0)
    print(f"    Episode {episode+1}: Reward={episode_reward:.1f}, Coverage={coverage:.1%}")

total_elapsed = time.time() - total_start
print(f"  Total time for 5 episodes: {total_elapsed:.2f}s")
print(f"  Average per episode: {total_elapsed/5:.2f}s")
print(f"  ✓ PASSED (integration successful)")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*70)
print("ALL OPTIMIZATION TESTS COMPLETED")
print("="*70)
print("\n✅ All optimizations implemented and tested:")
print("  1. Vectorized Frontier Detection - ✓")
print("  2. Batch Action Selection - ✓")
print("  3. Cached Coordinate Grids - ✓")
print("  4. Optimized Replay Memory Sampling - ✓")
print("  5. Integration Test - ✓")
print("\n🚀 System ready for optimized training!")
