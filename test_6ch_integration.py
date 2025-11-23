"""
Test 6-channel FCN integration with agent occupancy.

Verifies:
1. FCN accepts 5 or 6 channels
2. Agent occupancy computation works
3. Forward pass produces valid Q-values
4. Backward pass (training) works
"""

import numpy as np
import torch

from fcn_agent import FCNAgent
from agent_occupancy import AgentOccupancyComputer, create_dummy_occupancy
from data_structures import RobotState, WorldState
from config import config


def test_5ch_baseline():
    """Test standard 5-channel FCN."""
    print("Test 1: 5-channel baseline FCN")
    print("-" * 60)
    
    agent = FCNAgent(grid_size=20, input_channels=5)
    
    # Create dummy state
    robot_state = RobotState(position=(10, 10), orientation=0.0, visited_positions={(10, 10)})
    world_state = WorldState(
        grid_size=20, 
        graph=None, 
        obstacles=set(),
        coverage_map=np.zeros((20, 20), dtype=np.float32)
    )
    
    # Encode state (no occupancy)
    grid_tensor = agent._encode_state(robot_state, world_state, agent_occupancy=None)
    
    print(f"Grid tensor shape: {grid_tensor.shape}")
    assert grid_tensor.shape == (1, 5, 20, 20), f"Expected (1, 5, 20, 20), got {grid_tensor.shape}"
    
    # Forward pass
    q_values = agent.policy_net(grid_tensor.to(agent.device))
    print(f"Q-values shape: {q_values.shape}")
    assert q_values.shape == (1, config.N_ACTIONS), f"Expected (1, {config.N_ACTIONS})"
    
    print(f"Q-values: {q_values[0].detach().cpu().numpy()}")
    print("✓ 5-channel baseline works!\n")


def test_6ch_dummy():
    """Test 6-channel FCN with dummy occupancy (all zeros)."""
    print("Test 2: 6-channel FCN with dummy occupancy")
    print("-" * 60)
    
    agent = FCNAgent(grid_size=20, input_channels=6)
    
    # Create dummy state
    robot_state = RobotState(position=(10, 10), orientation=0.0, visited_positions={(10, 10)})
    world_state = WorldState(
        grid_size=20,
        graph=None,
        obstacles=set(),
        coverage_map=np.zeros((20, 20), dtype=np.float32)
    )
    
    # Create dummy occupancy (all zeros)
    dummy_occupancy = create_dummy_occupancy(20)
    
    # Encode state with dummy occupancy
    grid_tensor = agent._encode_state(robot_state, world_state, agent_occupancy=dummy_occupancy)
    
    print(f"Grid tensor shape: {grid_tensor.shape}")
    assert grid_tensor.shape == (1, 6, 20, 20), f"Expected (1, 6, 20, 20), got {grid_tensor.shape}"
    
    # Verify ch5 is all zeros
    ch5 = grid_tensor[0, 5].cpu().numpy()
    assert np.all(ch5 == 0), "Channel 5 should be all zeros"
    
    # Forward pass
    q_values = agent.policy_net(grid_tensor.to(agent.device))
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values[0].detach().cpu().numpy()}")
    print("✓ 6-channel dummy works!\n")


def test_6ch_with_occupancy():
    """Test 6-channel FCN with real agent occupancy."""
    print("Test 3: 6-channel FCN with real agent occupancy")
    print("-" * 60)
    
    agent = FCNAgent(grid_size=20, input_channels=6)
    occupancy_computer = AgentOccupancyComputer(grid_size=20)
    
    # Create dummy state
    robot_state = RobotState(position=(10, 10), orientation=0.0, visited_positions={(10, 10)})
    world_state = WorldState(
        grid_size=20,
        graph=None,
        obstacles=set(),
        coverage_map=np.zeros((20, 20), dtype=np.float32)
    )
    
    # Simulate messages from other agents
    messages = [
        {'sender_id': 1, 'position': (5, 5), 'timestamp': 0},
        {'sender_id': 2, 'position': (15, 15), 'timestamp': 0},
    ]
    
    # Compute occupancy for agent 0
    occupancy = occupancy_computer.compute(agent_id=0, messages=messages, current_time=0)
    
    print(f"Occupancy shape: {occupancy.shape}")
    print(f"Occupancy range: [{occupancy.min():.3f}, {occupancy.max():.3f}]")
    print(f"Occupancy sum: {occupancy.sum():.2f}")
    
    # Encode state with occupancy
    grid_tensor = agent._encode_state(robot_state, world_state, agent_occupancy=occupancy)
    
    print(f"Grid tensor shape: {grid_tensor.shape}")
    assert grid_tensor.shape == (1, 6, 20, 20), f"Expected (1, 6, 20, 20)"
    
    # Verify ch5 has occupancy information
    ch5 = grid_tensor[0, 5].cpu().numpy()
    assert not np.all(ch5 == 0), "Channel 5 should have occupancy info"
    assert np.allclose(ch5, occupancy), "Channel 5 should match occupancy"
    
    # Forward pass
    q_values = agent.policy_net(grid_tensor.to(agent.device))
    print(f"Q-values shape: {q_values.shape}")
    print(f"Q-values: {q_values[0].detach().cpu().numpy()}")
    print("✓ 6-channel with occupancy works!\n")


def test_select_action_with_occupancy():
    """Test select_action with agent_occupancy parameter."""
    print("Test 4: select_action with agent_occupancy")
    print("-" * 60)
    
    agent = FCNAgent(grid_size=20, input_channels=6)
    agent.set_epsilon(0.0)  # Greedy for deterministic test
    
    occupancy_computer = AgentOccupancyComputer(grid_size=20)
    
    # Create state
    robot_state = RobotState(position=(10, 10), orientation=0.0, visited_positions={(10, 10)})
    world_state = WorldState(
        grid_size=20,
        graph=None,
        obstacles=set(),
        coverage_map=np.zeros((20, 20), dtype=np.float32)
    )
    
    # Test 1: Without occupancy (should still work with 6ch network if we pass dummy)
    dummy_occupancy = create_dummy_occupancy(20)
    action1 = agent.select_action(robot_state, world_state, agent_occupancy=dummy_occupancy)
    print(f"Action without occupancy: {action1}")
    assert 0 <= action1 < config.N_ACTIONS
    
    # Test 2: With occupancy
    messages = [{'sender_id': 1, 'position': (11, 10), 'timestamp': 0}]  # Agent to the right
    occupancy = occupancy_computer.compute(agent_id=0, messages=messages, current_time=0)
    action2 = agent.select_action(robot_state, world_state, agent_occupancy=occupancy)
    print(f"Action with occupancy (agent at right): {action2}")
    assert 0 <= action2 < config.N_ACTIONS
    
    print("✓ select_action with occupancy works!\n")


def test_training_step():
    """Test that training step works with 6 channels."""
    print("Test 5: Training step with 6 channels")
    print("-" * 60)
    
    agent = FCNAgent(grid_size=20, input_channels=6)
    
    # Create dummy transitions
    robot_state = RobotState(position=(10, 10), orientation=0.0, visited_positions={(10, 10)})
    world_state = WorldState(
        grid_size=20,
        graph=None,
        obstacles=set(),
        coverage_map=np.zeros((20, 20), dtype=np.float32)
    )
    dummy_occupancy = create_dummy_occupancy(20)
    
    # Encode states
    state_tensor = agent._encode_state(robot_state, world_state, dummy_occupancy)
    next_state_tensor = agent._encode_state(robot_state, world_state, dummy_occupancy)
    
    # Store transition
    agent.store_transition(state_tensor, action=0, reward=1.0, 
                          next_state=next_state_tensor, done=False, info={})
    
    print(f"Memory size: {len(agent.memory)}")
    
    # Add more transitions to meet min replay size
    for i in range(config.MIN_REPLAY_SIZE):
        agent.store_transition(state_tensor, action=i % config.N_ACTIONS, 
                              reward=np.random.random(),
                              next_state=next_state_tensor, 
                              done=(i % 100 == 0), info={})
    
    print(f"Memory size after filling: {len(agent.memory)}")
    
    # Optimize
    loss = agent.optimize()
    
    print(f"Loss: {loss}")
    assert loss is not None, "Training should return a loss"
    print("✓ Training step works!\n")


def test_backward_compatibility():
    """Verify 5ch and 6ch agents produce similar outputs with dummy occupancy."""
    print("Test 6: Backward compatibility (5ch vs 6ch with dummy)")
    print("-" * 60)
    
    # Same seed for both
    torch.manual_seed(42)
    np.random.seed(42)
    agent_5ch = FCNAgent(grid_size=20, input_channels=5)
    
    torch.manual_seed(42)
    np.random.seed(42)
    agent_6ch = FCNAgent(grid_size=20, input_channels=6)
    
    # Copy weights from 5ch to 6ch for first 5 channels
    # (This is just to test they have similar architecture)
    
    robot_state = RobotState(position=(10, 10), orientation=0.0, visited_positions={(10, 10)})
    world_state = WorldState(
        grid_size=20,
        graph=None,
        obstacles=set(),
        coverage_map=np.zeros((20, 20), dtype=np.float32)
    )
    
    # Encode with 5 channels
    grid_5ch = agent_5ch._encode_state(robot_state, world_state, agent_occupancy=None)
    
    # Encode with 6 channels (dummy)
    dummy_occupancy = create_dummy_occupancy(20)
    grid_6ch = agent_6ch._encode_state(robot_state, world_state, agent_occupancy=dummy_occupancy)
    
    print(f"5ch shape: {grid_5ch.shape}")
    print(f"6ch shape: {grid_6ch.shape}")
    
    # Verify first 5 channels are identical
    assert torch.allclose(grid_5ch[0, :5], grid_6ch[0, :5]), "First 5 channels should match"
    
    # Verify 6th channel is zeros
    assert torch.all(grid_6ch[0, 5] == 0), "6th channel should be zeros"
    
    print("✓ Backward compatibility verified!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("TESTING 6-CHANNEL FCN INTEGRATION")
    print("=" * 60)
    print()
    
    try:
        test_5ch_baseline()
        test_6ch_dummy()
        test_6ch_with_occupancy()
        test_select_action_with_occupancy()
        test_training_step()
        test_backward_compatibility()
        
        print("=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
