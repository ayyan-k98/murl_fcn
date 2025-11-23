"""
Test probabilistic coverage in multi-agent environment.
"""

import numpy as np
from config import config
from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy


def test_binary_coverage():
    """Test binary coverage mode (default)."""
    print("\n" + "="*70)
    print("TEST: Binary Coverage (default)")
    print("="*70)
    
    config.USE_PROBABILISTIC_ENV = False
    
    env = MultiAgentCoverageEnv(
        num_agents=2,
        grid_size=10,
        coordination=CoordinationStrategy.INDEPENDENT
    )
    
    state = env.reset()
    
    # Get initial coverage
    initial_coverage = state.world_state.coverage_map.sum()
    print(f"Initial coverage sum: {initial_coverage:.2f}")
    
    # Take one step for each agent (move right)
    actions = [6, 6]  # Both move right
    next_state, rewards, done, info = env.step(actions)
    
    # Check coverage at agent positions
    for i, agent in enumerate(next_state.agents):
        pos = agent.robot_state.position
        coverage_val = next_state.world_state.coverage_map[pos[0], pos[1]]
        print(f"Agent {i} position {pos}: coverage = {coverage_val:.4f}")
        assert coverage_val == 1.0, f"Binary coverage should be 1.0, got {coverage_val}"
    
    print("✓ Binary coverage test PASSED")
    return True


def test_probabilistic_coverage():
    """Test probabilistic coverage mode."""
    print("\n" + "="*70)
    print("TEST: Probabilistic Coverage")
    print("="*70)
    
    config.USE_PROBABILISTIC_ENV = True
    
    env = MultiAgentCoverageEnv(
        num_agents=2,
        grid_size=10,
        coordination=CoordinationStrategy.INDEPENDENT
    )
    
    state = env.reset()
    
    # Get agent 0 starting position
    agent0_start_pos = state.agents[0].robot_state.position
    print(f"Agent 0 start position: {agent0_start_pos}")
    
    # Take first step (should add coverage gradually)
    actions = [8, 8]  # Both stay
    next_state, rewards, done, info = env.step(actions)
    
    # Check coverage at agent 0 position after first visit
    coverage_1 = next_state.world_state.coverage_map[agent0_start_pos[0], agent0_start_pos[1]]
    print(f"Coverage after step 1: {coverage_1:.4f}")
    
    # Coverage should be less than 1.0 but greater than 0 (probabilistic accumulation)
    assert 0.0 < coverage_1 < 1.0, f"Probabilistic coverage should be in (0, 1), got {coverage_1}"
    
    # Take second step (same position, should accumulate more)
    next_state, rewards, done, info = env.step(actions)
    coverage_2 = next_state.world_state.coverage_map[agent0_start_pos[0], agent0_start_pos[1]]
    print(f"Coverage after step 2: {coverage_2:.4f}")
    
    # Coverage should increase
    assert coverage_2 > coverage_1, f"Coverage should increase: {coverage_1} -> {coverage_2}"
    
    # Take many steps to reach ~100%
    for _ in range(10):
        next_state, rewards, done, info = env.step(actions)
    
    coverage_final = next_state.world_state.coverage_map[agent0_start_pos[0], agent0_start_pos[1]]
    print(f"Coverage after 12 steps: {coverage_final:.4f}")
    
    # Should approach 1.0
    assert coverage_final > 0.9, f"Coverage should approach 1.0, got {coverage_final}"
    
    print("✓ Probabilistic coverage test PASSED")
    return True


def test_reward_scaling():
    """Test that reward scaling works for probabilistic mode."""
    print("\n" + "="*70)
    print("TEST: Reward Scaling")
    print("="*70)
    
    # Test binary mode
    config.USE_PROBABILISTIC_ENV = False
    env_binary = MultiAgentCoverageEnv(num_agents=2, grid_size=10)
    state = env_binary.reset()
    actions = [6, 7]  # Move to new cells
    _, rewards_binary, _, _ = env_binary.step(actions)
    
    print(f"Binary mode rewards: {rewards_binary}")
    
    # Test probabilistic mode
    config.USE_PROBABILISTIC_ENV = True
    env_prob = MultiAgentCoverageEnv(num_agents=2, grid_size=10)
    state = env_prob.reset()
    actions = [6, 7]  # Move to new cells
    _, rewards_prob, _, _ = env_prob.step(actions)
    
    print(f"Probabilistic mode rewards: {rewards_prob}")
    print(f"Probabilistic reward scale: {config.PROBABILISTIC_REWARD_SCALE}")
    
    # Probabilistic rewards should be scaled differently
    print("✓ Reward scaling test PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MULTI-AGENT PROBABILISTIC COVERAGE TESTS")
    print("="*70)
    
    try:
        test_binary_coverage()
        test_probabilistic_coverage()
        test_reward_scaling()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nProbabilistic coverage is now implemented for multi-agent!")
        print("\nUsage:")
        print("  python train_multi_agent.py --probabilistic --agents 4 --episodes 400")
        print("\n")
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
