"""
Multi-Agent Integration Tests

Comprehensive tests for the multi-agent coverage system.

Tests:
    1. Environment initialization and reset
    2. Multi-agent step execution
    3. Coordination strategies
    4. Trainer initialization
    5. Episode training
    6. Save/load functionality
    7. Validation
    8. Integration test
"""

import numpy as np
import torch
import tempfile
import os

from multi_agent_env import (
    MultiAgentCoverageEnv,
    CoordinationStrategy,
    AgentState,
    MultiAgentState
)
from multi_agent_trainer import MultiAgentTrainer
from multi_agent_config import ma_config


def test_environment_initialization():
    """Test 1: Environment initialization and reset."""
    print("\n" + "="*70)
    print("Test 1: Environment Initialization and Reset")
    print("="*70)

    # Test different team sizes
    for num_agents in [2, 4, 8]:
        env = MultiAgentCoverageEnv(
            num_agents=num_agents,
            grid_size=20,
            coordination=CoordinationStrategy.INDEPENDENT
        )

        state = env.reset()

        assert state.num_agents == num_agents, f"Expected {num_agents} agents, got {state.num_agents}"
        assert len(state.get_agent_positions()) == num_agents, "Position count mismatch"

        # Check agents are spatially distributed
        positions = state.get_agent_positions()
        for i, pos1 in enumerate(positions):
            for j, pos2 in enumerate(positions):
                if i != j:
                    assert pos1 != pos2, f"Agents {i} and {j} at same position {pos1}"

        print(f"  ✓ {num_agents} agents: Initialized correctly, positions unique")

    print("  ✓ Test 1 PASSED\n")


def test_multi_agent_step():
    """Test 2: Multi-agent step execution."""
    print("="*70)
    print("Test 2: Multi-Agent Step Execution")
    print("="*70)

    env = MultiAgentCoverageEnv(num_agents=4, grid_size=20)
    state = env.reset()

    # Execute 10 steps
    for step in range(10):
        actions = [np.random.randint(0, 9) for _ in range(4)]
        next_state, rewards, done, info = env.step(actions)

        # Check returns
        assert len(rewards) == 4, f"Expected 4 rewards, got {len(rewards)}"
        assert isinstance(done, bool), f"done should be bool, got {type(done)}"
        assert 'team_coverage_gain' in info, "Missing team_coverage_gain in info"
        assert 'collisions' in info, "Missing collisions in info"
        assert 'agent_collisions' in info, "Missing agent_collisions in info"

        # Check state update
        assert next_state.step_count == step + 1, f"Step count mismatch"

        state = next_state

    print(f"  ✓ 10 steps executed successfully")
    print(f"  ✓ Final coverage: {info['coverage_pct']*100:.1f}%")
    print(f"  ✓ Total collisions: {sum(info['collisions'])}")
    print("  ✓ Test 2 PASSED\n")


def test_coordination_strategies():
    """Test 3: Coordination strategies."""
    print("="*70)
    print("Test 3: Coordination Strategies")
    print("="*70)

    strategies = [
        CoordinationStrategy.INDEPENDENT,
        CoordinationStrategy.VORONOI,
        CoordinationStrategy.MARKET,
        CoordinationStrategy.HIERARCHICAL
    ]

    for strategy in strategies:
        env = MultiAgentCoverageEnv(
            num_agents=4,
            grid_size=20,
            coordination=strategy
        )

        state = env.reset()

        # Run 20 steps
        for _ in range(20):
            actions = [np.random.randint(0, 9) for _ in range(4)]
            state, rewards, done, info = env.step(actions)

            if done:
                break

        print(f"  ✓ {strategy.value:12s}: {info['coverage_pct']*100:5.1f}% coverage")

    print("  ✓ Test 3 PASSED\n")


def test_trainer_initialization():
    """Test 4: Trainer initialization."""
    print("="*70)
    print("Test 4: Trainer Initialization")
    print("="*70)

    # Test parameter sharing
    trainer_shared = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        parameter_sharing=True,
        shared_replay=True
    )

    assert trainer_shared.parameter_sharing == True
    assert trainer_shared.shared_replay == True
    assert len(set(id(agent) for agent in trainer_shared.agents)) == 1, "Agents should be same object"
    print("  ✓ Parameter sharing: Agents share same network")

    # Test independent networks
    trainer_independent = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        parameter_sharing=False,
        shared_replay=False
    )

    assert trainer_independent.parameter_sharing == False
    assert trainer_independent.shared_replay == False
    assert len(set(id(agent) for agent in trainer_independent.agents)) == 4, "Agents should be different"
    print("  ✓ Independent: Each agent has own network")

    print("  ✓ Test 4 PASSED\n")


def test_episode_training():
    """Test 5: Episode training."""
    print("="*70)
    print("Test 5: Episode Training")
    print("="*70)

    env = MultiAgentCoverageEnv(num_agents=4, grid_size=20)
    trainer = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        parameter_sharing=True,
        shared_replay=True
    )

    # Train 3 episodes
    for ep in range(3):
        episode_info = trainer.train_episode(env, map_type='empty')

        assert 'team_reward' in episode_info
        assert 'team_coverage' in episode_info
        assert 'episode_length' in episode_info

        print(f"  Episode {ep+1}: Reward={episode_info['team_reward']:6.1f}, "
              f"Coverage={episode_info['team_coverage']*100:5.1f}%, "
              f"Length={episode_info['episode_length']}")

    # Check metrics tracked
    assert len(trainer.metrics['team_rewards']) == 3
    assert len(trainer.metrics['team_coverages']) == 3

    print("  ✓ Test 5 PASSED\n")


def test_save_load():
    """Test 6: Save/load functionality."""
    print("="*70)
    print("Test 6: Save/Load Functionality")
    print("="*70)

    # Create and train trainer
    env = MultiAgentCoverageEnv(num_agents=4, grid_size=20)
    trainer1 = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        parameter_sharing=True
    )

    # Train 1 episode
    trainer1.train_episode(env, map_type='empty')

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'test_checkpoint.pth')
        trainer1.save(save_path)

        assert os.path.exists(save_path), "Checkpoint file not created"
        print(f"  ✓ Saved checkpoint to {save_path}")

        # Load into new trainer
        trainer2 = MultiAgentTrainer(
            num_agents=4,
            grid_size=20,
            parameter_sharing=True
        )

        trainer2.load(save_path)
        print(f"  ✓ Loaded checkpoint successfully")

        # Compare network parameters
        params1 = list(trainer1.agents[0].policy_net.parameters())
        params2 = list(trainer2.agents[0].policy_net.parameters())

        assert len(params1) == len(params2), "Parameter count mismatch"

        for p1, p2 in zip(params1, params2):
            assert torch.allclose(p1, p2), "Parameters don't match after load"

        print(f"  ✓ Network parameters match after load")

    print("  ✓ Test 6 PASSED\n")


def test_validation():
    """Test 7: Validation."""
    print("="*70)
    print("Test 7: Validation")
    print("="*70)

    env = MultiAgentCoverageEnv(num_agents=4, grid_size=20)
    trainer = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        parameter_sharing=True
    )

    # Train a few episodes
    for _ in range(3):
        trainer.train_episode(env, map_type='empty')

    # Run validation
    val_results = trainer.validate(
        env,
        num_episodes=5,
        map_types=['empty', 'random']
    )

    assert 'mean_coverage' in val_results
    assert 'std_coverage' in val_results
    assert 'mean_team_reward' in val_results
    assert 'per_map_coverage' in val_results

    print(f"  Mean Coverage: {val_results['mean_coverage']*100:.1f}%")
    print(f"  Std Coverage: {val_results['std_coverage']*100:.1f}%")
    print(f"  Mean Team Reward: {val_results['mean_team_reward']:.1f}")

    for map_type, coverage in val_results['per_map_coverage'].items():
        print(f"  {map_type:10s}: {coverage*100:.1f}%")

    print("  ✓ Test 7 PASSED\n")


def test_integration():
    """Test 8: Full integration test."""
    print("="*70)
    print("Test 8: Full Integration Test")
    print("="*70)

    # Test complete workflow
    print("  1. Creating environment and trainer...")
    env = MultiAgentCoverageEnv(
        num_agents=4,
        grid_size=20,
        coordination=CoordinationStrategy.VORONOI
    )

    trainer = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        coordination=CoordinationStrategy.VORONOI,
        parameter_sharing=True,
        shared_replay=True
    )

    print("  2. Training 5 episodes...")
    for ep in range(5):
        episode_info = trainer.train_episode(env, map_type='empty')
        if (ep + 1) % 2 == 0:
            print(f"     Episode {ep+1}: Coverage={episode_info['team_coverage']*100:.1f}%")

    print("  3. Running validation...")
    val_results = trainer.validate(env, num_episodes=3)
    print(f"     Validation Coverage: {val_results['mean_coverage']*100:.1f}%")

    print("  4. Saving checkpoint...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'integration_test.pth')
        trainer.save(save_path)

        print("  5. Loading checkpoint...")
        new_trainer = MultiAgentTrainer(
            num_agents=4,
            grid_size=20,
            coordination=CoordinationStrategy.VORONOI,
            parameter_sharing=True
        )
        new_trainer.load(save_path)

        print("  6. Testing loaded trainer...")
        val_results2 = new_trainer.validate(env, num_episodes=3)
        print(f"     Loaded Trainer Coverage: {val_results2['mean_coverage']*100:.1f}%")

        # Results should be very similar (greedy policy)
        assert abs(val_results['mean_coverage'] - val_results2['mean_coverage']) < 0.1, \
            "Loaded trainer performance differs significantly"

    print("  ✓ Test 8 PASSED\n")


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "="*70)
    print("MULTI-AGENT INTEGRATION TESTS")
    print("="*70)

    try:
        test_environment_initialization()
        test_multi_agent_step()
        test_coordination_strategies()
        test_trainer_initialization()
        test_episode_training()
        test_save_load()
        test_validation()
        test_integration()

        print("="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\n✅ Multi-agent system is fully functional!")

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
