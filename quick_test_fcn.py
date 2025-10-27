"""
Quick Diagnostic for FCN Agent

Train FCN for N episodes then immediately test learned policy.
Determines if agent is actually learning or just random.

Usage:
    python quick_test_fcn.py                    # 50 episodes (default)
    python quick_test_fcn.py --episodes 10      # Quick test
    python quick_test_fcn.py --episodes 200     # Full test
    python quick_test_fcn.py --checkpoint ./checkpoints/fcn_checkpoint_ep200.pt
"""

import argparse
import os
import time
import numpy as np
import torch

from config import config
from environment import CoverageEnvironment
from fcn_agent import FCNAgent


def train_fcn_quick(num_episodes: int = 50) -> FCNAgent:
    """
    Quick training for N episodes.

    Args:
        num_episodes: Number of episodes to train

    Returns:
        agent: Trained FCN agent
    """
    print("="*70)
    print("TRAINING FCN AGENT")
    print("="*70)
    print(f"Episodes: {num_episodes}")
    print(f"Grid size: 20")
    print(f"Device: {config.DEVICE}")
    print(f"Max steps: 350")
    print("="*70)

    agent = FCNAgent(grid_size=20)

    for episode in range(num_episodes):
        # Create empty grid environment
        env = CoverageEnvironment(grid_size=20, map_type='empty')
        state = env.reset()

        episode_reward = 0

        for step in range(350):
            # Encode and select action
            grid_tensor = agent._encode_state(state, env.world_state)
            action = agent.select_action_from_tensor(grid_tensor)

            # Step
            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            # Encode next state
            next_grid_tensor = agent._encode_state(next_state, env.world_state)

            # Store transition
            agent.store_transition(
                grid_tensor, action, reward, next_grid_tensor, done, info
            )

            # Optimize
            if step % config.TRAIN_FREQ == 0 and len(agent.memory) >= config.MIN_REPLAY_SIZE:
                agent.optimize()

            state = next_state

            if done:
                break

        # Decay epsilon
        agent.decay_epsilon(decay_rate=0.995)

        # Update target network periodically
        if (episode + 1) % 50 == 0:
            agent.update_target_network()

        # Progress
        if (episode + 1) % 10 == 0 or episode == num_episodes - 1:
            coverage = info.get('coverage_pct', 0.0)
            print(f"Ep {episode+1:3d}/{num_episodes} | "
                  f"Cov: {coverage:5.1%} | "
                  f"R: {episode_reward:7.1f} | "
                  f"ε: {agent.epsilon:.3f}")

    # Save checkpoint
    os.makedirs('./checkpoints', exist_ok=True)
    checkpoint_path = f'./checkpoints/fcn_quick_ep{num_episodes}.pt'
    agent.save(checkpoint_path)
    print(f"\n✓ Checkpoint saved: {checkpoint_path}")

    return agent


def test_random_baseline(num_episodes: int = 10) -> dict:
    """
    Test pure random policy (lower bound).

    Args:
        num_episodes: Number of test episodes

    Returns:
        results: Dictionary with coverage stats
    """
    print("\n" + "="*70)
    print("TEST 1: RANDOM BASELINE")
    print("="*70)

    coverages = []

    for ep in range(num_episodes):
        env = CoverageEnvironment(grid_size=20, map_type='empty')
        state = env.reset()

        for step in range(350):
            action = np.random.randint(0, 9)
            state, reward, done, info = env.step(action)

            if done:
                break

        coverage = info.get('coverage_pct', 0.0)
        coverages.append(coverage)

    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)

    print(f"Random policy: {mean_cov:.1%} ± {std_cov:.1%}")

    return {'mean': mean_cov, 'std': std_cov, 'coverages': coverages}


def test_greedy_policy(agent: FCNAgent, num_episodes: int = 10) -> dict:
    """
    Test learned greedy policy (ε=0) - THE KEY TEST.

    This shows if agent actually learned something.
    If greedy < random, agent learned NOTHING.

    Args:
        agent: FCN agent to test
        num_episodes: Number of test episodes

    Returns:
        results: Dictionary with coverage stats
    """
    print("\n" + "="*70)
    print("TEST 2: LEARNED GREEDY POLICY (ε=0)")
    print("="*70)

    # Save original epsilon
    original_epsilon = agent.epsilon

    # Pure greedy (NO exploration)
    agent.set_epsilon(0.0)

    coverages = []

    for ep in range(num_episodes):
        env = CoverageEnvironment(grid_size=20, map_type='empty')
        state = env.reset()

        for step in range(350):
            action = agent.select_action(state, env.world_state)
            state, reward, done, info = env.step(action)

            if done:
                break

        coverage = info.get('coverage_pct', 0.0)
        coverages.append(coverage)

    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)

    print(f"Greedy policy: {mean_cov:.1%} ± {std_cov:.1%}")

    # Restore epsilon
    agent.set_epsilon(original_epsilon)

    return {'mean': mean_cov, 'std': std_cov, 'coverages': coverages}


def test_with_exploration(agent: FCNAgent, num_episodes: int = 10, epsilon: float = 0.5) -> dict:
    """
    Test with exploration (should match training performance).

    Args:
        agent: FCN agent to test
        num_episodes: Number of test episodes
        epsilon: Exploration rate

    Returns:
        results: Dictionary with coverage stats
    """
    print("\n" + "="*70)
    print(f"TEST 3: WITH EXPLORATION (ε={epsilon})")
    print("="*70)

    # Save original epsilon
    original_epsilon = agent.epsilon

    # Set test epsilon
    agent.set_epsilon(epsilon)

    coverages = []

    for ep in range(num_episodes):
        env = CoverageEnvironment(grid_size=20, map_type='empty')
        state = env.reset()

        for step in range(350):
            action = agent.select_action(state, env.world_state)
            state, reward, done, info = env.step(action)

            if done:
                break

        coverage = info.get('coverage_pct', 0.0)
        coverages.append(coverage)

    mean_cov = np.mean(coverages)
    std_cov = np.std(coverages)

    print(f"Policy with ε={epsilon}: {mean_cov:.1%} ± {std_cov:.1%}")

    # Restore epsilon
    agent.set_epsilon(original_epsilon)

    return {'mean': mean_cov, 'std': std_cov, 'coverages': coverages}


def interpret_results(random_results: dict, greedy_results: dict, explore_results: dict, num_episodes: int):
    """
    Interpret diagnostic results and provide recommendation.

    Args:
        random_results: Random baseline results
        greedy_results: Greedy policy results
        explore_results: Exploration policy results
        num_episodes: Number of training episodes
    """
    print("\n" + "="*70)
    print("RESULTS & INTERPRETATION")
    print("="*70)

    random_mean = random_results['mean']
    greedy_mean = greedy_results['mean']
    explore_mean = explore_results['mean']

    print(f"\nAfter {num_episodes} episodes of training:")
    print(f"  Random baseline:       {random_mean:.1%}")
    print(f"  Greedy policy (ε=0):   {greedy_mean:.1%}  ← KEY METRIC")
    print(f"  With exploration:      {explore_mean:.1%}")

    # Analysis
    learning_benefit = greedy_mean - random_mean
    exploration_benefit = explore_mean - greedy_mean

    print(f"\nAnalysis:")
    print(f"  Learning benefit:     {learning_benefit:+.1%} (greedy vs random)")
    print(f"  Exploration benefit:  {exploration_benefit:+.1%}")

    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    if greedy_mean < random_mean * 0.8:
        # Greedy is 20% worse than random - COMPLETE FAILURE
        print(f"\n🔴 AGENT LEARNED NOTHING")
        print(f"\nGreedy policy ({greedy_mean:.1%}) is worse than random ({random_mean:.1%}).")
        print(f"After {num_episodes} episodes, there's no learning.")
        print(f"\nPossible reasons:")
        print(f"  • Network too small for task")
        print(f"  • Learning rate too high/low")
        print(f"  • Epsilon not decaying properly")
        print(f"  • Reward signal too weak")
        print(f"\nRECOMMENDATION: 🔄 DEBUG HYPERPARAMETERS")
        print(f"  Check learning rate, network size, epsilon decay.")

    elif greedy_mean < random_mean * 1.1:
        # Greedy is less than 10% better than random - MINIMAL LEARNING
        print(f"\n🟡 MINIMAL LEARNING")
        print(f"\nGreedy policy ({greedy_mean:.1%}) is barely better than random ({random_mean:.1%}).")
        print(f"Agent learned something but very weak.")
        print(f"\nRECOMMENDATION: 🔧 NEEDS MORE TRAINING")
        print(f"  • Train for more episodes (try 200-400)")
        print(f"  • Check if epsilon is too high")
        print(f"  • Increase learning rate slightly")

    elif greedy_mean < random_mean * 1.25:
        # Greedy is 10-25% better - WEAK LEARNING
        print(f"\n🟠 WEAK BUT LEARNING")
        print(f"\nGreedy policy ({greedy_mean:.1%}) is somewhat better than random ({random_mean:.1%}).")
        print(f"Agent is learning but needs more training.")
        print(f"\nRECOMMENDATION: ✅ CONTINUE TRAINING")
        print(f"  • Train to 400-800 episodes")
        print(f"  • Should reach 50-65% validation")

    elif greedy_mean < random_mean * 1.5:
        # Greedy is 25-50% better - DECENT LEARNING
        print(f"\n🟢 DECENT LEARNING")
        print(f"\nGreedy policy ({greedy_mean:.1%}) is significantly better than random ({random_mean:.1%}).")
        print(f"Agent learned good strategies.")
        print(f"\nRECOMMENDATION: ✅ CONTINUE TRAINING")
        print(f"  • Train to 800-1000 episodes")
        print(f"  • Expected final: 60-70% validation")

    else:
        # Greedy is 50%+ better - EXCELLENT LEARNING
        print(f"\n🟢 EXCELLENT LEARNING")
        print(f"\nGreedy policy ({greedy_mean:.1%}) is much better than random ({random_mean:.1%}).")
        print(f"Agent learned strong strategies.")
        print(f"\nRECOMMENDATION: ✅ AGENT WORKING GREAT")
        print(f"  • Continue training to convergence")
        print(f"  • Expected final: 65-75% validation")

    print("="*70)


def main():
    """Main diagnostic function."""
    parser = argparse.ArgumentParser(
        description='Quick diagnostic: Train FCN then test if it learned'
    )
    parser.add_argument(
        '--episodes', type=int, default=50,
        help='Number of episodes to train (default: 50)'
    )
    parser.add_argument(
        '--test-episodes', type=int, default=10,
        help='Number of episodes per test (default: 10)'
    )
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Use existing checkpoint instead of training'
    )

    args = parser.parse_args()

    print("="*70)
    print("QUICK DIAGNOSTIC: FCN TRAIN + TEST")
    print("="*70)

    if args.checkpoint is None:
        # Mode: Train then test
        print(f"\nMode: Train then test")
        print(f"Training episodes: {args.episodes}")
        print(f"Test episodes: {args.test_episodes} per test")
        print(f"\nEstimated time: {args.episodes * 0.4:.1f} minutes")
        print()

        # Train
        start_time = time.time()
        agent = train_fcn_quick(num_episodes=args.episodes)
        train_time = time.time() - start_time

        print(f"\n✓ Training complete in {train_time/60:.1f} minutes")

    else:
        # Mode: Load checkpoint and test
        print(f"\nMode: Test existing checkpoint")
        print(f"Checkpoint: {args.checkpoint}")
        print(f"Test episodes: {args.test_episodes} per test")
        print()

        # Load agent
        agent = FCNAgent(grid_size=20)
        agent.load(args.checkpoint)
        print()

        # Extract episode count from filename if possible
        import re
        match = re.search(r'ep(\d+)', args.checkpoint)
        args.episodes = int(match.group(1)) if match else None

    # Run diagnostic tests
    print("\n" + "="*70)
    print("RUNNING DIAGNOSTIC TESTS")
    print("="*70)

    # Test 1: Random baseline
    random_results = test_random_baseline(num_episodes=args.test_episodes)

    # Test 2: Greedy policy (THE KEY TEST)
    greedy_results = test_greedy_policy(agent, num_episodes=args.test_episodes)

    # Test 3: With exploration
    explore_results = test_with_exploration(
        agent, num_episodes=args.test_episodes, epsilon=0.5
    )

    # Interpret
    interpret_results(
        random_results, greedy_results, explore_results,
        num_episodes=args.episodes if args.episodes else "?"
    )

    print("\n" + "="*70)
    print("DIAGNOSTIC COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
