"""
Multi-Agent Evaluation Script

Comprehensive evaluation of trained multi-agent coverage systems.

Usage:
    python evaluate_multi_agent.py --checkpoint model.pth --episodes 50
    python evaluate_multi_agent.py --checkpoint model.pth --test-team-sizes 2,3,4,5
    python evaluate_multi_agent.py --checkpoint model.pth --test-all-strategies

Key Features:
    - Test across multiple team sizes (generalization)
    - Test across coordination strategies
    - Test across map types
    - Detailed performance analysis
    - Visualization of results
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from collections import defaultdict

from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
from multi_agent_trainer import MultiAgentTrainer
from multi_agent_config import ma_config


def evaluate_configuration(
    trainer: MultiAgentTrainer,
    num_agents: int,
    coordination: CoordinationStrategy,
    map_types: List[str],
    episodes_per_map: int = 10,
    grid_size: int = 20
) -> Dict:
    """
    Evaluate a specific configuration.

    Args:
        trainer: Trained multi-agent trainer
        num_agents: Number of agents to test
        coordination: Coordination strategy
        map_types: List of map types to test
        episodes_per_map: Episodes per map type
        grid_size: Grid size

    Returns:
        results: Dict with evaluation results
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING: {num_agents} agents, {coordination.value} coordination")
    print(f"{'='*70}")

    # Create environment
    env = MultiAgentCoverageEnv(
        num_agents=num_agents,
        grid_size=grid_size,
        coordination=coordination
    )

    # Store original num_agents
    original_num_agents = trainer.num_agents

    # Update trainer for this team size (hacky but works for evaluation)
    if num_agents != original_num_agents:
        print(f"⚠ Warning: Trainer trained with {original_num_agents} agents, "
              f"testing with {num_agents} agents (parameter sharing)")

    results = {
        'num_agents': num_agents,
        'coordination': coordination.value,
        'coverages': [],
        'team_rewards': [],
        'lengths': [],
        'collisions': [],
        'agent_collisions': [],
        'per_map': defaultdict(lambda: {
            'coverages': [],
            'rewards': [],
            'lengths': []
        })
    }

    total_episodes = len(map_types) * episodes_per_map

    for map_idx, map_type in enumerate(map_types):
        print(f"\n  Testing {map_type} maps...")

        for ep in range(episodes_per_map):
            # Reset
            state = env.reset(map_type=map_type)
            observations = env.get_observations()

            team_reward = 0.0
            step_count = 0
            episode_collisions = 0
            episode_agent_collisions = 0
            done = False

            # Run episode (greedy)
            while not done:
                actions = trainer.select_actions(observations, epsilon=0.0)
                next_state, rewards, done, info = env.step(actions)
                next_observations = env.get_observations()

                team_reward += sum(rewards)
                episode_collisions += sum(info['collisions'])
                episode_agent_collisions += sum(info['agent_collisions'])
                step_count += 1

                observations = next_observations

            final_coverage = info['coverage_pct']

            # Store results
            results['coverages'].append(final_coverage)
            results['team_rewards'].append(team_reward)
            results['lengths'].append(step_count)
            results['collisions'].append(episode_collisions)
            results['agent_collisions'].append(episode_agent_collisions)

            results['per_map'][map_type]['coverages'].append(final_coverage)
            results['per_map'][map_type]['rewards'].append(team_reward)
            results['per_map'][map_type]['lengths'].append(step_count)

            # Progress
            episode_num = map_idx * episodes_per_map + ep + 1
            if episode_num % 5 == 0 or episode_num == total_episodes:
                print(f"    [{episode_num}/{total_episodes}] "
                      f"Coverage: {final_coverage*100:.1f}%, "
                      f"Reward: {team_reward:.1f}")

    # Compute statistics
    print(f"\n  Results Summary:")
    print(f"    Mean Coverage: {np.mean(results['coverages'])*100:.1f}% "
          f"(±{np.std(results['coverages'])*100:.1f}%)")
    print(f"    Mean Team Reward: {np.mean(results['team_rewards']):.1f}")
    print(f"    Mean Length: {np.mean(results['lengths']):.1f}")
    print(f"    Mean Collisions: {np.mean(results['collisions']):.1f} "
          f"({np.mean(results['agent_collisions']):.1f} agent-agent)")

    print(f"\n  Per-Map Coverage:")
    for map_type, map_results in results['per_map'].items():
        mean_cov = np.mean(map_results['coverages'])
        std_cov = np.std(map_results['coverages'])
        print(f"    {map_type:12s}: {mean_cov*100:.1f}% (±{std_cov*100:.1f}%)")

    return results


def evaluate_all_configurations(
    trainer: MultiAgentTrainer,
    team_sizes: List[int],
    coordination_strategies: List[CoordinationStrategy],
    map_types: List[str],
    episodes_per_map: int = 10,
    grid_size: int = 20
) -> Dict:
    """
    Comprehensive evaluation across all configurations.

    Args:
        trainer: Trained multi-agent trainer
        team_sizes: List of team sizes to test
        coordination_strategies: List of strategies to test
        map_types: List of map types to test
        episodes_per_map: Episodes per map type
        grid_size: Grid size

    Returns:
        all_results: Dict with all evaluation results
    """
    print(f"\n{'='*70}")
    print(f"COMPREHENSIVE MULTI-AGENT EVALUATION")
    print(f"{'='*70}")
    print(f"Team Sizes: {team_sizes}")
    print(f"Coordination Strategies: {[s.value for s in coordination_strategies]}")
    print(f"Map Types: {map_types}")
    print(f"Episodes per Map: {episodes_per_map}")
    print(f"{'='*70}")

    all_results = []

    for num_agents in team_sizes:
        for coordination in coordination_strategies:
            results = evaluate_configuration(
                trainer=trainer,
                num_agents=num_agents,
                coordination=coordination,
                map_types=map_types,
                episodes_per_map=episodes_per_map,
                grid_size=grid_size
            )
            all_results.append(results)

    return all_results


def plot_evaluation_results(all_results: List[Dict], output_dir: str):
    """
    Plot comprehensive evaluation results.

    Args:
        all_results: List of result dicts
        output_dir: Output directory for plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Extract data
    team_sizes = sorted(set(r['num_agents'] for r in all_results))
    strategies = sorted(set(r['coordination'] for r in all_results))

    # Plot 1: Coverage by Team Size and Strategy
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in strategies:
        strategy_results = [r for r in all_results if r['coordination'] == strategy]

        sizes = []
        means = []
        stds = []

        for size in team_sizes:
            size_results = [r for r in strategy_results if r['num_agents'] == size]
            if len(size_results) > 0:
                coverages = size_results[0]['coverages']
                sizes.append(size)
                means.append(np.mean(coverages) * 100)
                stds.append(np.std(coverages) * 100)

        ax.errorbar(sizes, means, yerr=stds, marker='o', label=strategy, capsize=5)

    ax.set_xlabel('Team Size (Number of Agents)')
    ax.set_ylabel('Coverage (%)')
    ax.set_title('Coverage Performance by Team Size and Coordination Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_by_team_size.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/coverage_by_team_size.png")
    plt.close()

    # Plot 2: Per-Map Coverage Comparison
    map_types = list(all_results[0]['per_map'].keys())

    fig, axes = plt.subplots(1, len(strategies), figsize=(5*len(strategies), 5))
    if len(strategies) == 1:
        axes = [axes]

    for ax, strategy in zip(axes, strategies):
        strategy_results = [r for r in all_results if r['coordination'] == strategy]

        # Group by team size
        x_pos = np.arange(len(map_types))
        width = 0.8 / len(team_sizes)

        for i, size in enumerate(team_sizes):
            size_results = [r for r in strategy_results if r['num_agents'] == size]
            if len(size_results) == 0:
                continue

            map_coverages = []
            for map_type in map_types:
                coverages = size_results[0]['per_map'][map_type]['coverages']
                map_coverages.append(np.mean(coverages) * 100)

            offset = (i - len(team_sizes)/2 + 0.5) * width
            ax.bar(x_pos + offset, map_coverages, width, label=f'{size} agents')

        ax.set_xlabel('Map Type')
        ax.set_ylabel('Coverage (%)')
        ax.set_title(f'{strategy.capitalize()} Coordination')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(map_types, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'coverage_by_map_type.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/coverage_by_map_type.png")
    plt.close()

    # Plot 3: Collision Analysis
    fig, ax = plt.subplots(figsize=(10, 6))

    for strategy in strategies:
        strategy_results = [r for r in all_results if r['coordination'] == strategy]

        sizes = []
        collision_rates = []

        for size in team_sizes:
            size_results = [r for r in strategy_results if r['num_agents'] == size]
            if len(size_results) > 0:
                collisions = size_results[0]['agent_collisions']
                lengths = size_results[0]['lengths']
                rate = np.mean(np.array(collisions) / np.array(lengths))
                sizes.append(size)
                collision_rates.append(rate)

        ax.plot(sizes, collision_rates, marker='o', label=strategy)

    ax.set_xlabel('Team Size (Number of Agents)')
    ax.set_ylabel('Agent-Agent Collision Rate')
    ax.set_title('Collision Rate by Team Size and Coordination Strategy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'collision_analysis.png'), dpi=150)
    print(f"✓ Saved: {output_dir}/collision_analysis.png")
    plt.close()


def print_summary_table(all_results: List[Dict]):
    """Print summary table of all results."""
    print(f"\n{'='*70}")
    print(f"EVALUATION SUMMARY TABLE")
    print(f"{'='*70}")

    print(f"\n{'Team Size':<12} {'Strategy':<15} {'Coverage':<15} {'Reward':<12} {'Collisions':<12}")
    print(f"{'-'*70}")

    for result in all_results:
        team_size = result['num_agents']
        strategy = result['coordination']
        mean_cov = np.mean(result['coverages']) * 100
        std_cov = np.std(result['coverages']) * 100
        mean_reward = np.mean(result['team_rewards'])
        mean_coll = np.mean(result['agent_collisions'])

        print(f"{team_size:<12} {strategy:<15} "
              f"{mean_cov:5.1f}% ±{std_cov:4.1f}%  "
              f"{mean_reward:9.1f}   {mean_coll:6.1f}")

    print(f"{'='*70}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained multi-agent coverage system"
    )

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )

    parser.add_argument(
        '--episodes',
        type=int,
        default=10,
        help='Episodes per configuration'
    )

    parser.add_argument(
        '--test-team-sizes',
        type=str,
        default='2,3,4,5',
        help='Comma-separated team sizes to test (e.g., "2,3,4,5")'
    )

    parser.add_argument(
        '--test-strategies',
        type=str,
        default='independent,voronoi,market',
        help='Comma-separated strategies to test'
    )

    parser.add_argument(
        '--test-maps',
        type=str,
        default='empty,random,maze',
        help='Comma-separated map types to test'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='multi_agent_results/evaluation',
        help='Output directory for results'
    )

    args = parser.parse_args()

    # Parse arguments
    team_sizes = [int(x) for x in args.test_team_sizes.split(',')]
    map_types = args.test_maps.split(',')

    strategy_map = {
        'independent': CoordinationStrategy.INDEPENDENT,
        'voronoi': CoordinationStrategy.VORONOI,
        'market': CoordinationStrategy.MARKET,
        'hierarchical': CoordinationStrategy.HIERARCHICAL
    }
    strategies = [strategy_map[s] for s in args.test_strategies.split(',')]

    # Load trainer
    print(f"\n{'='*70}")
    print(f"LOADING CHECKPOINT")
    print(f"{'='*70}")
    print(f"Checkpoint: {args.checkpoint}")

    # Create dummy trainer to load checkpoint
    trainer = MultiAgentTrainer(
        num_agents=4,  # Will be overridden by checkpoint
        grid_size=ma_config.GRID_SIZE,
        coordination=CoordinationStrategy.INDEPENDENT,
        parameter_sharing=True,
        shared_replay=True
    )

    trainer.load(args.checkpoint)
    print(f"✓ Checkpoint loaded\n")

    # Evaluate
    all_results = evaluate_all_configurations(
        trainer=trainer,
        team_sizes=team_sizes,
        coordination_strategies=strategies,
        map_types=map_types,
        episodes_per_map=args.episodes,
        grid_size=ma_config.GRID_SIZE
    )

    # Print summary
    print_summary_table(all_results)

    # Plot results
    print(f"\n{'='*70}")
    print(f"GENERATING PLOTS")
    print(f"{'='*70}")

    plot_evaluation_results(all_results, args.output_dir)

    print(f"\n✓ Evaluation complete!")
    print(f"  Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
