"""
Multi-Agent Training Script

Train multi-robot coverage system with CTDE and curriculum learning.

Usage:
    python train_multi_agent.py --episodes 1000 --coordination market
    python train_multi_agent.py --episodes 1000 --agents 4 --parameter_sharing

Key Features:
    - CTDE (Centralized Training, Decentralized Execution)
    - 6-phase curriculum learning
    - Multiple coordination strategies
    - Validation across team sizes and strategies
    - Comprehensive logging and visualization
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime
from typing import Optional

from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
from multi_agent_trainer import MultiAgentTrainer
from multi_agent_config import ma_config
from config import config
from communication import get_communication_protocol
from agent_occupancy import AgentOccupancyComputer


def create_directories():
    """Create necessary directories for results."""
    os.makedirs(ma_config.VIS_DIR, exist_ok=True)
    os.makedirs(ma_config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ma_config.METRICS_DIR, exist_ok=True)
    print(f"✓ Created directories:")
    print(f"  {ma_config.VIS_DIR}")
    print(f"  {ma_config.CHECKPOINT_DIR}")
    print(f"  {ma_config.METRICS_DIR}")


def log_episode(episode: int, episode_info: dict, trainer: MultiAgentTrainer):
    """Log episode information."""
    team_reward = episode_info['team_reward']
    coverage = episode_info['team_coverage']
    length = episode_info['episode_length']
    collisions = episode_info['collisions']
    agent_collisions = episode_info['agent_collisions']
    epsilon = episode_info['epsilon']

    print(f"Ep {episode:4d} | "
          f"Reward: {team_reward:7.1f} | "
          f"Coverage: {coverage*100:5.1f}% | "
          f"Length: {length:3d} | "
          f"Collisions: {collisions:2d} ({agent_collisions} agent) | "
          f"ε: {epsilon:.3f}")


def validate_and_save(
    episode: int,
    trainer: MultiAgentTrainer,
    env: MultiAgentCoverageEnv,
    experiment_name: str
):
    """Run validation and save checkpoint."""
    print(f"\n{'='*70}")
    print(f"VALIDATION @ Episode {episode}")
    print(f"{'='*70}")

    # Validate on current configuration
    val_results = trainer.validate(
        env,
        num_episodes=ma_config.VALIDATION_EPISODES,
        map_types=ma_config.VALIDATION_MAP_TYPES
    )

    print(f"\nValidation Results:")
    print(f"  Mean Coverage: {val_results['mean_coverage']*100:.1f}% "
          f"(±{val_results['std_coverage']*100:.1f}%)")
    print(f"  Mean Team Reward: {val_results['mean_team_reward']:.1f}")
    print(f"  Mean Length: {val_results['mean_length']:.0f}")
    print(f"  Mean Collisions: {val_results['mean_collisions']:.1f}")

    print(f"\nPer-Map Coverage:")
    for map_type, coverage in val_results['per_map_coverage'].items():
        print(f"  {map_type:12s}: {coverage*100:.1f}%")

    # Save checkpoint
    checkpoint_path = os.path.join(
        ma_config.CHECKPOINT_DIR,
        f"{experiment_name}_ep{episode}.pth"
    )
    trainer.save(checkpoint_path)
    print(f"\n✓ Saved checkpoint: {checkpoint_path}")

    print(f"{'='*70}\n")

    return val_results


def train_multi_agent(
    num_agents: int = 4,
    total_episodes: int = 1000,
    coordination: CoordinationStrategy = CoordinationStrategy.INDEPENDENT,
    parameter_sharing: bool = True,
    shared_replay: bool = True,
    use_curriculum: bool = True,
    use_6ch: bool = False,
    comm_protocol: str = 'none',
    experiment_name: Optional[str] = None
):
    """
    Main multi-agent training loop.

    Args:
        num_agents: Number of agents
        total_episodes: Total training episodes
        coordination: Coordination strategy
        parameter_sharing: Use parameter sharing
        shared_replay: Use shared replay memory
        use_curriculum: Use curriculum learning
        use_6ch: Use 6-channel input (adds agent occupancy channel)
        comm_protocol: Communication protocol ('none', 'full_state', 'attention')
        experiment_name: Experiment name (auto-generated if None)
    """
    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ch_suffix = "6ch" if use_6ch else "5ch"
        comm_suffix = f"_{comm_protocol}" if comm_protocol != 'none' else ""
        experiment_name = f"ma{num_agents}_{coordination.value}_{ch_suffix}{comm_suffix}_{timestamp}"

    print(f"\n{'='*70}")
    print(f"MULTI-AGENT COVERAGE TRAINING")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Agents: {num_agents}")
    print(f"Input Channels: {6 if use_6ch else 5} ({'with agent occupancy' if use_6ch else 'baseline'})")
    print(f"Communication: {comm_protocol}")
    print(f"Coordination: {coordination.value}")
    print(f"Parameter Sharing: {parameter_sharing}")
    print(f"Shared Replay: {shared_replay}")
    print(f"Curriculum: {use_curriculum}")
    print(f"Total Episodes: {total_episodes}")
    print(f"{'='*70}\n")

    # Create directories
    create_directories()

    # Initialize environment
    env = MultiAgentCoverageEnv(
        num_agents=num_agents,
        grid_size=ma_config.GRID_SIZE,
        sensor_range=ma_config.SENSOR_RANGE,
        communication_range=ma_config.COMMUNICATION_RANGE,
        coordination=coordination,
        team_reward_weight=ma_config.TEAM_REWARD_WEIGHT,
        collision_penalty=ma_config.AGENT_COLLISION_PENALTY
    )

    # Initialize trainer
    trainer = MultiAgentTrainer(
        num_agents=num_agents,
        grid_size=ma_config.GRID_SIZE,
        coordination=coordination,
        parameter_sharing=parameter_sharing,
        shared_replay=shared_replay,
        input_channels=6 if use_6ch else 5
    )

    # Initialize communication protocol
    comm_manager = get_communication_protocol(
        protocol_name=comm_protocol,
        num_agents=num_agents,
        grid_size=ma_config.GRID_SIZE,
        comm_range=ma_config.COMMUNICATION_RANGE
    )
    print(f"✓ Communication protocol: {comm_protocol}")

    # Initialize agent occupancy computer (if using 6 channels)
    occupancy_computer = None
    if use_6ch:
        occupancy_computer = AgentOccupancyComputer(
            grid_size=ma_config.GRID_SIZE,
            base_sigma=0.5,
            max_velocity=1.0,
            time_decay_rate=0.1
        )
        print(f"✓ Agent occupancy computation enabled")

    print(f"✓ Environment and trainer initialized\n")

    # Training metrics
    all_validation_results = []
    start_time = time.time()

    # Current curriculum phase
    current_phase = None

    # Training loop
    for episode in range(total_episodes):

        # Update curriculum phase
        if use_curriculum:
            phase = ma_config.get_phase(episode)

            if phase != current_phase:
                current_phase = phase
                print(f"\n{'='*70}")
                print(f"CURRICULUM PHASE CHANGE @ Episode {episode}")
                print(f"{'='*70}")
                print(f"Phase: {phase['name']}")
                print(f"Episodes: {phase['start_ep']}-{phase['end_ep']}")
                print(f"Expected Coverage: {phase['expected_coverage']*100:.0f}%")
                print(f"Epsilon Floor: {phase['epsilon_floor']}")
                print(f"{'='*70}\n")

                # Update environment configuration
                if phase['num_agents'] != num_agents:
                    print(f"⚠ Warning: Phase requires {phase['num_agents']} agents, "
                          f"but training with {num_agents}. Continuing...")

                # Update coordination if different
                if phase['coordination'] != env.coordination:
                    env.coordination = phase['coordination']
                    print(f"✓ Updated coordination to {phase['coordination'].value}")

            # Get map type from curriculum
            map_type = ma_config.get_map_type(episode)

            # Update epsilon based on curriculum
            epsilon_floor = phase['epsilon_floor']
            trainer.epsilon = max(epsilon_floor, trainer.epsilon * phase['epsilon_decay'])
            trainer.set_epsilon(trainer.epsilon)

        else:
            # No curriculum - use default settings
            map_type = None
            trainer.decay_epsilon(decay_rate=0.995)

        # Train episode
        episode_info = trainer.train_episode(
            env, 
            map_type=map_type,
            comm_manager=comm_manager,
            occupancy_computer=occupancy_computer
        )

        # Log progress
        if episode % ma_config.LOG_FREQ == 0:
            log_episode(episode, episode_info, trainer)

        # Validation
        if (episode + 1) % ma_config.VALIDATION_FREQ == 0:
            val_results = validate_and_save(
                episode + 1,
                trainer,
                env,
                experiment_name
            )
            all_validation_results.append({
                'episode': episode + 1,
                'results': val_results
            })

        # Periodic stats
        if (episode + 1) % ma_config.PLOT_FREQ == 0:
            stats = trainer.get_training_stats(window=100)
            print(f"\n{'='*70}")
            print(f"TRAINING STATS @ Episode {episode + 1}")
            print(f"{'='*70}")
            print(f"  Mean Team Reward (100 ep): {stats['mean_team_reward']:.1f}")
            print(f"  Mean Coverage (100 ep): {stats['mean_coverage']*100:.1f}%")
            print(f"  Mean Length (100 ep): {stats['mean_length']:.0f}")
            print(f"  Mean Collisions (100 ep): {stats['mean_collisions']:.1f}")
            print(f"  Epsilon: {stats['epsilon']:.3f}")

            if 'mean_loss' in stats:
                print(f"  Mean Loss (100 ep): {stats['mean_loss']:.4f}")

            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            remaining = (total_episodes - episode - 1) / eps_per_sec

            print(f"\n  Elapsed: {elapsed/3600:.1f}h")
            print(f"  Speed: {eps_per_sec:.2f} ep/s")
            print(f"  Remaining: {remaining/3600:.1f}h")
            print(f"{'='*70}\n")

    # Final validation
    print(f"\n{'='*70}")
    print(f"FINAL VALIDATION")
    print(f"{'='*70}\n")

    final_val_results = validate_and_save(
        total_episodes,
        trainer,
        env,
        experiment_name
    )

    # Save final model
    final_path = os.path.join(
        ma_config.CHECKPOINT_DIR,
        f"{experiment_name}_FINAL.pth"
    )
    trainer.save(final_path)
    print(f"✓ Saved final model: {final_path}")

    # Training summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Episodes: {total_episodes}")
    print(f"  Total Time: {total_time/3600:.2f} hours")
    print(f"  Average Speed: {total_episodes/total_time:.2f} ep/s")
    print(f"\n  Final Coverage: {final_val_results['mean_coverage']*100:.1f}% "
          f"(±{final_val_results['std_coverage']*100:.1f}%)")
    print(f"  Final Team Reward: {final_val_results['mean_team_reward']:.1f}")
    print(f"{'='*70}\n")

    return trainer, all_validation_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train multi-agent coverage system"
    )

    # Training parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=1000,
        help='Total training episodes'
    )

    parser.add_argument(
        '--agents',
        type=int,
        default=4,
        help='Number of agents [2-8]'
    )

    parser.add_argument(
        '--coordination',
        type=str,
        default='independent',
        choices=['independent', 'voronoi', 'market', 'hierarchical'],
        help='Coordination strategy'
    )

    parser.add_argument(
        '--no-parameter-sharing',
        action='store_true',
        help='Disable parameter sharing (independent networks)'
    )

    parser.add_argument(
        '--no-shared-replay',
        action='store_true',
        help='Disable shared replay memory (separate buffers)'
    )

    parser.add_argument(
        '--no-curriculum',
        action='store_true',
        help='Disable curriculum learning'
    )

    parser.add_argument(
        '--use-6ch',
        action='store_true',
        help='Use 6-channel input with agent occupancy (enables proactive coordination)'
    )

    parser.add_argument(
        '--comm-protocol',
        type=str,
        default='none',
        choices=['none', 'full_state', 'attention', 'commnet', 'targeted'],
        help='Communication protocol (default: none)'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )

    args = parser.parse_args()

    # Parse coordination strategy
    coordination_map = {
        'independent': CoordinationStrategy.INDEPENDENT,
        'voronoi': CoordinationStrategy.VORONOI,
        'market': CoordinationStrategy.MARKET,
        'hierarchical': CoordinationStrategy.HIERARCHICAL
    }
    coordination = coordination_map[args.coordination]

    # Train
    train_multi_agent(
        num_agents=args.agents,
        total_episodes=args.episodes,
        coordination=coordination,
        parameter_sharing=not args.no_parameter_sharing,
        shared_replay=not args.no_shared_replay,
        use_curriculum=not args.no_curriculum,
        use_6ch=args.use_6ch,
        comm_protocol=args.comm_protocol,
        experiment_name=args.experiment_name
    )


if __name__ == "__main__":
    main()
