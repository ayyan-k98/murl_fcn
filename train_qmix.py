"""
QMIX Training Script

Train multi-agent coverage system using QMIX (Monotonic Value Function Factorization).

QMIX Architecture:
    - Individual agent Q-networks (FCN + Spatial Softmax)
    - Mixing network with monotonic weights (hypernetworks)
    - Centralized training, decentralized execution (CTDE)

Usage:
    python train_qmix.py --episodes 400 --agents 4
    python train_qmix.py --episodes 400 --agents 4 --use-6ch --comm-protocol full_state

Features:
    - 6-phase curriculum learning
    - Optional agent occupancy channel (6ch input)
    - Multiple communication protocols
    - Collision avoidance strategies
    - Optional potential-based reward shaping (PBRS)
    - Comprehensive logging and validation
"""

import argparse
import os
import time
import numpy as np
from datetime import datetime
from typing import Optional

from qmix_agent import QMIXAgent
from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
from multi_agent_config import ma_config
from config import config
from communication import get_communication_protocol
from agent_occupancy import AgentOccupancyComputer
from collision_avoidance import CollisionAvoider
from potential_based_shaping import get_shaper_config
from coordination_metrics import CoordinationAnalyzer, CoordinationMetrics, coordination_score


def create_directories():
    """Create necessary directories for results."""
    os.makedirs(ma_config.VIS_DIR, exist_ok=True)
    os.makedirs(ma_config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(ma_config.METRICS_DIR, exist_ok=True)
    print(f"+ Created directories:")
    print(f"  {ma_config.VIS_DIR}")
    print(f"  {ma_config.CHECKPOINT_DIR}")
    print(f"  {ma_config.METRICS_DIR}")


def log_episode(episode: int, metrics: dict):
    """Log episode information (ASCII-safe)."""
    team_reward = metrics.get('team_reward', 0)
    coverage = metrics.get('coverage', 0)
    length = metrics.get('episode_length', 0)
    collisions = metrics.get('collisions', 0)
    agent_collisions = metrics.get('agent_collisions', 0)
    epsilon = metrics.get('epsilon', 0)
    loss = metrics.get('loss', None)
    coord_score = metrics.get('coordination_score', 0)
    coord_metrics = metrics.get('coordination_metrics', None)

    log_str = (f"Ep {episode:4d} | "
               f"Cov: {coverage*100:5.1f}% | "
               f"Coord: {coord_score:.1f}/100 | "
               f"Rew: {team_reward:7.1f} | "
               f"Len: {length:3d} | "
               f"Eps: {epsilon:.3f}")
    
    if loss is not None:
        log_str += f" | Loss: {loss:.4f}"
    
    print(log_str)
    
    # Print detailed coordination breakdown every 50 episodes
    if coord_metrics and episode % 50 == 0:
        print(f"  └─ Overlap: {coord_metrics.overlap.overlap_ratio*100:.1f}% | "
              f"Efficiency: {coord_metrics.efficiency.exploration_efficiency*100:.1f}% | "
              f"Balance: {coord_metrics.load_balance.balance_ratio:.2f} | "
              f"Collisions: {coord_metrics.collisions.agent_agent + coord_metrics.collisions.agent_obstacle}")


def validate_qmix(
    qmix_agent: QMIXAgent,
    env: MultiAgentCoverageEnv,
    num_episodes: int = 10,
    map_types: Optional[list] = None,
    comm_manager=None,
    occupancy_computer=None,
    verbose: bool = True
) -> dict:
    """
    Validate QMIX agent.

    Args:
        qmix_agent: QMIX agent to validate
        env: Environment
        num_episodes: Number of validation episodes per map type
        map_types: List of map types to validate on
        comm_manager: Communication manager (optional)
        occupancy_computer: Agent occupancy computer (optional)
        verbose: Print validation results

    Returns:
        results: Dictionary with validation metrics
    """
    if map_types is None:
        map_types = ma_config.VALIDATION_MAP_TYPES

    # Save current epsilon
    original_epsilon = qmix_agent.epsilon
    qmix_agent.epsilon = 0.1  # Low epsilon for validation

    all_coverages = []
    all_rewards = []
    all_lengths = []
    all_collisions = []
    all_coordination_scores = []
    per_map_results = {}

    for map_type in map_types:
        map_coverages = []
        map_rewards = []
        map_lengths = []
        map_collisions = []

        for ep in range(num_episodes):
            state = env.reset(map_type=map_type)
            observations = env.get_observations()

            episode_reward = 0
            step_count = 0
            episode_collisions = 0
            episode_agent_collisions = 0
            
            # Initialize coordination tracking
            coord_analyzer = CoordinationAnalyzer(env.num_agents, env.grid_size)

            done = False

            while not done and step_count < config.MAX_EPISODE_STEPS:
                # Communication phase (if enabled)
                messages = []
                if comm_manager is not None:
                    messages = comm_manager.communicate(observations, state)

                # Compute occupancies (if using 6 channels)
                agent_occupancies = None
                if occupancy_computer is not None and messages:
                    agent_occupancies = [
                        occupancy_computer.compute(i, messages, step_count)
                        for i in range(env.num_agents)
                    ]

                # Select actions
                actions = qmix_agent.select_actions(
                    observations,
                    epsilon=qmix_agent.epsilon,
                    agent_occupancies=agent_occupancies
                )

                # Execute
                state, rewards, done, info = env.step(actions)
                next_observations = env.get_observations()

                episode_reward += sum(rewards)
                episode_collisions += sum(info['collisions'])
                episode_agent_collisions += sum(info['agent_collisions'])
                
                # Update coordination metrics
                agent_positions = [obs['robot_state'].position for obs in next_observations]
                visited_maps = [np.array(obs['world_state'].coverage_map > 0, dtype=bool) for obs in next_observations]
                coord_analyzer.update(
                    positions=agent_positions,
                    visited_maps=visited_maps,
                    actions=actions,
                    messages=messages
                )
                
                observations = next_observations
                step_count += 1

            # Record metrics
            final_coverage = info['coverage_pct']
            coord_metrics = coord_analyzer.get_metrics()
            coord_score_val = coordination_score(coord_metrics)
            
            map_coverages.append(final_coverage)
            map_rewards.append(episode_reward)
            map_lengths.append(step_count)
            map_collisions.append(episode_collisions)
            all_coordination_scores.append(coord_score_val)

        # Aggregate for this map type
        per_map_results[map_type] = {
            'mean_coverage': np.mean(map_coverages),
            'std_coverage': np.std(map_coverages),
            'mean_reward': np.mean(map_rewards),
            'mean_length': np.mean(map_lengths),
            'mean_collisions': np.mean(map_collisions)
        }

        all_coverages.extend(map_coverages)
        all_rewards.extend(map_rewards)
        all_lengths.extend(map_lengths)
        all_collisions.extend(map_collisions)

    # Overall results
    results = {
        'mean_coverage': np.mean(all_coverages),
        'std_coverage': np.std(all_coverages),
        'mean_reward': np.mean(all_rewards),
        'mean_length': np.mean(all_lengths),
        'mean_collisions': np.mean(all_collisions),
        'mean_coordination_score': np.mean(all_coordination_scores),
        'std_coordination_score': np.std(all_coordination_scores),
        'per_map': per_map_results
    }

    # Restore epsilon
    qmix_agent.epsilon = original_epsilon

    if verbose:
        print(f"\nValidation Results ({len(all_coverages)} episodes):")
        print(f"  Mean Coverage: {results['mean_coverage']*100:.1f}% "
              f"(+/- {results['std_coverage']*100:.1f}%)")
        print(f"  Mean Reward: {results['mean_reward']:.1f}")
        print(f"  Mean Length: {results['mean_length']:.0f}")
        print(f"  Mean Collisions: {results['mean_collisions']:.1f}")
        print(f"  Mean Coordination Score: {results['mean_coordination_score']:.1f}/100 "
              f"(+/- {results['std_coordination_score']:.1f})")
        print(f"\nPer-Map Results:")
        for map_type, res in per_map_results.items():
            print(f"  {map_type:12s}: {res['mean_coverage']*100:.1f}%")

    return results


def train_qmix(
    num_agents: int = 4,
    total_episodes: int = 400,
    grid_size: int = 20,
    use_curriculum: bool = True,
    use_6ch: bool = False,
    comm_protocol: str = 'none',
    collision_strategy: str = 'filter',
    use_pbrs: bool = False,
    pbrs_config: str = 'frontier',
    experiment_name: Optional[str] = None,
    resume_from: Optional[str] = None
):
    """
    Main QMIX training loop.

    Args:
        num_agents: Number of agents
        total_episodes: Total training episodes
        grid_size: Grid size
        use_curriculum: Use curriculum learning
        use_6ch: Use 6-channel input with agent occupancy
        comm_protocol: Communication protocol
        collision_strategy: Collision avoidance strategy
        use_pbrs: Use potential-based reward shaping
        pbrs_config: PBRS configuration
        experiment_name: Experiment name (auto-generated if None)
    """
    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ch_suffix = "6ch" if use_6ch else "5ch"
        comm_suffix = f"_{comm_protocol}" if comm_protocol != 'none' else ""
        pbrs_suffix = f"_pbrs_{pbrs_config}" if use_pbrs else ""
        experiment_name = f"qmix_{num_agents}ag_{ch_suffix}{comm_suffix}{pbrs_suffix}_{timestamp}"

    print(f"\n{'='*70}")
    print(f"QMIX TRAINING")
    print(f"{'='*70}")
    print(f"Experiment: {experiment_name}")
    print(f"Agents: {num_agents}")
    print(f"Grid Size: {grid_size}")
    print(f"Input Channels: {6 if use_6ch else 5} ({'with agent occupancy' if use_6ch else 'baseline'})")
    print(f"Communication: {comm_protocol}")
    print(f"Collision Strategy: {collision_strategy}")
    print(f"Curriculum: {use_curriculum}")
    if config.USE_PROBABILISTIC_ENV:
        print(f"Environment: PROBABILISTIC (sigmoid coverage)")
    else:
        print(f"Environment: BINARY (instant coverage)")
    print(f"PBRS: {use_pbrs} ({pbrs_config if use_pbrs else 'disabled'})")
    print(f"Total Episodes: {total_episodes}")
    print(f"{'='*70}\n")

    # Create directories
    create_directories()

    # Initialize environment
    env = MultiAgentCoverageEnv(
        num_agents=num_agents,
        grid_size=grid_size,
        sensor_range=ma_config.SENSOR_RANGE,
        communication_range=ma_config.COMMUNICATION_RANGE,
        coordination=CoordinationStrategy.INDEPENDENT,
        team_reward_weight=ma_config.TEAM_REWARD_WEIGHT,
        collision_penalty=ma_config.AGENT_COLLISION_PENALTY
    )

    # Initialize QMIX agent
    qmix_agent = QMIXAgent(
        num_agents=num_agents,
        grid_size=grid_size,
        input_channels=6 if use_6ch else 5,
        learning_rate=config.LEARNING_RATE,
        gamma=config.GAMMA,
        device=config.DEVICE
    )

    # Load pre-trained single-agent checkpoint if provided
    if resume_from is not None:
        print(f"\n{'='*70}")
        print(f"LOADING PRE-TRAINED CHECKPOINT")
        print(f"{'='*70}")
        print(f"Checkpoint: {resume_from}")
        
        import torch
        checkpoint = torch.load(resume_from, map_location=config.DEVICE)
        
        # Load weights into each agent's Q-network
        for i in range(num_agents):
            qmix_agent.agent_qnets[i].load_state_dict(checkpoint['policy_net_state_dict'])
            qmix_agent.target_qnets[i].load_state_dict(checkpoint['policy_net_state_dict'])
            print(f"+ Loaded checkpoint into agent {i} Q-network")
        
        print(f"+ All {num_agents} agents initialized with same pre-trained weights")
        print(f"+ Mixing network will be trained from scratch")
        print(f"{'='*70}\n")

    # Initialize communication protocol
    comm_manager = get_communication_protocol(
        protocol_name=comm_protocol,
        num_agents=num_agents,
        grid_size=grid_size,
        comm_range=ma_config.COMMUNICATION_RANGE
    )
    print(f"+ Communication protocol: {comm_protocol}")

    # Initialize agent occupancy computer (if using 6 channels)
    occupancy_computer = None
    if use_6ch:
        occupancy_computer = AgentOccupancyComputer(
            grid_size=grid_size,
            base_sigma=0.5,
            max_velocity=1.0,
            time_decay_rate=0.1
        )
        print(f"+ Agent occupancy computation enabled")

    # Initialize collision avoider
    collision_avoider = CollisionAvoider(
        grid_size=grid_size,
        num_agents=num_agents,
        strategy=collision_strategy
    )
    print(f"+ Collision avoidance: {collision_strategy}")

    # Initialize PBRS (if enabled)
    reward_shaper = None
    if use_pbrs:
        reward_shaper = get_shaper_config(pbrs_config)
        print(f"+ PBRS enabled: {pbrs_config}")
    else:
        print(f"+ PBRS disabled (recommended for initial training)")

    print(f"+ Environment and QMIX agent initialized\n")

    # Training metrics
    all_validation_results = []
    start_time = time.time()
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

            # Get map type from curriculum
            map_type = ma_config.get_map_type(episode)

            # Update epsilon based on curriculum
            epsilon_floor = phase['epsilon_floor']
            qmix_agent.epsilon = max(epsilon_floor, qmix_agent.epsilon * phase['epsilon_decay'])

        else:
            # No curriculum - use default settings
            map_type = None
            qmix_agent.epsilon *= 0.995  # Standard decay

        # Reset environment
        state = env.reset(map_type=map_type)
        observations = env.get_observations()

        episode_reward = 0
        episode_length = 0
        episode_collisions = 0
        episode_agent_collisions = 0
        episode_losses = []
        
        # Initialize coordination tracking
        coord_analyzer = CoordinationAnalyzer(num_agents, grid_size)

        done = False

        # Episode loop
        while not done and episode_length < config.MAX_EPISODE_STEPS:
            # Communication phase (if enabled)
            messages = []
            if comm_manager is not None:
                messages = comm_manager.communicate(observations, state)

            # Compute agent occupancies (if using 6 channels)
            agent_occupancies = None
            if occupancy_computer is not None and messages:
                agent_occupancies = [
                    occupancy_computer.compute(i, messages, episode_length)
                    for i in range(num_agents)
                ]

            # Select actions with occupancies
            actions = qmix_agent.select_actions(
                observations,
                epsilon=qmix_agent.epsilon,
                agent_occupancies=agent_occupancies
            )

            # Apply collision avoidance based on strategy
            if collision_strategy == 'resolve':
                safe_actions = collision_avoider.resolve_collision(
                    actions,
                    [obs['robot_state'].position for obs in observations],
                    env.world_state.obstacles
                )
            elif collision_strategy == 'filter':
                # Filter actions individually - use original actions if no collisions
                safe_actions = actions
            else:
                # No collision avoidance
                safe_actions = actions

            # Execute actions
            next_state, rewards, done, info = env.step(safe_actions)
            next_observations = env.get_observations()

            # Apply PBRS (if enabled)
            if reward_shaper is not None:
                shaped_rewards = reward_shaper.shape_rewards(
                    observations, safe_actions, next_observations, rewards
                )
            else:
                shaped_rewards = rewards

            # Compute next occupancies (if using 6 channels)
            next_agent_occupancies = None
            if occupancy_computer is not None and messages:
                next_messages = comm_manager.communicate(next_observations, next_state) if comm_manager else messages
                next_agent_occupancies = [
                    occupancy_computer.compute(i, next_messages, episode_length + 1)
                    for i in range(num_agents)
                ]

            # Store transition
            qmix_agent.store_transition(
                observations,
                safe_actions,
                shaped_rewards,
                next_observations,
                done,
                state,
                next_state,
                agent_occupancies=agent_occupancies,
                next_agent_occupancies=next_agent_occupancies
            )

            # Optimize
            if episode_length % config.TRAIN_FREQ == 0:
                loss = qmix_agent.optimize()
                if loss is not None:
                    episode_losses.append(loss)

            # Update target networks
            if episode_length % config.TARGET_UPDATE_FREQ == 0:
                qmix_agent.update_target_networks()

            # Track metrics
            episode_reward += sum(shaped_rewards)
            episode_collisions += sum(info['collisions'])
            episode_agent_collisions += sum(info['agent_collisions'])
            
            # Update coordination metrics
            agent_positions = [obs['robot_state'].position for obs in next_observations]
            visited_maps = [np.array(obs['world_state'].coverage_map > 0, dtype=bool) for obs in next_observations]
            coord_analyzer.update(
                positions=agent_positions,
                visited_maps=visited_maps,
                actions=actions,
                messages=messages
            )

            # Update for next step
            observations = next_observations
            state = next_state
            episode_length += 1

        # Episode metrics
        final_coverage = info['coverage_pct']
        mean_loss = np.mean(episode_losses) if episode_losses else None
        coord_metrics = coord_analyzer.get_metrics()
        coord_score_val = coordination_score(coord_metrics)

        episode_metrics = {
            'team_reward': episode_reward,
            'coverage': final_coverage,
            'episode_length': episode_length,
            'collisions': episode_collisions,
            'agent_collisions': episode_agent_collisions,
            'epsilon': qmix_agent.epsilon,
            'loss': mean_loss,
            'coordination_score': coord_score_val,
            'coordination_metrics': coord_metrics
        }

        # Log progress
        if episode % ma_config.LOG_FREQ == 0:
            log_episode(episode, episode_metrics)

        # Validation
        if (episode + 1) % ma_config.VALIDATION_FREQ == 0:
            print(f"\n{'='*70}")
            print(f"VALIDATION @ Episode {episode + 1}")
            print(f"{'='*70}")

            val_results = validate_qmix(
                qmix_agent,
                env,
                num_episodes=ma_config.VALIDATION_EPISODES,
                map_types=ma_config.VALIDATION_MAP_TYPES,
                comm_manager=comm_manager,
                occupancy_computer=occupancy_computer,
                verbose=True
            )

            all_validation_results.append({
                'episode': episode + 1,
                'results': val_results
            })

            # Save checkpoint
            checkpoint_path = os.path.join(
                ma_config.CHECKPOINT_DIR,
                f"{experiment_name}_ep{episode+1}.pth"
            )
            qmix_agent.save(checkpoint_path)
            print(f"\n+ Saved checkpoint: {checkpoint_path}")
            print(f"{'='*70}\n")

        # Periodic stats
        if (episode + 1) % ma_config.PLOT_FREQ == 0:
            elapsed = time.time() - start_time
            eps_per_sec = (episode + 1) / elapsed
            remaining = (total_episodes - episode - 1) / eps_per_sec

            print(f"\n{'='*70}")
            print(f"TRAINING STATS @ Episode {episode + 1}")
            print(f"{'='*70}")
            print(f"  Elapsed: {elapsed/3600:.2f}h")
            print(f"  Speed: {eps_per_sec:.2f} ep/s")
            print(f"  Remaining: {remaining/3600:.2f}h")
            print(f"  Epsilon: {qmix_agent.epsilon:.3f}")
            if mean_loss is not None:
                print(f"  Loss: {mean_loss:.4f}")
            print(f"{'='*70}\n")

    # Final validation
    print(f"\n{'='*70}")
    print(f"FINAL VALIDATION")
    print(f"{'='*70}")

    final_val_results = validate_qmix(
        qmix_agent,
        env,
        num_episodes=20,  # More episodes for final validation
        map_types=ma_config.VALIDATION_MAP_TYPES,
        comm_manager=comm_manager,
        occupancy_computer=occupancy_computer,
        verbose=True
    )

    # Save final model
    final_path = os.path.join(
        ma_config.CHECKPOINT_DIR,
        f"{experiment_name}_FINAL.pth"
    )
    qmix_agent.save(final_path)
    print(f"\n+ Saved final model: {final_path}")

    # Training summary
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"  Total Episodes: {total_episodes}")
    print(f"  Total Time: {total_time/3600:.2f} hours")
    print(f"  Average Speed: {total_episodes/total_time:.2f} ep/s")
    print(f"\n  Final Coverage: {final_val_results['mean_coverage']*100:.1f}% "
          f"(+/- {final_val_results['std_coverage']*100:.1f}%)")
    print(f"  Final Reward: {final_val_results['mean_reward']:.1f}")
    print(f"{'='*70}\n")

    return qmix_agent, all_validation_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train multi-agent coverage system with QMIX"
    )

    # Training parameters
    parser.add_argument(
        '--episodes',
        type=int,
        default=400,
        help='Total training episodes (default: 400)'
    )

    parser.add_argument(
        '--agents',
        type=int,
        default=4,
        help='Number of agents [2-8] (default: 4)'
    )

    parser.add_argument(
        '--grid-size',
        type=int,
        default=20,
        help='Grid size (default: 20)'
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
        '--collision-strategy',
        type=str,
        default='filter',
        choices=['none', 'filter', 'sequential', 'resolve'],
        help='Collision avoidance strategy (default: filter)'
    )

    parser.add_argument(
        '--use-pbrs',
        action='store_true',
        help='Enable potential-based reward shaping (NOT recommended for initial training)'
    )

    parser.add_argument(
        '--pbrs-config',
        type=str,
        default='frontier',
        choices=['frontier', 'local', 'expected', 'coordination'],
        help='PBRS configuration (default: frontier)'
    )

    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Experiment name (auto-generated if not provided)'
    )

    parser.add_argument(
        '--resume-from',
        type=str,
        default=None,
        help='Path to single-agent checkpoint (fcn_final.pt) to initialize agent Q-networks'
    )

    parser.add_argument(
        '--probabilistic',
        action='store_true',
        help='Use probabilistic environment (sigmoid coverage) instead of binary'
    )

    args = parser.parse_args()

    # Apply probabilistic environment setting if specified
    if args.probabilistic:
        config.USE_PROBABILISTIC_ENV = True

    # Train
    train_qmix(
        num_agents=args.agents,
        total_episodes=args.episodes,
        grid_size=args.grid_size,
        use_curriculum=not args.no_curriculum,
        use_6ch=args.use_6ch,
        comm_protocol=args.comm_protocol,
        collision_strategy=args.collision_strategy,
        use_pbrs=args.use_pbrs,
        pbrs_config=args.pbrs_config,
        experiment_name=args.experiment_name,
        resume_from=args.resume_from
    )


if __name__ == "__main__":
    main()
