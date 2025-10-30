"""
Train multi-agent system on 40×40 grids - Phase 1: Refined Independent Agents

This script trains 4 independent agents with:
- 6-channel input (including agent occupancy)
- Optimal sensor range (8.5) and comm range (15.0)
- Correct sigmoid parameters for 40×40 scale (k=0.92, r0=3.2)
- Extended Phase 1 (300 episodes) for coordination learning
- Full state communication (will refine later)

Phase 1 Goals:
- Overlap < 20% by episode 300
- Coverage > 80% on empty maps
- Positive rewards (collision bug fixed)
- Train-val gap < 5%

Next Phase:
- Add QMIX coordination
- Switch to position-velocity communication
- Add team rewards with overlap penalty
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# Local directory creation utility (since utils.ensure_dir does not exist)
def ensure_dir(path):
    """Create directory if it does not exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from multi_agent_config_40x40_phase1 import MultiAgentConfig40x40 as Config
from multi_agent_trainer import MultiAgentTrainer
from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
from communication import FullStateSharing
from agent_occupancy import AgentOccupancyComputer
from coordination_metrics import CoordinationAnalyzer
from config import config  # Base config for collision penalty


def create_results_dir():
    """Create directory structure for results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir = Path("multi_agent_results_40x40_phase1") / timestamp
    
    ensure_dir(base_dir / "checkpoints")
    ensure_dir(base_dir / "metrics")
    ensure_dir(base_dir / "visualizations")
    ensure_dir(base_dir / "logs")
    
    return base_dir


def setup_environment():
    """Create environment with Phase 1 configuration."""
    env = MultiAgentCoverageEnv(
        grid_size=Config.GRID_SIZE,
        num_agents=Config.NUM_AGENTS,
        sensor_range=Config.SENSOR_RANGE,
        communication_range=Config.COMMUNICATION_RANGE,
        coordination=Config.COORDINATION,
        collision_penalty=Config.COLLISION_PENALTY,  # -0.25 (FIXED!)
        team_reward_weight=Config.TEAM_REWARD_WEIGHT
    )
    # Set max_steps after instantiation
    env.max_steps = Config.MAX_STEPS
    
    print(f"✓ Environment created:")
    print(f"  Grid: {Config.GRID_SIZE}×{Config.GRID_SIZE}")
    print(f"  Agents: {Config.NUM_AGENTS}")
    print(f"  Sensor range: {Config.SENSOR_RANGE}")
    print(f"  Comm range: {Config.COMMUNICATION_RANGE}")
    print(f"  Sigmoid: k={Config.PROBABILISTIC_COVERAGE_STEEPNESS}, r0={Config.PROBABILISTIC_COVERAGE_MIDPOINT}")
    print(f"  Collision penalty: {Config.COLLISION_PENALTY}")
    
    return env


def setup_communication(env):
    """Create communication manager."""
    comm_manager = FullStateSharing(
        num_agents=Config.NUM_AGENTS,
        grid_size=Config.GRID_SIZE
    )
    
    print(f"✓ Communication: {Config.COMMUNICATION_TYPE}, range={Config.COMMUNICATION_RANGE}")
    
    return comm_manager


def setup_occupancy_computer():
    """Create agent occupancy computer for 6th channel."""
    occupancy_computer = AgentOccupancyComputer(
        grid_size=Config.GRID_SIZE,
        base_sigma=Config.POSITION_UNCERTAINTY_BASE,
        max_velocity=1.0,  # Agents move 1 cell per step
        time_decay_rate=Config.POSITION_UNCERTAINTY_GROWTH / Config.COMMUNICATION_FREQUENCY
    )
    
    print(f"✓ Agent occupancy: σ_base={Config.POSITION_UNCERTAINTY_BASE}, σ_growth={Config.POSITION_UNCERTAINTY_GROWTH}")
    
    return occupancy_computer


def setup_trainer(env, comm_manager, occupancy_computer):
    """Create multi-agent trainer."""
    trainer = MultiAgentTrainer(
        num_agents=Config.NUM_AGENTS,
        grid_size=Config.GRID_SIZE,
        learning_rate=Config.LEARNING_RATE,
        gamma=Config.GAMMA,
        target_update_freq=Config.TARGET_UPDATE_FREQ,
        communication_manager=comm_manager,
        coordination_strategy=Config.COORDINATION,
        parameter_sharing=Config.PARAMETER_SHARING,
        shared_replay=Config.SHARED_REPLAY,
        input_channels=Config.INPUT_CHANNELS,  # 6 channels!
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print(f"✓ Trainer created:")
    print(f"  Input channels: {Config.INPUT_CHANNELS}")
    print(f"  Parameter sharing: {Config.PARAMETER_SHARING}")
    print(f"  Coordination: {Config.COORDINATION.value}")
    print(f"  Device: {trainer.device}")
    
    return trainer


def train_episode(trainer, env, comm_manager, occupancy_computer, epsilon, map_type='empty'):
    """Train one episode."""
    return trainer.train_episode(
        env=env,
        epsilon=epsilon,
        comm_manager=comm_manager,
        occupancy_computer=occupancy_computer,
        map_type=map_type
    )


def validate(trainer, env, comm_manager, occupancy_computer, num_episodes=20):
    """Run validation episodes on all map types."""
    results = {
        'coverage': [],
        'overlap': [],
        'collisions': [],
        'rewards': [],
        'efficiency': [],
        'corridor_coverage': []
    }
    
    print(f"\n{'='*70}")
    print("VALIDATION")
    print(f"{'='*70}")
    
    for map_type in Config.VALIDATION_MAP_TYPES:
        type_results = []
        
        for i in range(num_episodes // len(Config.VALIDATION_MAP_TYPES)):
            metrics = trainer.train_episode(
                env=env,
                epsilon=0.0,  # Greedy policy for validation
                comm_manager=comm_manager,
                occupancy_computer=occupancy_computer,
                map_type=map_type,
                training=False
            )
            type_results.append(metrics)
        
        # Aggregate results for this map type
        avg_coverage = np.mean([m['coverage'] for m in type_results])
        avg_overlap = np.mean([m['overlap'] for m in type_results])
        avg_reward = np.mean([m['total_reward'] for m in type_results])
        
        results['coverage'].extend([m['coverage'] for m in type_results])
        results['overlap'].extend([m['overlap'] for m in type_results])
        results['collisions'].extend([m['collisions'] for m in type_results])
        results['rewards'].extend([m['total_reward'] for m in type_results])
        results['efficiency'].extend([m['efficiency'] for m in type_results])
        
        # Track corridor performance separately
        if map_type == 'corridor':
            results['corridor_coverage'].extend([m['coverage'] for m in type_results])
        
        print(f"  {map_type:10s}: Cov={avg_coverage:.2%}, Overlap={avg_overlap:.2%}, Reward={avg_reward:+.1f}")
    
    # Compute aggregated statistics
    summary = {
        'coverage_mean': np.mean(results['coverage']),
        'coverage_std': np.std(results['coverage']),
        'overlap_mean': np.mean(results['overlap']),
        'overlap_std': np.std(results['overlap']),
        'collision_mean': np.mean(results['collisions']),
        'reward_mean': np.mean(results['rewards']),
        'reward_std': np.std(results['rewards']),
        'efficiency_mean': np.mean(results['efficiency']),
        'corridor_mean': np.mean(results['corridor_coverage']) if results['corridor_coverage'] else 0.0
    }
    
    print(f"\nSummary:")
    print(f"  Coverage: {summary['coverage_mean']:.2%} ± {summary['coverage_std']:.2%}")
    print(f"  Overlap: {summary['overlap_mean']:.2%} ± {summary['overlap_std']:.2%}")
    print(f"  Collisions: {summary['collision_mean']:.1f}")
    print(f"  Reward: {summary['reward_mean']:+.1f} ± {summary['reward_std']:.1f}")
    print(f"  Efficiency: {summary['efficiency_mean']:.2%}")
    if results['corridor_coverage']:
        print(f"  Corridor: {summary['corridor_mean']:.2%}")
    print(f"{'='*70}\n")
    
    return summary


def plot_training_curves(history, results_dir):
    """Plot training metrics."""
    episodes = list(range(1, len(history['coverage']) + 1))
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('40×40 Phase 1 Training: Refined Independent Agents (6 Channels)', fontsize=16)
    
    # Coverage
    axes[0, 0].plot(episodes, history['coverage'], label='Train', alpha=0.7)
    if history['val_coverage']:
        val_episodes = [i * Config.VALIDATION_FREQ for i in range(1, len(history['val_coverage']) + 1)]
        axes[0, 0].plot(val_episodes, history['val_coverage'], label='Val', marker='o')
    axes[0, 0].axhline(y=0.80, color='g', linestyle='--', label='Target 80%')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Coverage')
    axes[0, 0].set_title('Coverage')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Overlap
    axes[0, 1].plot(episodes, history['overlap'], alpha=0.7)
    axes[0, 1].axhline(y=0.20, color='r', linestyle='--', label='Phase 1 Target 20%')
    axes[0, 1].axhline(y=0.15, color='g', linestyle='--', label='Final Target 15%')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Overlap')
    axes[0, 1].set_title('Coverage Overlap')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Rewards
    axes[0, 2].plot(episodes, history['rewards'], alpha=0.7)
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].set_xlabel('Episode')
    axes[0, 2].set_ylabel('Total Reward')
    axes[0, 2].set_title('Rewards (should be positive!)')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Collisions
    axes[1, 0].plot(episodes, history['collisions'], alpha=0.7)
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Collisions')
    axes[1, 0].set_title('Collisions per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Efficiency
    axes[1, 1].plot(episodes, history['efficiency'], alpha=0.7)
    axes[1, 1].axhline(y=0.75, color='g', linestyle='--', label='Target 75%')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Efficiency')
    axes[1, 1].set_title('Coverage Efficiency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Epsilon
    axes[1, 2].plot(episodes, history['epsilon'], alpha=0.7)
    axes[1, 2].set_xlabel('Episode')
    axes[1, 2].set_ylabel('Epsilon')
    axes[1, 2].set_title('Exploration Rate')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'training_curves.png', dpi=150)
    print(f"✓ Saved training curves to {results_dir / 'training_curves.png'}")


def main():
    """Main training loop."""
    print("\n" + "="*70)
    print("MULTI-AGENT TRAINING: 40×40 PHASE 1")
    print("Refined Independent Agents with 6 Channels")
    print("="*70 + "\n")
    
    # Print configuration
    Config.print_config()
    
    # Create results directory
    results_dir = create_results_dir()
    print(f"\n✓ Results directory: {results_dir}\n")
    
    # Setup
    env = setup_environment()
    comm_manager = setup_communication(env)
    occupancy_computer = setup_occupancy_computer()
    trainer = setup_trainer(env, comm_manager, occupancy_computer)

    # Optionally load from checkpoint
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint to load')
    args, _ = parser.parse_known_args()
    if args.checkpoint is not None:
        print(f"\n✓ Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=trainer.device)
        for agent, state_dict in zip(trainer.agents, checkpoint['agents']):
            agent.policy_net.load_state_dict(state_dict)
        print("✓ Checkpoint loaded.")
    
    # Coordination analyzer
    coord_analyzer = CoordinationAnalyzer(
        grid_size=Config.GRID_SIZE,
        num_agents=Config.NUM_AGENTS
    )
    
    # Training history
    history = {
        'coverage': [],
        'overlap': [],
        'rewards': [],
        'collisions': [],
        'efficiency': [],
        'epsilon': [],
        'val_coverage': [],
        'val_overlap': [],
        'val_rewards': [],
        'corridor_coverage': []
    }
    
    # Training loop
    print(f"\n{'='*70}")
    print("STARTING TRAINING")
    print(f"{'='*70}\n")
    
    epsilon = Config.EPSILON_START
    best_val_coverage = 0.0
    
    for episode in range(1, Config.TOTAL_EPISODES + 1):
        # Get current phase
        phase = Config.get_phase(episode)
        
        # Update epsilon decay rate based on phase
        epsilon_decay = phase['epsilon_decay']
        
        # Sample map type for this episode
        map_type = Config.get_map_type(episode)
        
        # Train episode
        metrics = train_episode(
            trainer, env, comm_manager, occupancy_computer, 
            epsilon, map_type
        )
        
        # Update epsilon
        epsilon = max(Config.EPSILON_MIN, epsilon * epsilon_decay)
        
        # Record metrics
        history['coverage'].append(metrics['coverage'])
        history['overlap'].append(metrics['overlap'])
        history['rewards'].append(metrics['total_reward'])
        history['collisions'].append(metrics['collisions'])
        history['efficiency'].append(metrics['efficiency'])
        history['epsilon'].append(epsilon)
        
        # Logging
        if episode % Config.LOG_FREQ == 0:
            print(f"[Ep {episode:4d}] Phase {phase['phase']} | {map_type:8s} | "
                  f"Cov: {metrics['coverage']:.2%} | Overlap: {metrics['overlap']:.2%} | "
                  f"Reward: {metrics['total_reward']:+6.1f} | Coll: {metrics['collisions']:3.0f} | "
                  f"Eff: {metrics['efficiency']:.2%} | ε: {epsilon:.3f}")
        
        # Validation
        if episode % Config.VALIDATION_FREQ == 0:
            val_results = validate(trainer, env, comm_manager, occupancy_computer, 
                                  num_episodes=Config.VALIDATION_EPISODES)
            
            history['val_coverage'].append(val_results['coverage_mean'])
            history['val_overlap'].append(val_results['overlap_mean'])
            history['val_rewards'].append(val_results['reward_mean'])
            history['corridor_coverage'].append(val_results['corridor_mean'])
            
            # Check for improvement
            if val_results['coverage_mean'] > best_val_coverage:
                best_val_coverage = val_results['coverage_mean']
                torch.save({
                    'episode': episode,
                    'agents': [agent.policy_net.state_dict() for agent in trainer.agents],
                    'epsilon': epsilon,
                    'val_coverage': val_results['coverage_mean'],
                    'config': vars(Config)
                }, results_dir / 'checkpoints' / 'best_model.pt')
                print(f"  ✓ New best validation coverage: {best_val_coverage:.2%}")
        
        # Checkpointing
        if episode % Config.SAVE_FREQ == 0:
            torch.save({
                'episode': episode,
                'agents': [agent.policy_net.state_dict() for agent in trainer.agents],
                'epsilon': epsilon,
                'history': history,
                'config': vars(Config)
            }, results_dir / 'checkpoints' / f'checkpoint_ep{episode}.pt')
            print(f"  ✓ Saved checkpoint at episode {episode}")
        
        # Plot training curves
        if episode % Config.PLOT_FREQ == 0:
            plot_training_curves(history, results_dir)
    
    # Final validation
    print(f"\n{'='*70}")
    print("FINAL VALIDATION")
    print(f"{'='*70}")
    final_results = validate(trainer, env, comm_manager, occupancy_computer, 
                            num_episodes=50)
    
    # Save final model
    torch.save({
        'episode': Config.TOTAL_EPISODES,
        'agents': [agent.policy_net.state_dict() for agent in trainer.agents],
        'epsilon': epsilon,
        'history': history,
        'final_val': final_results,
        'config': vars(Config)
    }, results_dir / 'checkpoints' / 'final_model.pt')
    
    # Final plots
    plot_training_curves(history, results_dir)
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_dir}")
    print(f"Best validation coverage: {best_val_coverage:.2%}")
    print(f"Final validation coverage: {final_results['coverage_mean']:.2%}")
    print(f"Final overlap: {final_results['overlap_mean']:.2%}")
    print(f"Final corridor performance: {final_results['corridor_mean']:.2%}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
