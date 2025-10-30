"""
Visualize Trained Multi-Agent System

Quick script to visualize trained multi-agent coverage.
"""

import argparse
import torch
import numpy as np
from multi_agent_trainer import MultiAgentTrainer
from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
from multi_agent_vis import MultiAgentVisualizer, visualize_episode
from config import config


def visualize_checkpoint(
    checkpoint_path: str,
    num_agents: int = 4,
    map_type: str = 'random',
    num_episodes: int = 1,
    save_frames: bool = False
):
    """
    Load checkpoint and visualize episodes.
    
    Args:
        checkpoint_path: Path to saved checkpoint
        num_agents: Number of agents
        map_type: Map type to test
        num_episodes: Number of episodes to visualize
        save_frames: If True, save frames to disk
    """
    print(f"\n{'='*70}")
    print(f"MULTI-AGENT VISUALIZATION")
    print(f"{'='*70}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Agents: {num_agents}")
    print(f"Map Type: {map_type}")
    print(f"Episodes: {num_episodes}")
    print(f"{'='*70}\n")
    
    # Load checkpoint to get configuration
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    
    # Create environment
    env = MultiAgentCoverageEnv(
        num_agents=num_agents,
        grid_size=20,
        coordination=CoordinationStrategy[checkpoint['coordination'].upper()]
    )
    
    # Create trainer
    trainer = MultiAgentTrainer(
        num_agents=num_agents,
        grid_size=20,
        coordination=CoordinationStrategy[checkpoint['coordination'].upper()],
        parameter_sharing=checkpoint.get('parameter_sharing', True),
        shared_replay=True
    )
    
    # Load trained weights
    trainer.load(checkpoint_path)
    print(f"✓ Loaded checkpoint")
    
    # Visualizer
    visualizer = MultiAgentVisualizer(grid_size=20)
    
    # Run episodes
    for ep in range(num_episodes):
        print(f"\n{'='*70}")
        print(f"EPISODE {ep + 1}/{num_episodes}")
        print(f"{'='*70}\n")
        
        # Reset
        state = env.reset(map_type=map_type)
        observations = env.get_observations()
        
        done = False
        step = 0
        episode_reward = 0
        
        # Display initial state
        print(f"Initial state (Step 0):")
        if not save_frames:
            visualizer.render_episode_state(
                env, state,
                title=f"Episode {ep + 1} - Step 0"
            )
        
        # Run episode
        while not done and step < 350:
            # Select actions (greedy)
            actions = trainer.select_actions(observations, epsilon=0.0)
            
            # Step
            next_state, rewards, done, info = env.step(actions)
            next_observations = env.get_observations()
            
            episode_reward += sum(rewards)
            
            # Visualize every 50 steps
            if step % 50 == 0 and step > 0:
                print(f"\nStep {step}:")
                print(f"  Coverage: {info['coverage_pct']*100:.1f}%")
                print(f"  Reward: {sum(rewards):.2f}")
                print(f"  Collisions: {sum(info['collisions'])}")
                
                if not save_frames:
                    visualizer.render_episode_state(
                        env, next_state,
                        title=f"Episode {ep + 1} - Step {step}"
                    )
            
            observations = next_observations
            state = next_state
            step += 1
        
        # Final state
        print(f"\n{'='*70}")
        print(f"EPISODE {ep + 1} RESULTS")
        print(f"{'='*70}")
        print(f"Steps: {step}")
        print(f"Final Coverage: {info['coverage_pct']*100:.1f}%")
        print(f"Total Reward: {episode_reward:.2f}")
        print(f"Total Collisions: {sum(info['collisions'])}")
        print(f"Agent Collisions: {sum(info['agent_collisions'])}")
        print(f"{'='*70}\n")
        
        visualizer.render_episode_state(
            env, state,
            title=f"Episode {ep + 1} - FINAL (Coverage: {info['coverage_pct']*100:.1f}%)"
        )
    
    print(f"\n✓ Visualization complete")


def visualize_training_metrics(checkpoint_path: str):
    """
    Load checkpoint and plot training metrics.
    
    Args:
        checkpoint_path: Path to saved checkpoint
    """
    print(f"\nLoading training metrics from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'metrics' not in checkpoint:
        print("❌ No metrics found in checkpoint")
        return
    
    metrics = checkpoint['metrics']
    
    visualizer = MultiAgentVisualizer(grid_size=20)
    visualizer.plot_training_metrics(
        metrics,
        window=100,
        save_path=None  # Display instead of save
    )
    
    print(f"\n✓ Training metrics plotted")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trained multi-agent system"
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint file'
    )
    
    parser.add_argument(
        '--agents',
        type=int,
        default=4,
        help='Number of agents (default: 4)'
    )
    
    parser.add_argument(
        '--map-type',
        type=str,
        default='random',
        choices=['empty', 'random', 'room', 'corridor', 'cave'],
        help='Map type to visualize'
    )
    
    parser.add_argument(
        '--episodes',
        type=int,
        default=1,
        help='Number of episodes to visualize'
    )
    
    parser.add_argument(
        '--metrics-only',
        action='store_true',
        help='Only plot training metrics (no episode visualization)'
    )
    
    parser.add_argument(
        '--save-frames',
        action='store_true',
        help='Save episode frames to disk'
    )
    
    args = parser.parse_args()
    
    if args.metrics_only:
        # Just plot training curves
        visualize_training_metrics(args.checkpoint)
    else:
        # Run and visualize episodes
        visualize_checkpoint(
            checkpoint_path=args.checkpoint,
            num_agents=args.agents,
            map_type=args.map_type,
            num_episodes=args.episodes,
            save_frames=args.save_frames
        )


if __name__ == "__main__":
    main()
