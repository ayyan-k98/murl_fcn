"""
Training Loop for FCN Agent with Comprehensive Visualization

Extends train_fcn.py with:
- Static visualizations during training (every N episodes)
- Full validation visualizations with GIF generation
- Coverage trajectory analysis
"""

import os
import sys
import time
import argparse
from typing import Optional
import numpy as np

from config import config
from data_structures import CoverageMetrics, RobotState, WorldState
from environment import CoverageEnvironment
from fcn_agent import FCNAgent
from curriculum import CurriculumManager
from visualization import CoverageVisualizer
import utils


def train_with_visualization(
    num_episodes: int = 1600,
    grid_size: int = 20,
    validate_interval: int = 50,
    checkpoint_interval: int = 100,
    viz_interval: int = 50,  # Visualize training every N episodes
    resume_from: Optional[str] = None,
    verbose: bool = True,
    use_6ch: bool = False,
    save_dir: str = "training_visualizations"
) -> tuple:
    """
    Train with comprehensive visualization.
    
    Args:
        num_episodes: Number of episodes to train
        grid_size: Map size
        validate_interval: Validate every N episodes
        checkpoint_interval: Save checkpoint every N episodes
        viz_interval: Create visualizations every N episodes
        resume_from: Path to checkpoint to resume from
        verbose: Print training progress
        use_6ch: Use 6 channels instead of 5
        save_dir: Directory for visualizations
        
    Returns:
        agent: Trained FCN agent
        metrics: Training metrics
    """
    # Initialize visualizer
    visualizer = CoverageVisualizer(save_dir=save_dir)
    
    if verbose:
        print("="*80)
        print("FCN AGENT TRAINING WITH VISUALIZATION")
        print("="*80)
        env_type = "Probabilistic" if config.USE_PROBABILISTIC_ENV else "Binary"
        print(f"Environment: {env_type} Coverage")
        if config.USE_PROBABILISTIC_ENV:
            print(f"  Sigmoid parameters: k={config.PROBABILISTIC_COVERAGE_STEEPNESS}, r0={config.PROBABILISTIC_COVERAGE_MIDPOINT}")
        print(f"Grid Size: {grid_size}x{grid_size}")
        print(f"Episodes: {num_episodes}")
        print(f"Validation Interval: {validate_interval}")
        print(f"Visualization Interval: {viz_interval}")
        print(f"Save Directory: {save_dir}")
        print("="*80 + "\n")

    # Initialize environment and agent
    env = CoverageEnvironment(grid_size=grid_size, map_type="empty")
    agent = FCNAgent(grid_size=grid_size, use_6ch=use_6ch)
    curriculum = CurriculumManager()
    metrics = CoverageMetrics()

    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from:
        agent.load(resume_from)
        if verbose:
            print(f"✓ Resumed from checkpoint: {resume_from}\n")

    # Training loop
    try:
        for episode in range(start_episode, num_episodes):
            # Get curriculum phase
            phase = curriculum.get_current_phase(episode)
            map_type = phase.map_type

            # Reset environment
            env.map_type = map_type
            state = env.reset()
            
            # Track trajectory for visualization
            trajectory = [state.position]
            episode_reward = 0
            done = False
            step_count = 0

            while not done and step_count < config.MAX_EPISODE_STEPS:
                # Select action
                action = agent.select_action(state, env.world_state)

                # Execute action
                next_state, reward, done, info = env.step(action)
                trajectory.append(next_state.position)
                episode_reward += reward

                # Store experience
                agent.store_experience(state, action, reward, next_state, done)

                # Train agent
                agent.train_step()

                state = next_state
                step_count += 1

            # Record metrics
            coverage_pct = info['coverage_percentage']
            metrics.record_episode(episode_reward, coverage_pct / 100, step_count)
            metrics.record_epsilon(agent.epsilon)

            # Decay epsilon
            agent.decay_epsilon(phase.epsilon_decay)

            # Verbose output
            if verbose and episode % 10 == 0:
                recent_cov = np.mean(metrics.episode_coverages[-10:]) * 100
                recent_reward = np.mean(metrics.episode_rewards[-10:])
                print(f"Episode {episode+1:4d} | Phase {phase.name:8s} | "
                      f"Cov: {coverage_pct:5.1f}% (avg: {recent_cov:5.1f}%) | "
                      f"Reward: {episode_reward:7.2f} (avg: {recent_reward:7.2f}) | "
                      f"Steps: {step_count:3d} | Eps: {agent.epsilon:.3f}")

            # Create training visualization
            if (episode + 1) % viz_interval == 0:
                if verbose:
                    print(f"\n  Creating visualization for episode {episode+1}...")
                
                visualizer.plot_episode_static(
                    world_state=env.world_state,
                    robot_state=env.robot_state,
                    trajectory=trajectory,
                    episode_num=episode + 1,
                    coverage_pct=coverage_pct,
                    episode_reward=episode_reward,
                    mode="train",
                    map_type=map_type,
                    save=True,
                    show=False
                )

            # Validation with full visualization
            if (episode + 1) % validate_interval == 0:
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"VALIDATION @ Episode {episode + 1}")
                    print(f"{'='*80}")
                
                val_results = validate_with_visualization(
                    agent=agent,
                    visualizer=visualizer,
                    grid_size=grid_size,
                    episode_num=episode + 1,
                    verbose=verbose,
                    use_6ch=use_6ch
                )
                
                metrics.validation_scores[episode + 1] = val_results
                
                if verbose:
                    print(f"  Empty Grid:   {val_results['empty']:.1%}")
                    print(f"  Random Obs:   {val_results['random']:.1%}")
                    print(f"  Rooms:        {val_results['room']:.1%}")
                    print(f"  Average:      {val_results['avg']:.1%}")
                    print(f"{'='*80}\n")

            # Update target network
            if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()

            # Save checkpoint
            if (episode + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(
                    config.CHECKPOINT_DIR,
                    f"fcn_checkpoint_ep{episode + 1}.pt"
                )
                agent.save(checkpoint_path)
                if verbose:
                    print(f"  ✓ Checkpoint saved: {checkpoint_path}")

        # Training completed
        if verbose:
            print("\n" + "="*80)
            print("TRAINING COMPLETED")
            print("="*80)
            print(f"Total episodes: {num_episodes}")
            print(f"Final epsilon: {agent.epsilon:.3f}")
            print(f"Average coverage (last 100): {np.mean(metrics.episode_coverages[-100:]):.1%}")
            print("="*80)

    except KeyboardInterrupt:
        if verbose:
            print("\n\n⚠ Training interrupted by user")
            print(f"Completed {episode} episodes")

    return agent, metrics


def validate_with_visualization(
    agent: FCNAgent,
    visualizer: CoverageVisualizer,
    grid_size: int,
    episode_num: int,
    verbose: bool = True,
    use_6ch: bool = False,
    create_gifs: bool = True
) -> dict:
    """
    Validate agent with full visualization (static images + GIFs).
    
    Args:
        agent: Trained agent
        visualizer: Visualizer instance
        grid_size: Map size
        episode_num: Current training episode
        verbose: Print progress
        use_6ch: Use 6 channels
        create_gifs: Create animated GIFs
        
    Returns:
        dict: Validation results {map_type: coverage}
    """
    map_types = ['empty', 'random', 'room']
    results = {}
    
    # Validation with low epsilon
    original_epsilon = agent.epsilon
    agent.epsilon = config.VALIDATION_EPSILON
    
    for map_type in map_types:
        if verbose:
            print(f"  Validating on {map_type}...", end=" ", flush=True)
        
        env = CoverageEnvironment(grid_size=grid_size, map_type=map_type)
        coverages = []
        
        # Run validation episodes
        num_val_episodes = config.VALIDATION_EPISODES
        
        for val_ep in range(num_val_episodes):
            state = env.reset()
            trajectory = [state.position]
            frames = []  # For GIF creation
            episode_reward = 0
            done = False
            step_count = 0
            
            max_steps = config.VALIDATION_MAX_STEPS if config.FAST_VALIDATION else config.MAX_EPISODE_STEPS
            
            while not done and step_count < max_steps:
                # Store frame for GIF (sample every 5 steps)
                if create_gifs and val_ep == 0 and step_count % 5 == 0:
                    # Deep copy states for frame
                    import copy
                    frames.append({
                        'world_state': copy.deepcopy(env.world_state),
                        'robot_state': copy.deepcopy(env.robot_state),
                        'step': step_count,
                        'coverage': env.world_state.coverage_map.sum() / (env.world_state.grid_size ** 2 - len(env.world_state.obstacles))
                    })
                
                action = agent.select_action(state, env.world_state, deterministic=False)
                next_state, reward, done, info = env.step(action)
                trajectory.append(next_state.position)
                episode_reward += reward
                state = next_state
                step_count += 1
            
            coverage = info['coverage_percentage'] / 100
            coverages.append(coverage)
            
            # Create static visualization for first validation episode
            if val_ep == 0:
                visualizer.plot_episode_static(
                    world_state=env.world_state,
                    robot_state=env.robot_state,
                    trajectory=trajectory,
                    episode_num=episode_num,
                    coverage_pct=info['coverage_percentage'],
                    episode_reward=episode_reward,
                    mode="validation",
                    map_type=map_type,
                    save=True,
                    show=False
                )
                
                # Create GIF
                if create_gifs and len(frames) > 0:
                    try:
                        visualizer.create_episode_gif(
                            frames=frames,
                            episode_num=episode_num,
                            mode=f"validation_{map_type}",
                            fps=5
                        )
                    except Exception as e:
                        if verbose:
                            print(f"\n  ⚠ Warning: GIF creation failed: {e}")
        
        avg_coverage = np.mean(coverages)
        results[map_type] = avg_coverage
        
        if verbose:
            print(f"{avg_coverage:.1%}")
    
    # Restore original epsilon
    agent.epsilon = original_epsilon
    
    # Calculate average
    results['avg'] = np.mean([results[mt] for mt in map_types])
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train FCN agent with visualization')
    parser.add_argument('--episodes', type=int, default=400,
                       help='Number of episodes (default: 400)')
    parser.add_argument('--grid-size', type=int, default=20,
                       help='Grid size (default: 20)')
    parser.add_argument('--validate-interval', type=int, default=50,
                       help='Validation interval (default: 50)')
    parser.add_argument('--viz-interval', type=int, default=50,
                       help='Visualization interval (default: 50)')
    parser.add_argument('--checkpoint-interval', type=int, default=100,
                       help='Checkpoint interval (default: 100)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint')
    parser.add_argument('--save-dir', type=str, default='training_visualizations',
                       help='Visualization save directory')
    parser.add_argument('--probabilistic', action='store_true',
                       help='Use probabilistic coverage environment')
    parser.add_argument('--6ch', action='store_true',
                       help='Use 6 channels instead of 5')
    
    args = parser.parse_args()
    
    # Set probabilistic environment
    if args.probabilistic:
        config.USE_PROBABILISTIC_ENV = True
    
    # Train
    agent, metrics = train_with_visualization(
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        validate_interval=args.validate_interval,
        checkpoint_interval=args.checkpoint_interval,
        viz_interval=args.viz_interval,
        resume_from=args.resume,
        verbose=True,
        use_6ch=args.__dict__['6ch'],
        save_dir=args.save_dir
    )
    
    # Save final model and metrics
    final_model_path = os.path.join(config.CHECKPOINT_DIR, "fcn_final.pt")
    agent.save(final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    metrics_path = os.path.join(config.CHECKPOINT_DIR, "fcn_metrics.pkl")
    utils.save_metrics(metrics, metrics_path)
    
    # Create final plots
    print("\n" + "="*80)
    print("GENERATING FINAL PLOTS")
    print("="*80)
    
    plots_dir = os.path.join(args.save_dir, "final_plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Training curves
    utils.plot_training_curves(
        metrics,
        save_path=os.path.join(plots_dir, "training_curves.png"),
        show=False
    )
    
    # Validation results
    if len(metrics.validation_scores) > 0:
        utils.plot_validation_results(
            metrics,
            save_path=os.path.join(plots_dir, "validation_results.png"),
            show=False
        )
    
    # Curriculum phases
    curriculum = CurriculumManager()
    utils.plot_curriculum_phases(
        metrics,
        curriculum,
        save_path=os.path.join(plots_dir, "curriculum_phases.png"),
        show=False
    )
    
    print(f"✓ Final plots saved to: {plots_dir}")
    print("="*80)
    
    # Final statistics
    utils.print_statistics(metrics, window=100)
    
    print("\n✓ Training with visualization complete!")
