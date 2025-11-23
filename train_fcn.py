"""
Training Loop for FCN Agent (Stage 1)

Single-agent training with FCN + Spatial Softmax architecture.
Drop-in replacement for GAT training script.

Key Differences from train.py:
- Uses FCNAgent instead of CoverageAgent (GAT)
- Direct grid encoding (no graph construction)
- 3× faster per episode (no graph overhead)
- Grid-size invariant (can change grid size during training)
"""

import os
import time
from typing import Optional
import numpy as np

from config import config
from data_structures import CoverageMetrics
from environment import CoverageEnvironment
from fcn_agent import FCNAgent
from curriculum import CurriculumManager


def train_fcn_stage1(
    num_episodes: int = 1600,
    grid_size: int = 20,
    validate_interval: int = 50,
    checkpoint_interval: int = 100,
    resume_from: Optional[str] = None,
    verbose: bool = True,
    use_6ch: bool = False
) -> tuple:
    """
    Train Stage 1 using FCN + Spatial Softmax architecture.

    Args:
        num_episodes: Number of episodes to train
        grid_size: Map size
        validate_interval: Validate every N episodes
        checkpoint_interval: Save checkpoint every N episodes
        resume_from: Path to checkpoint to resume from
        verbose: Print training progress
        use_6ch: Use 6 channels (dummy zeros for ch5) instead of 5

    Returns:
        agent: Trained FCN agent
        metrics: Training metrics
    """
    if verbose:
        print("=" * 80)
        print("STAGE 1: FCN + SPATIAL SOFTMAX TRAINING")
        print("=" * 80)
        print(f"Episodes: {num_episodes}")
        print(f"Grid size: {grid_size}")
        print(f"Device: {config.DEVICE}")
        print(f"Architecture: FCN + Spatial Softmax (grid-invariant)")
        print(f"Input channels: {6 if use_6ch else 5} ({'6ch with dummy' if use_6ch else '5ch baseline'})")
        if config.USE_PROBABILISTIC_ENV:
            print(f"Environment: PROBABILISTIC (sigmoid coverage)")
        else:
            print(f"Environment: BINARY (instant coverage)")
        print("=" * 80)

    # Initialize components
    agent = FCNAgent(grid_size=grid_size, input_channels=6 if use_6ch else 5)
    curriculum = CurriculumManager()
    metrics = CoverageMetrics()

    # Create checkpoint directory
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    # Resume from checkpoint if specified
    start_episode = 0
    if resume_from is not None:
        agent.load(resume_from)
        # Note: FCNAgent doesn't store episode number in checkpoint
        # You may want to parse it from filename if needed
        if verbose:
            print(f"✓ Resumed from checkpoint: {resume_from}")

    # Print curriculum overview
    if verbose and start_episode == 0:
        print("\n" + curriculum.get_summary())
        print()

    # Track phase transitions for epsilon resets
    previous_phase_idx = curriculum.get_current_phase(start_episode).start_ep

    # Training loop with error handling
    try:
        for episode in range(start_episode, num_episodes):
            episode_start_time = time.time()

            # Get map type from curriculum
            map_type = curriculum.get_map_type(episode)

            # Check for phase transition
            current_phase = curriculum.get_current_phase(episode)
            if current_phase.start_ep != previous_phase_idx:
                # New phase started - reset epsilon to enable exploration
                phase_duration = current_phase.end_ep - current_phase.start_ep
                epsilon_start = 0.5  # Start at moderate exploration level
                agent.set_epsilon(epsilon_start)
                previous_phase_idx = current_phase.start_ep
                
                if verbose:
                    print(f"\n{'='*80}")
                    print(f"PHASE TRANSITION → {current_phase.name}")
                    print(f"Episodes: {current_phase.start_ep}-{current_phase.end_ep} ({phase_duration} episodes)")
                    print(f"Epsilon reset: {agent.epsilon:.3f} → will decay to {current_phase.epsilon_floor:.3f}")
                    print(f"{'='*80}\n")

            # Get phase-specific epsilon parameters from curriculum
            epsilon_floor = curriculum.get_epsilon_floor(episode)
            epsilon_decay = curriculum.get_epsilon_decay(episode)

            # NOTE: epsilon_floor is enforced AFTER decay, not before
            # This allows epsilon to decay naturally within each phase

            # Create environment
            env = CoverageEnvironment(grid_size=grid_size, map_type=map_type)
            state = env.reset()

            # Episode loop
            episode_reward = 0
            episode_loss = []

            # Timing breakdown (only for first few episodes)
            enable_timing = config.ENABLE_TIMING_BREAKDOWN and episode < 5
            time_encoding = 0 if enable_timing else None
            time_action = 0 if enable_timing else None
            time_env = 0 if enable_timing else None
            time_train = 0 if enable_timing else None

            # Prepare dummy occupancy channel if using 6 channels
            dummy_occupancy = None
            if use_6ch:
                dummy_occupancy = np.zeros((grid_size, grid_size), dtype=np.float32)

            for step in range(config.MAX_EPISODE_STEPS):
                # Encode state to grid (ONCE per step)
                if enable_timing:
                    t0 = time.time()
                grid_tensor = agent._encode_state(state, env.world_state, dummy_occupancy)
                if enable_timing:
                    time_encoding += time.time() - t0

                # Select action using pre-encoded grid
                if enable_timing:
                    t0 = time.time()
                action = agent.select_action_from_tensor(grid_tensor)
                if enable_timing:
                    time_action += time.time() - t0

                # Step environment
                if enable_timing:
                    t0 = time.time()
                next_state, reward, done, info = env.step(action)
                episode_reward += reward

                # Encode next state
                next_grid_tensor = agent._encode_state(next_state, env.world_state, dummy_occupancy)
                if enable_timing:
                    time_env += time.time() - t0

                # Store transition
                agent.store_transition(
                    grid_tensor, action, reward, next_grid_tensor, done, info
                )

                # Optimize every TRAIN_FREQ steps
                if step % config.TRAIN_FREQ == 0 and len(agent.memory) >= config.MIN_REPLAY_SIZE:
                    if enable_timing:
                        t0 = time.time()
                    loss = agent.optimize()
                    if enable_timing:
                        time_train += time.time() - t0
                    if loss is not None:
                        episode_loss.append(loss)
                        metrics.add_loss(loss)

                state = next_state

                if done:
                    break

            # Episode completed
            episode_time = time.time() - episode_start_time
            final_coverage = info.get('coverage_pct', 0.0)

            # Decay epsilon FIRST
            agent.decay_epsilon(decay_rate=epsilon_decay)
            
            # Then enforce floor (prevents going below minimum exploration)
            if agent.epsilon < epsilon_floor:
                agent.epsilon = epsilon_floor

            # Record metrics (using add_episode which takes all params)
            metrics.add_episode(
                reward=episode_reward,
                coverage=final_coverage,
                length=step + 1,
                epsilon=agent.epsilon
            )

            # Timing breakdown (first few episodes)
            if enable_timing:
                print(f"\n⏱ Timing Breakdown (Episode {episode}):")
                print(f"  Encoding: {time_encoding:.3f}s ({100*time_encoding/episode_time:.1f}%)")
                print(f"  Action:   {time_action:.3f}s ({100*time_action/episode_time:.1f}%)")
                print(f"  Env:      {time_env:.3f}s ({100*time_env/episode_time:.1f}%)")
                print(f"  Train:    {time_train:.3f}s ({100*time_train/episode_time:.1f}%)")
                print(f"  Total:    {episode_time:.3f}s")

            # Print progress
            if verbose and (episode + 1) % config.LOG_INTERVAL == 0:
                avg_coverage_10 = np.mean(metrics.episode_coverages[-10:]) if len(metrics.episode_coverages) >= 10 else np.mean(metrics.episode_coverages) if metrics.episode_coverages else 0.0
                avg_loss = np.mean(episode_loss) if episode_loss else 0.0

                print(f"Ep {episode + 1:4d}/{num_episodes} | "
                      f"Cov: {final_coverage:5.1%} (avg: {avg_coverage_10:5.1%}) | "
                      f"R: {episode_reward:7.1f} | "
                      f"ε: {agent.epsilon:.3f} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time: {episode_time:.1f}s")

                # Gradient monitoring
                if len(agent.grad_norm_history) > 0:
                    recent_grad_norm = agent.grad_norm_history[-1]
                    if recent_grad_norm > 15.0:
                        print(f"  ⚠ High gradient norm: {recent_grad_norm:.1f}")

            # Validation
            if (episode + 1) % validate_interval == 0:
                val_results = validate_fcn(agent, grid_size, verbose=verbose, use_6ch=use_6ch)
                metrics.validation_scores[episode + 1] = val_results

                if verbose:
                    print(f"\n{'='*80}")
                    print(f"VALIDATION @ Episode {episode + 1}")
                    print(f"{'='*80}")
                    print(f"  Empty Grid:   {val_results['empty']:.1%}")
                    print(f"  Random Obs:   {val_results['random']:.1%}")
                    print(f"  Rooms:        {val_results['room']:.1%}")
                    print(f"  Average:      {val_results['avg']:.1%}")
                    print(f"{'='*80}\n")

            # Update target network
            if (episode + 1) % config.TARGET_UPDATE_FREQ == 0:
                agent.update_target_network()
                if verbose:
                    print(f"  ✓ Target network updated")

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

        # Save emergency checkpoint
        emergency_path = os.path.join(
            config.CHECKPOINT_DIR,
            f"fcn_emergency_ep{episode}.pt"
        )
        agent.save(emergency_path)
        if verbose:
            print(f"✓ Emergency checkpoint saved: {emergency_path}")

    return agent, metrics


def validate_fcn(
    agent: FCNAgent,
    grid_size: int = 20,
    num_episodes: int = None,
    verbose: bool = True,
    use_6ch: bool = False
) -> dict:
    """
    Validate FCN agent on different map types.

    Args:
        agent: FCN agent to validate
        grid_size: Grid size
        num_episodes: Episodes per map type (default from config)
        verbose: Print validation progress
        use_6ch: If True, use dummy 6th channel (all zeros)

    Returns:
        results: Dictionary with coverage for each map type
    """
    if num_episodes is None:
        num_episodes = config.VALIDATION_EPISODES

    # Save current epsilon
    original_epsilon = agent.epsilon

    # Validation with low epsilon (mostly greedy)
    agent.set_epsilon(0.1)
    
    # Create dummy occupancy if using 6 channels
    dummy_occupancy = None
    if use_6ch:
        dummy_occupancy = np.zeros((grid_size, grid_size), dtype=np.float32)

    map_types = ['empty', 'random', 'room']
    results = {}

    for map_type in map_types:
        coverages = []

        for ep in range(num_episodes):
            env = CoverageEnvironment(grid_size=grid_size, map_type=map_type)
            state = env.reset()

            # Use VALIDATION_MAX_STEPS if fast validation enabled
            max_steps = config.VALIDATION_MAX_STEPS if config.FAST_VALIDATION else config.MAX_EPISODE_STEPS

            for step in range(max_steps):
                action = agent.select_action(state, env.world_state, agent_occupancy=dummy_occupancy)
                state, reward, done, info = env.step(action)

                if done:
                    break

            final_coverage = info.get('coverage_pct', 0.0)
            coverages.append(final_coverage)

        results[map_type] = np.mean(coverages)

    # Average across all map types
    results['avg'] = np.mean([results[mt] for mt in map_types])

    # Restore epsilon
    agent.set_epsilon(original_epsilon)

    return results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Train FCN agent for coverage task')
    parser.add_argument('--episodes', type=int, default=800,
                       help='Number of episodes to train (default: 800)')
    parser.add_argument('--grid-size', type=int, default=20,
                       help='Grid size (default: 20)')
    parser.add_argument('--validate-interval', type=int, default=100,
                       help='Validation interval (default: 100)')
    parser.add_argument('--checkpoint-interval', type=int, default=200,
                       help='Checkpoint interval (default: 200)')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    parser.add_argument('--probabilistic', action='store_true',
                       help='Use probabilistic environment (sigmoid coverage) instead of binary')
    parser.add_argument('--use-6ch', action='store_true',
                       help='Use 6 channels (dummy zeros for ch5) for baseline comparison')

    args = parser.parse_args()

    # Apply probabilistic environment setting if specified
    if args.probabilistic:
        config.USE_PROBABILISTIC_ENV = True

    # Train
    agent, metrics = train_fcn_stage1(
        num_episodes=args.episodes,
        grid_size=args.grid_size,
        validate_interval=args.validate_interval,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume,
        verbose=not args.quiet,
        use_6ch=args.use_6ch
    )

    # Final validation
    print("\n" + "="*80)
    print("FINAL VALIDATION")
    print("="*80)

    final_results = validate_fcn(agent, grid_size=args.grid_size, num_episodes=20, verbose=True, use_6ch=args.use_6ch)

    print(f"\nFinal Results:")
    print(f"  Empty Grid:   {final_results['empty']:.1%}")
    print(f"  Random Obs:   {final_results['random']:.1%}")
    print(f"  Rooms:        {final_results['room']:.1%}")
    print(f"  Average:      {final_results['avg']:.1%}")
    print("="*80)

    # Save final model
    final_path = os.path.join(config.CHECKPOINT_DIR, "fcn_final.pt")
    agent.save(final_path)
    print(f"\n✓ Final model saved: {final_path}")
