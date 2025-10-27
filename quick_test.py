"""
Grid-Size Generalization Testing for FCN Agent

Tests trained FCN agent on different grid sizes with mathematically-scaled
hyperparameters to validate Spatial Softmax grid-invariance.

Three scaling approaches:
- Conservative: Maintains difficulty (harder test)
- Balanced: Similar difficulty to training (recommended)
- Aggressive: Easier task (generous scaling)

Usage:
    # Test on 50×50 with balanced scaling (recommended)
    python quick_test.py --checkpoint ./checkpoints/fcn_checkpoint_ep1600.pt \
                          --grid-size 50 --episodes 20

    # Multi-scale test (20, 30, 40, 50)
    python quick_test.py --checkpoint ./checkpoints/fcn_checkpoint_ep1600.pt \
                          --mode multi --episodes-per-size 10

    # Test with specific scaling approach
    python quick_test.py --checkpoint ./checkpoints/fcn_checkpoint_ep1600.pt \
                          --grid-size 50 --scaling aggressive
"""

import os
import time
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Optional

from config import config
from environment import CoverageEnvironment
from fcn_agent import FCNAgent


@dataclass
class ScaledConfig:
    """Configuration scaled for different grid sizes."""
    grid_size: int
    max_steps: int
    sensor_range: float
    num_rays: int
    samples_per_ray: int
    sigmoid_steepness: float
    sigmoid_center: float
    coverage_increment: float
    name: str


def compute_scaled_config(
    base_size: int = 20,
    target_size: int = 50,
    approach: str = 'balanced'
) -> ScaledConfig:
    """
    Compute scaled configuration for target grid size.
    
    Args:
        base_size: Training grid size (20)
        target_size: Target test grid size (e.g., 50)
        approach: 'conservative', 'balanced', or 'aggressive'
        
    Returns:
        ScaledConfig with computed parameters
    """
    # Base configuration (20×20 training)
    base_max_steps = 350
    base_sensor_range = 5.0
    base_num_rays = 32
    base_samples_per_ray = 10
    base_sigmoid_center = 1.5
    
    # Compute scaling factors
    linear_ratio = target_size / base_size  # 2.5 for 50/20
    area_ratio = linear_ratio ** 2  # 6.25 for 50×50 vs 20×20
    
    if approach == 'conservative':
        # Conservative: Minimal scaling (maintains difficulty)
        sensor_scale = linear_ratio ** 0.3  # 1.32× for 50/20
        max_steps = int(base_max_steps * linear_ratio)  # 875
        
    elif approach == 'balanced':
        # Balanced: Moderate scaling (recommended)
        sensor_scale = linear_ratio ** 0.4  # 1.55× for 50/20
        max_steps = int(base_max_steps * linear_ratio * 1.03)  # 900 (with 3% buffer)
        
    elif approach == 'aggressive':
        # Aggressive: Generous scaling (easier task)
        sensor_scale = linear_ratio ** 0.5  # 1.58× for 50/20
        max_steps = int(base_max_steps * linear_ratio * 1.14)  # 1000
        
    else:
        raise ValueError(f"Unknown approach: {approach}")
    
    # Compute scaled parameters
    sensor_range = base_sensor_range * sensor_scale
    num_rays = int(base_num_rays * sensor_scale)
    samples_per_ray = int(base_samples_per_ray * sensor_scale)
    sigmoid_center = base_sigmoid_center * sensor_scale
    
    # Round to nice numbers
    sensor_range = round(sensor_range, 1)
    sigmoid_center = round(sigmoid_center, 1)
    
    return ScaledConfig(
        grid_size=target_size,
        max_steps=max_steps,
        sensor_range=sensor_range,
        num_rays=num_rays,
        samples_per_ray=samples_per_ray,
        sigmoid_steepness=2.0,  # Intrinsic property (don't scale)
        sigmoid_center=sigmoid_center,
        coverage_increment=0.3,  # Intrinsic property (don't scale)
        name=f"{approach.capitalize()}-{target_size}x{target_size}"
    )


def apply_scaled_config(scaled_cfg: ScaledConfig):
    """
    Apply scaled configuration to global config.
    
    Args:
        scaled_cfg: Scaled configuration to apply
    """
    # Backup original config
    original = {
        'GRID_SIZE': config.GRID_SIZE,
        'MAX_EPISODE_STEPS': config.MAX_EPISODE_STEPS,
        'SENSOR_RANGE': config.SENSOR_RANGE,
        'NUM_RAYS': config.NUM_RAYS,
        'SAMPLES_PER_RAY': config.SAMPLES_PER_RAY,
    }
    
    # Apply scaled config
    config.GRID_SIZE = scaled_cfg.grid_size
    config.MAX_EPISODE_STEPS = scaled_cfg.max_steps
    config.SENSOR_RANGE = scaled_cfg.sensor_range
    config.NUM_RAYS = scaled_cfg.num_rays
    config.SAMPLES_PER_RAY = scaled_cfg.samples_per_ray
    
    return original


def restore_config(original: dict):
    """Restore original configuration."""
    config.GRID_SIZE = original['GRID_SIZE']
    config.MAX_EPISODE_STEPS = original['MAX_EPISODE_STEPS']
    config.SENSOR_RANGE = original['SENSOR_RANGE']
    config.NUM_RAYS = original['NUM_RAYS']
    config.SAMPLES_PER_RAY = original['SAMPLES_PER_RAY']


def test_single_grid_size(
    checkpoint_path: str,
    grid_size: int,
    episodes: int = 20,
    scaling: str = 'balanced',
    verbose: bool = True
) -> dict:
    """
    Test agent on single grid size with scaled configuration.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        grid_size: Grid size to test
        episodes: Number of test episodes
        scaling: Scaling approach ('conservative', 'balanced', 'aggressive')
        verbose: Print progress
        
    Returns:
        results: Dict with coverage results per map type
    """
    # Load agent (always initialize with training size)
    training_size = 20
    agent = FCNAgent(grid_size=training_size)
    agent.load(checkpoint_path)
    agent.set_epsilon(0.0)  # Pure greedy
    
    # Compute scaled configuration
    scaled_cfg = compute_scaled_config(
        base_size=training_size,
        target_size=grid_size,
        approach=scaling
    )
    
    if verbose:
        print("=" * 80)
        print(f"TESTING ON {grid_size}×{grid_size} GRID ({scaled_cfg.name})")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Scaling: {scaling}")
        print(f"Episodes: {episodes}")
        print(f"\nScaled Parameters:")
        print(f"  Max steps:       {scaled_cfg.max_steps} (base: 350)")
        print(f"  Sensor range:    {scaled_cfg.sensor_range:.1f} (base: 5.0)")
        print(f"  Num rays:        {scaled_cfg.num_rays} (base: 32)")
        print(f"  Samples/ray:     {scaled_cfg.samples_per_ray} (base: 10)")
        print(f"  Sigmoid center:  {scaled_cfg.sigmoid_center:.1f} (base: 1.5)")
        print("=" * 80)
    
    # Apply scaled config
    original_cfg = apply_scaled_config(scaled_cfg)
    
    try:
        map_types = ['empty', 'random', 'room']
        results = {}
        
        for map_type in map_types:
            coverages = []
            episode_times = []
            
            if verbose:
                print(f"\nTesting {map_type} maps:")
            
            for ep in range(episodes):
                env = CoverageEnvironment(grid_size=grid_size, map_type=map_type)
                state = env.reset()
                
                episode_start = time.time()
                
                for step in range(scaled_cfg.max_steps):
                    action = agent.select_action(state, env.world_state)
                    state, reward, done, info = env.step(action)
                    
                    if done:
                        break
                
                episode_time = time.time() - episode_start
                final_coverage = info.get('coverage_pct', 0.0)
                
                coverages.append(final_coverage)
                episode_times.append(episode_time)
                
                if verbose and (ep + 1) % 5 == 0:
                    avg_cov = np.mean(coverages)
                    avg_time = np.mean(episode_times)
                    print(f"  Episode {ep+1}/{episodes}: "
                          f"{final_coverage:5.1%}, "
                          f"Avg {avg_cov:5.1%} "
                          f"({avg_time:5.1f}s/ep)")
            
            results[map_type] = {
                'mean': np.mean(coverages),
                'std': np.std(coverages),
                'time': np.mean(episode_times),
                'all': coverages
            }
            
            if verbose:
                print(f"  Final {map_type:10s}: "
                      f"{results[map_type]['mean']:5.1%} ± "
                      f"{results[map_type]['std']:5.1%} "
                      f"(avg: {results[map_type]['time']:5.1f}s)")
        
        # Average across map types
        results['avg'] = np.mean([results[mt]['mean'] for mt in map_types])
        
        if verbose:
            print(f"\n{'='*80}")
            print("SUMMARY")
            print(f"{'='*80}")
            print(f"Empty Grid:   {results['empty']['mean']:5.1%} ± {results['empty']['std']:5.1%}")
            print(f"Random Obs:   {results['random']['mean']:5.1%} ± {results['random']['std']:5.1%}")
            print(f"Rooms:        {results['room']['mean']:5.1%} ± {results['room']['std']:5.1%}")
            print(f"Average:      {results['avg']:5.1%}")
            print(f"{'='*80}\n")
        
        return results
        
    finally:
        # Always restore original config
        restore_config(original_cfg)


def test_multi_scale(
    checkpoint_path: str,
    grid_sizes: list = [20, 30, 40, 50],
    episodes_per_size: int = 10,
    scaling: str = 'balanced',
    verbose: bool = True
) -> dict:
    """
    Test agent on multiple grid sizes.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        grid_sizes: List of grid sizes to test
        episodes_per_size: Episodes per grid size
        scaling: Scaling approach
        verbose: Print progress
        
    Returns:
        results: Dict with results for each grid size
    """
    if verbose:
        print("=" * 80)
        print("MULTI-SCALE GENERALIZATION TEST")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Grid sizes: {grid_sizes}")
        print(f"Episodes per size: {episodes_per_size}")
        print(f"Scaling approach: {scaling}")
        print("=" * 80)
    
    all_results = {}
    
    for size in grid_sizes:
        results = test_single_grid_size(
            checkpoint_path=checkpoint_path,
            grid_size=size,
            episodes=episodes_per_size,
            scaling=scaling,
            verbose=verbose
        )
        all_results[size] = results
    
    # Print comparison table
    if verbose:
        print_comparison_table(all_results, grid_sizes[0])
    
    return all_results


def print_comparison_table(results: dict, training_size: int = 20):
    """
    Print comparison table of results across grid sizes.
    
    Args:
        results: Results dict from test_multi_scale
        training_size: Grid size used during training
    """
    print("\n" + "=" * 80)
    print("GRID-SIZE GENERALIZATION SUMMARY")
    print("=" * 80)
    
    grid_sizes = sorted(results.keys())
    map_types = ['empty', 'random', 'room', 'avg']
    
    # Header
    print(f"\n{'Grid Size':<12}", end='')
    for mt in map_types:
        print(f"{mt.capitalize():>12}", end='')
    print(f"{'Time/Ep':>12}  {'Retention':>10}")
    print("-" * 90)
    
    # Training size (baseline)
    baseline_results = results[training_size]
    baseline_avg = baseline_results['avg']
    
    for size in grid_sizes:
        size_results = results[size]
        
        marker = " *" if size == training_size else "  "
        print(f"{size}×{size}{marker}", end='')
        
        for mt in map_types[:-1]:
            coverage = size_results[mt]['mean']
            print(f"{coverage:>11.1%}", end='')
        
        print(f"{size_results['avg']:>11.1%}", end='')
        
        avg_time = np.mean([size_results[mt]['time'] for mt in map_types[:-1]])
        print(f"{avg_time:>11.1f}s", end='')
        
        # Retention percentage
        if size == training_size:
            print(f"{'100.0%':>10}")
        else:
            retention = (size_results['avg'] / baseline_avg) * 100
            print(f"{retention:>9.1f}%")
    
    print("-" * 90)
    print("* = Training grid size")
    
    # Generalization analysis
    print("\n" + "=" * 80)
    print("GENERALIZATION ANALYSIS")
    print("=" * 80)
    
    for size in grid_sizes:
        if size == training_size:
            continue
        
        test_coverage = results[size]['avg']
        retention = (test_coverage / baseline_avg) * 100
        area_ratio = (size / training_size) ** 2
        
        print(f"\n{size}×{size} Grid:")
        print(f"  Coverage:    {test_coverage:5.1%} (vs {baseline_avg:5.1%} on {training_size}×{training_size})")
        print(f"  Retention:   {retention:5.1f}% of training performance")
        print(f"  Area ratio:  {area_ratio:5.1f}× larger")
        
        if retention >= 90:
            status = "✅ EXCELLENT generalization"
        elif retention >= 80:
            status = "✅ GOOD generalization"
        elif retention >= 70:
            status = "⚠️  MODERATE generalization"
        else:
            status = "❌ POOR generalization"
        
        print(f"  Status:      {status}")
    
    print("=" * 80)


def compare_scaling_approaches(
    checkpoint_path: str,
    grid_size: int = 50,
    episodes: int = 10,
    verbose: bool = True
) -> dict:
    """
    Compare conservative, balanced, and aggressive scaling approaches.
    
    Args:
        checkpoint_path: Path to trained checkpoint
        grid_size: Grid size to test
        episodes: Number of episodes per approach
        verbose: Print progress
        
    Returns:
        results: Dict with results for each approach
    """
    approaches = ['conservative', 'balanced', 'aggressive']
    all_results = {}
    
    if verbose:
        print("=" * 80)
        print(f"SCALING APPROACH COMPARISON - {grid_size}×{grid_size} GRID")
        print("=" * 80)
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Episodes per approach: {episodes}")
        print("=" * 80)
    
    for approach in approaches:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Testing {approach.upper()} approach...")
            print(f"{'='*80}")
        
        results = test_single_grid_size(
            checkpoint_path=checkpoint_path,
            grid_size=grid_size,
            episodes=episodes,
            scaling=approach,
            verbose=verbose
        )
        all_results[approach] = results
    
    # Print comparison
    if verbose:
        print("\n" + "=" * 80)
        print("APPROACH COMPARISON")
        print("=" * 80)
        print(f"\n{'Approach':<15}{'Empty':>12}{'Random':>12}{'Rooms':>12}{'Average':>12}")
        print("-" * 63)
        
        for approach in approaches:
            r = all_results[approach]
            print(f"{approach.capitalize():<15}"
                  f"{r['empty']['mean']:>11.1%}"
                  f"{r['random']['mean']:>11.1%}"
                  f"{r['room']['mean']:>11.1%}"
                  f"{r['avg']:>11.1%}")
        
        print("-" * 63)
        print("\nRecommendation:")
        
        # Find best approach
        best = max(approaches, key=lambda a: all_results[a]['avg'])
        print(f"  Best performance: {best.capitalize()} ({all_results[best]['avg']:.1%})")
        print(f"  For validation:   Use 'balanced' for fair comparison")
        print(f"  For deployment:   Use '{best}' for best results")
        print("=" * 80)
    
    return all_results


# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Test FCN grid-size generalization with scaled hyperparameters',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test on 50×50 with balanced scaling (recommended)
  python quick_test.py --checkpoint ./checkpoints/fcn_checkpoint_ep1600.pt \\
                        --grid-size 50 --episodes 20
  
  # Multi-scale test (20, 30, 40, 50)
  python quick_test.py --checkpoint ./checkpoints/fcn_checkpoint_ep1600.pt \\
                        --mode multi --episodes-per-size 10
  
  # Compare scaling approaches
  python quick_test.py --checkpoint ./checkpoints/fcn_checkpoint_ep1600.pt \\
                        --mode compare --grid-size 50 --episodes 10
        """
    )
    
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to trained FCN checkpoint')
    
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'multi', 'compare'],
                       help='Test mode: single size, multi-scale, or compare approaches')
    
    # Single test arguments
    parser.add_argument('--grid-size', type=int, default=50,
                       help='Grid size for single test (default: 50)')
    
    parser.add_argument('--episodes', type=int, default=20,
                       help='Episodes for single test (default: 20)')
    
    parser.add_argument('--scaling', type=str, default='balanced',
                       choices=['conservative', 'balanced', 'aggressive'],
                       help='Scaling approach (default: balanced)')
    
    # Multi-scale arguments
    parser.add_argument('--grid-sizes', type=int, nargs='+',
                       default=[20, 30, 40, 50],
                       help='Grid sizes for multi-scale test (default: 20 30 40 50)')
    
    parser.add_argument('--episodes-per-size', type=int, default=10,
                       help='Episodes per grid size for multi-scale (default: 10)')
    
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress verbose output')
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: Checkpoint not found: {args.checkpoint}")
        exit(1)
    
    verbose = not args.quiet
    
    # Run appropriate test mode
    if args.mode == 'single':
        results = test_single_grid_size(
            checkpoint_path=args.checkpoint,
            grid_size=args.grid_size,
            episodes=args.episodes,
            scaling=args.scaling,
            verbose=verbose
        )
        
    elif args.mode == 'multi':
        results = test_multi_scale(
            checkpoint_path=args.checkpoint,
            grid_sizes=args.grid_sizes,
            episodes_per_size=args.episodes_per_size,
            scaling=args.scaling,
            verbose=verbose
        )
        
    elif args.mode == 'compare':
        results = compare_scaling_approaches(
            checkpoint_path=args.checkpoint,
            grid_size=args.grid_size,
            episodes=args.episodes,
            verbose=verbose
        )
    
    if verbose:
        print("\n✅ Testing completed successfully!")
