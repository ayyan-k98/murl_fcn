"""
Configuration file for Multi-Robot Coverage System

Contains all hyperparameters and settings for the FCN-based coverage agent.
"""

from dataclasses import dataclass
import torch


@dataclass
class Config:
    """Global configuration for the coverage task."""

    # ==================== Environment ====================
    GRID_SIZE: int = 20
    SENSOR_RANGE: float = 4.0  # 🚨 REDUCED from 5.0 - Make task harder
    COMM_RANGE: float = 10.0    # For Stage 2 multi-agent
    NUM_RAYS: int = 12          # 🚨 REDUCED from 16 - Less coverage per step
    SAMPLES_PER_RAY: int = 8    # 🚨 REDUCED from 10 - Coarser sampling
    MAX_EPISODE_STEPS: int = 350
    USE_PROBABILISTIC_ENV: bool = False  # Toggle between binary and probabilistic coverage
    
    # ==================== Multi-Agent Early Termination ====================
    # Early termination for multi-agent ONLY (single-agent trains full episodes)
    # Encourages agents to learn efficient coordination patterns
    ENABLE_EARLY_TERMINATION_MULTI: bool = True          # Enable early termination
    EARLY_TERM_COVERAGE_TARGET_MULTI: float = 0.90       # Terminate when 90% coverage reached
    EARLY_TERM_MIN_STEPS_MULTI: int = 150                # Don't terminate before 150 steps
    EARLY_TERM_COMPLETION_BONUS: float = 10.0            # Flat bonus for completing early
    EARLY_TERM_TIME_BONUS_PER_STEP: float = 0.05         # Additional bonus per step saved

    # ==================== Agent ====================
    N_ACTIONS: int = 9  # 8 directions + stay
    ACTION_NAMES = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'STAY']

    # Action deltas (dx, dy)
    ACTION_DELTAS = [
        (0, -1),   # N
        (1, -1),   # NE
        (1, 0),    # E
        (1, 1),    # SE
        (0, 1),    # S
        (-1, 1),   # SW
        (-1, 0),   # W
        (-1, -1),  # NW
        (0, 0)     # STAY
    ]

    # ==================== Learning (FCN + PROBABILISTIC OPTIMIZED) ====================
    LEARNING_RATE: float = 5e-5        # 🚨 EMERGENCY: Reduced from 1e-4 (2× reduction for stability)
    LEARNING_RATE_MIN: float = 1e-5    # REDUCED from 5e-5
    LR_DECAY_RATE: float = 0.9998      # Slower decay from 0.9995
    GAMMA: float = 0.99               # Discount factor
    BATCH_SIZE: int = 256             # Large batch for stable gradients
    REPLAY_BUFFER_SIZE: int = 50000   # 50k transitions (~200 episodes worth)
    TARGET_UPDATE_FREQ: int = 50      # 🚨 INCREASED from 100 - More frequent for stability
    MIN_REPLAY_SIZE: int = 200        # Start training after 200 transitions (~1 episode)
    TRAIN_FREQ: int = 4               # Train every 4 steps (balance speed vs sample efficiency)
    GRAD_CLIP_NORM: float = 3.0       # 🚨 EMERGENCY: Reduced from 5.0 - Aggressive clipping
    
    # N-step returns for better credit assignment
    N_STEP: int = 3                   # NEW: Use 3-step returns
    N_STEP_ENABLED: bool = True       # Toggle for n-step

    # ==================== Exploration (PHASE-SPECIFIC) ====================
    EPSILON_START: float = 1.0
    EPSILON_MIN: float = 0.05  # Global minimum (was 0.15)
    
    # Phase-specific decay rates (each phase has its own exploration strategy)
    # FIXED: Much faster decay for all phases to enable proper learning
    EPSILON_DECAY_PHASE1: float = 0.98    # FIXED from 0.985 - Reach 35% epsilon by ep 50
    EPSILON_DECAY_PHASE2: float = 0.985   # FIXED from 0.995
    EPSILON_DECAY_PHASE3: float = 0.987   # FIXED from 0.996
    EPSILON_DECAY_PHASE4: float = 0.990   # FIXED from 0.998
    EPSILON_DECAY_PHASE5: float = 0.985   # FIXED from 0.995
    EPSILON_DECAY_PHASE6: float = 0.987   # FIXED from 0.996
    EPSILON_DECAY_PHASE7: float = 0.990   # FIXED from 0.998
    EPSILON_DECAY_PHASE8: float = 0.985   # FIXED from 0.995
    EPSILON_DECAY_PHASE9: float = 0.985   # FIXED from 0.995
    EPSILON_DECAY_PHASE10: float = 0.990  # FIXED from 0.998
    EPSILON_DECAY_PHASE11: float = 0.987  # FIXED from 0.996
    EPSILON_DECAY_PHASE12: float = 0.990  # FIXED from 0.997
    EPSILON_DECAY_PHASE13: float = 0.992  # FIXED from 0.998
    
    # Legacy default (unused with curriculum)
    EPSILON_DECAY_RATE: float = 0.99  # Fallback for non-curriculum training

    # ==================== FCN Architecture ====================
    # FCN + Spatial Softmax: Grid-size invariant CNN architecture
    # Expected performance: 68-73% validation @ 800 episodes (22s/episode)
    CNN_HIDDEN_DIM: int = 128       # Feature channels (128 → 256 coords after spatial softmax)
    CNN_DROPOUT: float = 0.1        # Dropout in decision head
    USE_COORDCONV: bool = True      # Add coordinate channels (x, y) to input
    SPATIAL_SOFTMAX_TEMP: float = 1.0  # Temperature for spatial softmax attention

    # ==================== Rewards (EMERGENCY RESCALING - GRADIENT EXPLOSION FIX) ====================
    # 🚨 CRITICAL: Gradient norm rising: 18→36 (Q-values still accumulating)
    # Additional reduction needed + sensor tuning to make task appropriately challenging
    # 
    # Episode 300: Coverage 95.6%, Gradient 36.7 (exceeding threshold 25.0)
    # → Q-values still too large, need further reward reduction
    #
    # SOLUTION: Additional 40% reward reduction (total 4× from original)
    # Expected episode reward after fix: 750 × 0.6 = ~450 (more conservative)
    
    COVERAGE_REWARD: float = 1.2       # 🚨 FURTHER REDUCED from 2.0 (40% reduction)
    COVERAGE_THRESHOLD: float = 0.85   # 🚨 INCREASED from 0.5 - Cell must reach 85% to count as "covered"
    
    EXPLORATION_REWARD: float = 0.07   # 🚨 Reduced from 0.12 (40% reduction)
    FRONTIER_BONUS: float = 0.012      # 🚨 Reduced from 0.02 (40% reduction)
    FRONTIER_CAP: float = 0.25         # 🚨 Reduced from 0.4 (40% reduction)
    
    # NEW: Rotation penalties for smoother trajectories
    ROTATION_PENALTY_SMALL: float = -0.05   # 45° turn (adjacent action)
    ROTATION_PENALTY_MEDIUM: float = -0.10  # 90° turn
    ROTATION_PENALTY_LARGE: float = -0.15   # 135°-180° turn
    USE_ROTATION_PENALTY: bool = True       # Toggle rotation penalty
    
    COLLISION_PENALTY: float = -0.25   # 🚨 Reduced from -0.4 (40% reduction)
    STEP_PENALTY: float = -0.0012      # 🚨 Reduced from -0.002 (40% reduction)
    STAY_PENALTY: float = -0.012       # 🚨 Reduced from -0.02 (40% reduction)
    
    # Probabilistic environment parameters
    PROBABILISTIC_REWARD_SCALE: float = 1.0  # No extra scaling needed
    
    # Distance-based coverage sensor model (Equation 4 from paper)
    # P_cov(cell | robot) = 1 / (1 + e^(k*(r - r0)))
    # where r is euclidean distance, r0 is midpoint, k is steepness
    #
    # TUNED PARAMETERS (adjusted for SENSOR_RANGE=4.0):
    # For 20×20 grid with SENSOR_RANGE=4.0 (reduced from 5.0):
    #   Steeper falloff for more challenging coverage task
    #   r0 = 2.0 (midpoint closer in)
    #   k = 1.8 (steeper for harder task)
    #
    # Coverage profile (with COVERAGE_THRESHOLD=0.85):
    #   r=0.0: P_cov=0.943 (robot position - 1 step to cover)
    #   r=1.0: P_cov=0.762 (2 steps to cover)
    #   r=2.0: P_cov=0.500 (2 steps to cover)
    #   r=3.0: P_cov=0.165 (6 steps to cover - very challenging!)
    #   r=4.0: P_cov=0.055 (16 steps - nearly impossible!)
    #
    # FIXED: Recalibrated for SENSOR_RANGE = 8.5 (for 40×40 grid)
    # r_eff = 0.75 * 8.5 = 6.375, k = 5.888 / 6.375 = 0.923, r0 = 6.375 / 2 = 3.19
    PROBABILISTIC_COVERAGE_STEEPNESS: float = 0.92  # FIXED: Scaled for larger sensor range (was 1.8)
    PROBABILISTIC_COVERAGE_MIDPOINT: float = 3.2    # FIXED: Scaled for larger sensor range (was 2.0)

    # ==================== Multi-Agent Reward Normalization ====================
    # CRITICAL: Normalize rewards for QMIX to prevent gradient explosion
    # Literature: QMIX paper (Rashid et al., 2018), R2D2 (Pohlen et al., 2018)
    
    # Per-agent normalization: Divide by number of agents
    # Single-agent: 5,250/episode → Multi-agent (4): 21,000/episode
    # After normalization: 5,250/episode (same scale as single-agent)
    MULTI_AGENT_REWARD_NORMALIZE_BY_N: bool = True
    
    # Scale factor: Map rewards to manageable range
    # REDUCED from 10.0 to 1.0 to avoid over-scaling with probabilistic mode
    # With probabilistic (0.15x) + normalize_by_n (÷4) + scale (÷1.0) = 0.0375x
    MULTI_AGENT_REWARD_SCALE_FACTOR: float = 1.0  # Changed from 10.0
    
    # Optional clipping (disabled by default - scaling is sufficient)
    MULTI_AGENT_REWARD_CLIP_MIN: float = None  # Set to -1.0 for hard clipping
    MULTI_AGENT_REWARD_CLIP_MAX: float = None  # Set to +1.0 for hard clipping
    
    # Value rescaling (R2D2 style - for advanced use)
    # h(x) = sign(x)(√(|x|+1) - 1) + εx
    MULTI_AGENT_USE_VALUE_RESCALING: bool = False
    MULTI_AGENT_VALUE_RESCALE_EPS: float = 0.001

    # ==================== Gradient Stability (EMERGENCY TIGHTENING) ====================
    # 🚨 CRITICAL: Current gradient norm = 54.1 (exploding!)
    # Need aggressive clipping to prevent catastrophic failure in ~100 episodes
    GRAD_CLIP_THRESHOLD: float = 0.2   # 🚨 EMERGENCY: Reduced from 0.3 - Ultra-tight clipping
    AGC_CLIP_RATIO: float = 0.01       # Keep strong AGC (working well)
    AGC_EPS: float = 1e-3
    EXPLOSION_THRESHOLD: float = 25.0  # 🚨 EMERGENCY: Reduced from 50.0 - Detect immediately
    MAX_GRAD_NORM: float = 20.0        # 🚨 EMERGENCY: Reduced from 30.0 - Hard limit

    # ==================== Training ====================
    STAGE1_EPISODES: int = 1500  # 🔄 1200 curriculum + 300 consolidation
    VALIDATION_INTERVAL: int = 100  # INCREASED from 50 to reduce overhead (validate less frequently)
    VALIDATION_EPISODES: int = 8    # REDUCED from 10 for faster validation
    CHECKPOINT_INTERVAL: int = 100   # 🔄 REDUCED from 200 - More frequent checkpoints
    
    # ==================== Performance Optimizations ====================
    # Reduce per-episode overhead for faster training
    ENABLE_TIMING_BREAKDOWN: bool = False  # DISABLED - timing adds overhead
    GRADIENT_ACCUMULATION_STEPS: int = 1   # Future: accumulate gradients for larger effective batch
    
    # Optimize training frequency (already optimized - train every step)
    # TRAIN_FREQ: int = 1 (defined above in Learning section)
    
    # Reduce validation overhead
    FAST_VALIDATION: bool = False  # Use fewer steps for validation episodes (disabled for accurate results)
    VALIDATION_MAX_STEPS: int = 150  # REDUCED from 200 - Limit validation episode length (vs 350 for training)
    
    # GPU optimizations
    USE_AMP: bool = False  # Automatic Mixed Precision (float16) - can cause instability in RL
    PIN_MEMORY: bool = True  # Pin memory for faster GPU transfers
    NUM_WORKERS: int = 0  # DataLoader workers (0 = main thread only for RL)
    PERSISTENT_WORKERS: bool = False  # Keep workers alive between batches
    
    # Compilation optimizations (PyTorch 2.0+)
    COMPILE_MODEL: bool = False  # torch.compile() - can speed up but adds warmup time
    COMPILE_MODE: str = "default"  # "default", "reduce-overhead", "max-autotune"

    # ==================== Paths ====================
    CHECKPOINT_DIR: str = "./checkpoints"
    RESULTS_DIR: str = "./results"

    # ==================== Device ====================
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ==================== Logging ====================
    VERBOSE: bool = True
    LOG_INTERVAL: int = 50  # INCREASED from 10 - Log every 50 episodes (reduces I/O overhead)
    
    # ==================== Debugging ====================
    LOG_INVALID_ACTIONS: bool = True  # NEW: Log when argmax proposes invalid action
    LOG_STAY_RATE: bool = True        # NEW: Log % of STAY actions
    LOG_SPATIAL_STATS: bool = True    # NEW: Log spatial coverage statistics


# Global config instance
config = Config()


def print_config():
    """Print configuration summary with critical changes highlighted."""
    print("=" * 80)
    print("FCN-BASED COVERAGE SYSTEM - CONFIGURATION SUMMARY")
    print("=" * 80)
    print(f"Device: {config.DEVICE}")
    print(f"Grid Size: {config.GRID_SIZE}")
    print(f"Sensor Range (POMDP): {config.SENSOR_RANGE}")
    print(f"Action Space: {config.N_ACTIONS} actions")
    print(f"\n🔧 OPTIMIZED HYPERPARAMETERS:")
    print(f"  ✅ Epsilon Decay: 0.98 Phase1 (faster exploitation)")
    print(f"  ✅ Training Frequency: Every {config.TRAIN_FREQ} step(s)")
    print(f"  ✅ Min Replay Size: {config.MIN_REPLAY_SIZE}")
    print(f"  ✅ Batch Size: {config.BATCH_SIZE}")
    print(f"  ✅ Learning Rate: {config.LEARNING_RATE}")
    print(f"  ✅ Coverage Reward: {config.COVERAGE_REWARD}")
    print(f"\nFCN Architecture:")
    print(f"  Hidden Dim: {config.CNN_HIDDEN_DIM}")
    print(f"  Dropout: {config.CNN_DROPOUT}")
    print(f"  CoordConv: {'Enabled' if config.USE_COORDCONV else 'Disabled'}")
    print(f"  Spatial Softmax Temp: {config.SPATIAL_SOFTMAX_TEMP}")
    print(f"\nExpected Results @ 800 Episodes (Empty Grid):")
    print(f"  Validation Coverage: 68-73%")
    print(f"  Training Time: ~6-8 hours (22s/episode)")
    print(f"  Parameters: ~2.5M")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
