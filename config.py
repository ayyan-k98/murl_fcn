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
    SENSOR_RANGE: float = 5.0  # POMDP: Limited observation radius
    COMM_RANGE: float = 10.0    # For Stage 2 multi-agent
    NUM_RAYS: int = 8          # REDUCED from 12 for speed (33% faster raycast)
    SAMPLES_PER_RAY: int = 6   # REDUCED from 8 for speed (25% fewer samples)
    MAX_EPISODE_STEPS: int = 350  # REDUCED from 350 for faster training (43% speedup)
    USE_PROBABILISTIC_ENV: bool = False  # Toggle between binary and probabilistic coverage

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

    # ==================== Learning (FCN-OPTIMIZED) ====================
    LEARNING_RATE: float = 3e-4       # FCN-optimized: Lower than GAT (more stable with CNN)
    LEARNING_RATE_MIN: float = 5e-5   # Minimum LR for decay
    LR_DECAY_RATE: float = 0.9995     # Slow decay (reach min LR at ~6000 episodes)
    GAMMA: float = 0.99               # Discount factor
    BATCH_SIZE: int = 256             # Large batch for stable gradients
    REPLAY_BUFFER_SIZE: int = 50000   # 50k transitions (~200 episodes worth)
    TARGET_UPDATE_FREQ: int = 100     # Update target network every 100 episodes
    MIN_REPLAY_SIZE: int = 200        # Start training after 200 transitions (~1 episode)
    TRAIN_FREQ: int = 4               # Train every 4 steps (balance speed vs sample efficiency)
    GRAD_CLIP_NORM: float = 10.0      # Gradient clipping (FCN more stable than GAT)
    
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

    # ==================== Rewards (RESTORED ORIGINAL SCALE) ====================
    # CRITICAL: Restore full rewards - 20x reduction was too aggressive
    # Agent needs strong signal to learn spatial navigation
    COVERAGE_REWARD: float = 15.0      # INCREASED from 10.0 - Stronger learning signal
    EXPLORATION_REWARD: float = 1.0    # INCREASED from 0.5
    FRONTIER_BONUS: float = 0.1        # INCREASED from 0.05
    FRONTIER_CAP: float = 2.0          # INCREASED from 1.5
    COLLISION_PENALTY: float = -2.0    # RESTORED from -0.1
    STEP_PENALTY: float = -0.01        # RESTORED from -0.0005
    STAY_PENALTY: float = -0.1         # RESTORED from -0.005
    
    # Probabilistic environment parameters
    PROBABILISTIC_REWARD_SCALE: float = 0.15  # Reward scaling for probabilistic mode
    
    # Distance-based coverage sensor model (Equation 4 from paper)
    # P_cov(cell | robot) = 1 / (1 + e^(k*(r - r0)))
    # where r is euclidean distance, r0 is midpoint, k is steepness
    PROBABILISTIC_COVERAGE_MIDPOINT: float = 1.5   # r0: distance where P_cov = 0.5
    PROBABILISTIC_COVERAGE_STEEPNESS: float = 2.0  # k: sigmoid steepness (higher = sharper falloff)

    # ==================== Multi-Agent Reward Normalization ====================
    # CRITICAL: Normalize rewards for QMIX to prevent gradient explosion
    # Literature: QMIX paper (Rashid et al., 2018), R2D2 (Pohlen et al., 2018)
    
    # Per-agent normalization: Divide by number of agents
    # Single-agent: 5,250/episode → Multi-agent (4): 21,000/episode
    # After normalization: 5,250/episode (same scale as single-agent)
    MULTI_AGENT_REWARD_NORMALIZE_BY_N: bool = True
    
    # Scale factor: Map rewards to manageable range
    # Typical per-step reward: 0-20 → After scaling: 0-2
    # This keeps Q-values in range [0, ~50] instead of [0, 60,000]
    MULTI_AGENT_REWARD_SCALE_FACTOR: float = 10.0
    
    # Optional clipping (disabled by default - scaling is sufficient)
    MULTI_AGENT_REWARD_CLIP_MIN: float = None  # Set to -1.0 for hard clipping
    MULTI_AGENT_REWARD_CLIP_MAX: float = None  # Set to +1.0 for hard clipping
    
    # Value rescaling (R2D2 style - for advanced use)
    # h(x) = sign(x)(√(|x|+1) - 1) + εx
    MULTI_AGENT_USE_VALUE_RESCALING: bool = False
    MULTI_AGENT_VALUE_RESCALE_EPS: float = 0.001

    # ==================== Gradient Stability ====================
    GRAD_CLIP_THRESHOLD: float = 1.0   # Keep tight (working well)
    AGC_CLIP_RATIO: float = 0.01       # Keep strong AGC (working well)
    AGC_EPS: float = 1e-3
    EXPLOSION_THRESHOLD: float = 500.0
    MAX_GRAD_NORM: float = 200.0

    # ==================== Training ====================
    STAGE1_EPISODES: int = 2000  # EXTENDED from 1600 for better generalization & 92% success confidence
    VALIDATION_INTERVAL: int = 100  # INCREASED from 50 to reduce overhead (validate less frequently)
    VALIDATION_EPISODES: int = 8    # REDUCED from 10 for faster validation
    CHECKPOINT_INTERVAL: int = 200   # INCREASED from 100 to reduce I/O overhead
    
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
