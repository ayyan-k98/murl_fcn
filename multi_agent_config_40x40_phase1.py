"""
Multi-Agent Configuration - Phase 1: Refined Independent Agents

40×40 grid with 6-channel input (agent occupancy) and improved full_state communication.
Focus on independent agents with proper coordination signals.

Based on analysis of failed runs - fixes:
1. Correct sigmoid parameters for sensor_range=8.5
2. Add 6th channel (agent occupancy from communication)
3. Better sensor/comm range scaling
4. Longer Phase 1 training (empty maps for coordination learning)
"""

from multi_agent_env import CoordinationStrategy


class MultiAgentConfig40x40:
    """
    Refined independent multi-agent configuration for 40×40 grids.
    
    Key improvements over previous attempts:
    - Optimal sensor range (8.5, not 7.0)
    - Correct sigmoid parameters (k=0.92, r0=3.2)
    - 6-channel input with agent occupancy
    - Improved communication range (15.0 for 3σ coverage)
    - Extended Phase 1 for coordination learning
    """

    # ============================================================================
    # Environment Settings - OPTIMIZED FOR 40×40
    # ============================================================================

    # Grid dimensions
    GRID_SIZE = 40  # Intermediate scale (between 20 and 50)
    NUM_AGENTS = 4
    MAX_STEPS = 350  # Same time pressure as single-agent

    # Sensor parameters - OPTIMIZED SCALING
    SENSOR_RANGE = 8.5  # Optimal: 5.0 × (40/20)^0.4 = 8.7 ≈ 8.5
    NUM_RAYS = 45       # Scaled from 32: 32 × (8.5/5.0) = 54 → 45 balanced
    SAMPLES_PER_RAY = 17  # Scaled from 10: 10 × 1.7 = 17

    # Communication parameters
    COMMUNICATION_RANGE = 15.0  # Covers 3σ position uncertainty (was 13.0)
    COMMUNICATION_FREQUENCY = 5  # Every 5 steps
    COMMUNICATION_TYPE = "full_state"  # For now, will refine later

    # Coordination
    COORDINATION = CoordinationStrategy.INDEPENDENT  # Phase 1: independent agents
    PARAMETER_SHARING = False  # Independent networks (better)
    SHARED_REPLAY = True  # Shared experience

    # ============================================================================
    # Probabilistic Coverage - CORRECTED SIGMOID PARAMETERS
    # ============================================================================

    USE_PROBABILISTIC_ENV = True

    # CRITICAL FIX: Sigmoid parameters MUST match sensor_range!
    # 
    # Theory: Effective coverage radius r_eff = 0.75 × sensor_range
    #         For sensor_range = 8.5: r_eff = 6.375 cells
    #
    # Sigmoid parameters:
    #         k = 5.888 / r_eff = 5.888 / 6.375 = 0.923
    #         r0 = r_eff / 2 = 6.375 / 2 = 3.19
    #
    # Previous (WRONG): k=1.8, r0=3.0 (for sensor_range=5.0)
    # Corrected: k=0.92, r0=3.2 (for sensor_range=8.5)
    
    PROBABILISTIC_COVERAGE_STEEPNESS = 0.92  # FIXED from 1.8!
    PROBABILISTIC_COVERAGE_MIDPOINT = 3.2    # FIXED from 3.0!
    COVERAGE_THRESHOLD = 0.85  # Cell considered covered at 85%

    # ============================================================================
    # Input Channels - ADD 6TH CHANNEL!
    # ============================================================================

    INPUT_CHANNELS = 6  # CRITICAL: Was 5, now 6!
    
    # Channel breakdown:
    # 0: Visited cells (binary)
    # 1: Coverage probability (0-1, from sigmoid)
    # 2: Agent position (one-hot)
    # 3: Frontier cells (binary)
    # 4: Obstacles (binary)
    # 5: Other agents occupancy (NEW! probabilistic from communication)
    
    USE_AGENT_OCCUPANCY_CHANNEL = True  # Enable 6th channel

    # ============================================================================
    # Agent Occupancy Channel Parameters
    # ============================================================================

    # Position uncertainty model: σ = σ_base + σ_growth × Δt
    # where Δt is steps since last communication
    POSITION_UNCERTAINTY_BASE = 0.5    # Base uncertainty (cells)
    POSITION_UNCERTAINTY_GROWTH = 1.0  # Growth per step

    # At comm_freq=5: max uncertainty = 0.5 + 1.0×5 = 5.5 cells
    # 3σ coverage = 16.5 cells → comm_range=15.0 covers ~2.7σ (99.3%)

    # Occupancy computation
    OCCUPANCY_KERNEL_SIZE = 5  # Gaussian kernel for spreading uncertainty
    OCCUPANCY_NORMALIZATION = True  # Normalize to [0, 1]

    # ============================================================================
    # Reward Structure
    # ============================================================================

    # Individual rewards
    COVERAGE_REWARD = 2.0  # Per newly covered cell (NOT 10.0!)
    EXPLORATION_REWARD = 0.4  # Per newly sensed cell
    
    # Penalties
    COLLISION_PENALTY = -0.25  # Per collision (FIXED from -5.0!)
    STEP_PENALTY = -0.0012
    STAY_PENALTY = -0.012
    
    # Rotation penalties (smooth trajectories)
    ROTATION_PENALTY_SMALL = -0.05   # 45° turn
    ROTATION_PENALTY_MEDIUM = -0.10  # 90° turn
    ROTATION_PENALTY_LARGE = -0.15   # 135°-180° turn
    USE_ROTATION_PENALTY = True

    # Team rewards (Phase 1: minimal, will increase later)
    TEAM_REWARD_WEIGHT = 0.3
    USE_OVERLAP_PENALTY = False  # Disabled for Phase 1
    OVERLAP_PENALTY_SCALE = 0.0  # Will enable in Phase 2
    
    USE_DIVERSITY_BONUS = False  # Disabled for Phase 1
    DIVERSITY_BONUS_SCALE = 0.0
    
    USE_EFFICIENCY_BONUS = False  # Disabled for Phase 1
    EFFICIENCY_BONUS_SCALE = 0.0

    # Reward normalization (CRITICAL for multi-agent!)
    NORMALIZE_BY_N_AGENTS = True  # Divide by 4 to match single-agent scale
    REWARD_SCALE_FACTOR = 1.0  # Additional scaling if needed

    # ============================================================================
    # Learning Parameters
    # ============================================================================

    LEARNING_RATE = 5e-5  # Conservative (was probably too high before)
    LEARNING_RATE_MIN = 1e-5
    LR_DECAY_RATE = 0.9998
    
    GAMMA = 0.99
    BATCH_SIZE = 256
    REPLAY_BUFFER_SIZE = 50000
    TARGET_UPDATE_FREQ = 50
    MIN_REPLAY_SIZE = 200
    TRAIN_FREQ = 4
    GRAD_CLIP_NORM = 3.0  # Aggressive clipping

    # N-step returns
    N_STEP = 3
    N_STEP_ENABLED = True

    # ============================================================================
    # Exploration Schedule - PHASE SPECIFIC
    # ============================================================================

    EPSILON_START = 1.0
    EPSILON_MIN = 0.05

    # Phase-specific decay rates
    EPSILON_DECAY_PHASE1 = 0.98   # Fast decay in Phase 1 (coordination learning)
    EPSILON_DECAY_PHASE2 = 0.985
    EPSILON_DECAY_PHASE3 = 0.987
    EPSILON_DECAY_PHASE4 = 0.990

    # ============================================================================
    # Curriculum - EXTENDED PHASE 1
    # ============================================================================

    TOTAL_EPISODES = 1000
    
    # Phase 1: EXTENDED empty map training (300 episodes)
    # Goal: Learn coordination without obstacle complexity
    PHASE1_EPISODES = 300  # Was 200, now 300!
    PHASE1_MAP_MIX = {"empty": 1.0}
    PHASE1_TARGET_COVERAGE = 0.82
    PHASE1_TARGET_OVERLAP = 0.20  # <20% overlap by end of phase
    
    # Phase 2: Sparse obstacles (200 episodes)
    PHASE2_EPISODES = 200
    PHASE2_MAP_MIX = {"empty": 0.6, "random": 0.4}
    PHASE2_TARGET_COVERAGE = 0.80
    
    # Phase 3: Mixed environments (200 episodes)
    PHASE3_EPISODES = 200
    PHASE3_MAP_MIX = {"empty": 0.3, "random": 0.4, "room": 0.3}
    PHASE3_TARGET_COVERAGE = 0.78
    
    # Phase 4: Complex + corridors (100 episodes)
    PHASE4_EPISODES = 100
    PHASE4_MAP_MIX = {
        "empty": 0.2,
        "random": 0.3,
        "room": 0.3,
        "corridor": 0.2  # Coordination bottleneck test!
    }
    PHASE4_TARGET_COVERAGE = 0.75
    PHASE4_TARGET_CORRIDOR = 0.60  # Acceptable corridor performance

    # ============================================================================
    # Validation
    # ============================================================================

    VALIDATION_FREQ = 50  # Every 50 episodes
    VALIDATION_EPISODES = 20  # 20 validation episodes
    VALIDATION_MAP_TYPES = ['empty', 'random', 'room', 'corridor', 'cave']
    
    # Expected validation performance targets
    EXPECTED_VAL_COVERAGE_EP300 = 0.78  # End of Phase 1
    EXPECTED_VAL_COVERAGE_EP1000 = 0.72  # Final (with corridors)
    EXPECTED_VAL_GAP = 0.05  # <5% train-val gap (not 31%!)

    # ============================================================================
    # Logging and Checkpointing
    # ============================================================================

    LOG_FREQ = 10
    SAVE_FREQ = 100
    PLOT_FREQ = 50
    
    CHECKPOINT_DIR = "multi_agent_results/checkpoints"
    METRICS_DIR = "multi_agent_results/metrics"
    VIS_DIR = "multi_agent_results/visualizations"
    
    SAVE_VISUALIZATIONS = True
    TRACK_PER_AGENT_METRICS = True
    TRACK_COORDINATION_METRICS = True

    # ============================================================================
    # Expected Performance Milestones
    # ============================================================================

    MILESTONES = {
        'ep_100': {
            'coverage': 0.75,
            'overlap': 0.27,
            'efficiency': 0.60,
            'coord_score': 65
        },
        'ep_200': {
            'coverage': 0.80,
            'overlap': 0.22,
            'efficiency': 0.68,
            'coord_score': 72
        },
        'ep_300': {
            'coverage': 0.82,
            'overlap': 0.18,
            'efficiency': 0.72,
            'coord_score': 78
        },
        'ep_500': {
            'coverage': 0.80,  # Lower due to obstacles
            'overlap': 0.16,
            'efficiency': 0.75,
            'coord_score': 80
        },
        'ep_800': {
            'coverage': 0.78,  # Lower due to corridors
            'overlap': 0.15,
            'efficiency': 0.77,
            'coord_score': 82
        },
        'ep_1000': {
            'coverage': 0.76,
            'overlap': 0.14,
            'efficiency': 0.78,
            'coord_score': 84
        }
    }

    # ============================================================================
    # Helper Methods
    # ============================================================================

    @classmethod
    def get_phase(cls, episode: int) -> dict:
        """Get current curriculum phase."""
        if episode < cls.PHASE1_EPISODES:
            return {
                'phase': 1,
                'name': 'Extended Empty Maps (Coordination Learning)',
                'episodes': (0, cls.PHASE1_EPISODES),
                'map_mix': cls.PHASE1_MAP_MIX,
                'target_coverage': cls.PHASE1_TARGET_COVERAGE,
                'epsilon_decay': cls.EPSILON_DECAY_PHASE1
            }
        elif episode < cls.PHASE1_EPISODES + cls.PHASE2_EPISODES:
            return {
                'phase': 2,
                'name': 'Sparse Obstacles',
                'episodes': (cls.PHASE1_EPISODES, cls.PHASE1_EPISODES + cls.PHASE2_EPISODES),
                'map_mix': cls.PHASE2_MAP_MIX,
                'target_coverage': cls.PHASE2_TARGET_COVERAGE,
                'epsilon_decay': cls.EPSILON_DECAY_PHASE2
            }
        elif episode < cls.PHASE1_EPISODES + cls.PHASE2_EPISODES + cls.PHASE3_EPISODES:
            return {
                'phase': 3,
                'name': 'Mixed Environments',
                'episodes': (cls.PHASE1_EPISODES + cls.PHASE2_EPISODES,
                           cls.PHASE1_EPISODES + cls.PHASE2_EPISODES + cls.PHASE3_EPISODES),
                'map_mix': cls.PHASE3_MAP_MIX,
                'target_coverage': cls.PHASE3_TARGET_COVERAGE,
                'epsilon_decay': cls.EPSILON_DECAY_PHASE3
            }
        else:
            return {
                'phase': 4,
                'name': 'Complex + Corridors',
                'episodes': (cls.PHASE1_EPISODES + cls.PHASE2_EPISODES + cls.PHASE3_EPISODES, cls.TOTAL_EPISODES),
                'map_mix': cls.PHASE4_MAP_MIX,
                'target_coverage': cls.PHASE4_TARGET_COVERAGE,
                'epsilon_decay': cls.EPSILON_DECAY_PHASE4
            }

    @classmethod
    def get_map_type(cls, episode: int) -> str:
        """Sample map type for episode."""
        import numpy as np
        phase = cls.get_phase(episode)
        map_mix = phase['map_mix']
        
        map_types = list(map_mix.keys())
        probs = list(map_mix.values())
        
        return np.random.choice(map_types, p=probs)

    @classmethod
    def print_config(cls):
        """Print configuration summary."""
        print("=" * 70)
        print("MULTI-AGENT CONFIG: 40×40 REFINED INDEPENDENT + 6CH")
        print("=" * 70)
        print(f"  Grid Size: {cls.GRID_SIZE}×{cls.GRID_SIZE} ({cls.GRID_SIZE**2} cells)")
        print(f"  Agents: {cls.NUM_AGENTS}")
        print(f"  Sensor Range: {cls.SENSOR_RANGE} cells")
        print(f"  Comm Range: {cls.COMMUNICATION_RANGE} cells")
        print(f"  Input Channels: {cls.INPUT_CHANNELS} (WITH agent occupancy!)")
        print(f"")
        print(f"  Sigmoid: k={cls.PROBABILISTIC_COVERAGE_STEEPNESS}, r0={cls.PROBABILISTIC_COVERAGE_MIDPOINT}")
        print(f"  Coverage Threshold: {cls.COVERAGE_THRESHOLD}")
        print(f"")
        print(f"  Coordination: {cls.COORDINATION.value}")
        print(f"  Parameter Sharing: {cls.PARAMETER_SHARING}")
        print(f"  Communication: {cls.COMMUNICATION_TYPE}")
        print(f"")
        print(f"  Total Episodes: {cls.TOTAL_EPISODES}")
        print(f"  Phase 1 (Empty): {cls.PHASE1_EPISODES} episodes")
        print(f"  Phase 2 (Sparse): {cls.PHASE2_EPISODES} episodes")
        print(f"  Phase 3 (Mixed): {cls.PHASE3_EPISODES} episodes")
        print(f"  Phase 4 (Complex): {cls.PHASE4_EPISODES} episodes")
        print("=" * 70)


# Global instance
ma_config_40x40 = MultiAgentConfig40x40()


if __name__ == "__main__":
    # Test configuration
    ma_config_40x40.print_config()
    
    print("\n✓ Phase Breakdown:")
    for ep in [0, 150, 350, 600, 900]:
        phase = ma_config_40x40.get_phase(ep)
        print(f"  Episode {ep}: Phase {phase['phase']} - {phase['name']}")
        print(f"    Map mix: {phase['map_mix']}")
        print(f"    Target: {phase['target_coverage']*100:.0f}% coverage")
    
    print("\n✓ Performance Milestones:")
    for key, milestone in ma_config_40x40.MILESTONES.items():
        print(f"  {key}: Cov={milestone['coverage']*100:.0f}%, "
              f"Overlap={milestone['overlap']*100:.0f}%, "
              f"Eff={milestone['efficiency']*100:.0f}%, "
              f"Coord={milestone['coord_score']}")
    
    print("\n✓ Config test complete")
