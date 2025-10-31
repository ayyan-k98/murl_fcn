"""
Multi-Agent Configuration

Hyperparameters for multi-agent coverage training.

Extends base config.py with multi-agent specific settings.
"""

from multi_agent_env import CoordinationStrategy


class MultiAgentConfig:
    """Multi-agent training configuration."""

    # ============================================================================
    # Multi-Agent Environment Settings
    # ============================================================================

    # Number of agents (2-8 supported)
    NUM_AGENTS = 4

    # Grid size (can be different from single-agent training)
    # FIXED: Changed from 20 to 40 (matches actual training observed in logs)
    GRID_SIZE = 40

    # Sensor range (POMDP)
    # FIXED: Scaled from 5.0 for 20×20 to 8.5 for 40×40
    # Formula: 5.0 × (40/20)^0.4 = 5.0 × 1.74 = 8.7 ≈ 8.5
    SENSOR_RANGE = 8.5

    # Communication settings
    # CRITICAL: Enable communication for proper coordination!
    USE_COMMUNICATION = True  # Enable position-based communication
    COMM_PROTOCOL = 'position'  # Options: 'none', 'position'

    # Communication range between agents
    # FIXED: Increased to cover 3σ position uncertainty (99.7% confidence)
    # σ(t=5) = 0.5 + 1.0*5 = 5.5, 3σ = 16.5, using 15.0 for practical range
    COMMUNICATION_RANGE = 15.0

    # Communication frequency (every N steps)
    # 1 = every step (high overhead but perfect info)
    # 5 = every 5 steps (balanced, recommended)
    # 10 = every 10 steps (low overhead but stale info)
    COMMUNICATION_FREQUENCY = 5

    # Coordination strategy
    # Options: INDEPENDENT, HIERARCHICAL
    COORDINATION = CoordinationStrategy.INDEPENDENT

    # Team reward weight [0, 1]
    # 0.0 = purely individual rewards
    # 1.0 = purely team rewards
    # 0.3 = recommended (emphasize individual + coordination)
    TEAM_REWARD_WEIGHT = 0.3

    # Agent-agent collision penalty
    AGENT_COLLISION_PENALTY = -5.0
    
    # Team reward components
    USE_OVERLAP_PENALTY = True
    # FIXED: Reduced from 2.0 to 0.01 (200× reduction)
    # Even with fixed counting, 2.0 per cell per step was too high
    # 500 overlapping cells × 2.0 × 350 steps = -350,000 (catastrophic!)
    # 500 overlapping cells × 0.01 × 350 steps = -1,750 (reasonable)
    OVERLAP_PENALTY_SCALE = 0.01  # Penalty per overlapping cell per step
    
    USE_DIVERSITY_BONUS = True
    DIVERSITY_BONUS_SCALE = 0.5  # Reward for maintaining distance
    
    USE_EFFICIENCY_BONUS = True
    EFFICIENCY_BONUS_SCALE = 5.0  # Reward coverage/visits ratio

    # ============================================================================
    # Training Settings
    # ============================================================================

    # Parameter sharing
    # True: All agents share the same network (faster, less memory)
    # False: Each agent has independent network (more flexible)
    # FIXED: Changed to False - independent networks needed for POMDP with different viewpoints
    # Agents see different local observations and need specialization for coordination
    PARAMETER_SHARING = False

    # Shared replay memory
    # True: Single replay buffer for all agents
    # False: Separate buffers per agent
    SHARED_REPLAY = True

    # Training episodes
    # FIXED: Reduced from 1000 to 800 (matches new curriculum final phase)
    TOTAL_EPISODES = 800

    # Validation frequency
    VALIDATION_FREQ = 50

    # Validation episodes per checkpoint
    VALIDATION_EPISODES = 20

    # Save frequency
    SAVE_FREQ = 100

    # ============================================================================
    # Curriculum Learning (Multi-Agent)
    # ============================================================================

    # Multi-agent curriculum phases
    # FIXED: Start with 4 agents from beginning, add corridors progressively, reduce total episodes to 800
    # Focus: Target team size (4 agents) from start, vary map complexity not team size

    CURRICULUM_PHASES = [
        {
            'name': 'Phase 1: Formation (4 agents, empty maps, independent)',
            'start_ep': 0,
            'end_ep': 200,
            'num_agents': 4,  # Start with target team size
            'map_distribution': {'empty': 0.8, 'random': 0.2},
            'coordination': CoordinationStrategy.INDEPENDENT,
            'expected_coverage': 0.75,
            'epsilon_floor': 0.1,
            'epsilon_decay': 0.98,
            'description': 'Learn basic multi-agent coverage on simple maps'
        },
        {
            'name': 'Phase 2: Obstacles (4 agents, sparse obstacles, hierarchical)',
            'start_ep': 200,
            'end_ep': 400,
            'num_agents': 4,
            'map_distribution': {'empty': 0.5, 'random': 0.4, 'corridor': 0.1},  # Introduce corridors early!
            'coordination': CoordinationStrategy.HIERARCHICAL,  # Start coordination learning
            'expected_coverage': 0.78,
            'epsilon_floor': 0.08,
            'epsilon_decay': 0.98,
            'description': 'Learn coordination with sparse obstacles, introduce corridors'
        },
        {
            'name': 'Phase 3: Complex (4 agents, mixed maps, hierarchical)',
            'start_ep': 400,
            'end_ep': 600,
            'num_agents': 4,
            'map_distribution': {'empty': 0.3, 'random': 0.3, 'corridor': 0.2, 'cave': 0.2},  # More corridors + caves
            'coordination': CoordinationStrategy.HIERARCHICAL,
            'expected_coverage': 0.82,
            'epsilon_floor': 0.05,
            'epsilon_decay': 0.98,
            'description': 'Advanced coordination with complex maps and bottlenecks'
        },
        {
            'name': 'Phase 4: Final Challenge (4 agents, all maps, hierarchical)',
            'start_ep': 600,
            'end_ep': 800,  # Reduced from 1000
            'num_agents': 4,
            'map_distribution': {
                'empty': 0.2,
                'random': 0.3,
                'corridor': 0.3,  # Heavy corridor emphasis
                'cave': 0.2       # Use 'cave' instead of 'maze' (maze not implemented)
            },
            'coordination': CoordinationStrategy.HIERARCHICAL,
            'expected_coverage': 0.85,
            'epsilon_floor': 0.05,
            'epsilon_decay': 0.98,
            'description': 'Final training with corridor-heavy distribution to match validation'
        }
    ]

    # ============================================================================
    # Multi-Agent Validation Settings
    # ============================================================================

    # Test team sizes (for generalization testing)
    VALIDATION_TEAM_SIZES = [2, 3, 4, 5]

    # Test coordination strategies
    VALIDATION_STRATEGIES = [
        CoordinationStrategy.INDEPENDENT,
        CoordinationStrategy.HIERARCHICAL
    ]

    # Test map types
    VALIDATION_MAP_TYPES = ['empty', 'random', 'room', 'corridor', 'cave']

    # ============================================================================
    # Logging and Visualization
    # ============================================================================

    # Log frequency (episodes)
    LOG_FREQ = 10

    # Plot frequency (episodes)
    PLOT_FREQ = 50

    # Save visualizations
    SAVE_VISUALIZATIONS = True

    # Visualization directory
    VIS_DIR = "multi_agent_results/visualizations"

    # Checkpoint directory
    CHECKPOINT_DIR = "multi_agent_results/checkpoints"

    # Metrics directory
    METRICS_DIR = "multi_agent_results/metrics"

    # ============================================================================
    # Experiment Tracking
    # ============================================================================

    # Experiment name (auto-generated if None)
    EXPERIMENT_NAME = None

    # Track per-agent metrics
    TRACK_PER_AGENT_METRICS = True

    # Track coordination metrics (region overlap, task conflicts, etc.)
    TRACK_COORDINATION_METRICS = True

    # Save episode videos (warning: slow and large)
    SAVE_EPISODE_VIDEOS = False

    # ============================================================================
    # Performance Optimizations
    # ============================================================================

    # Batch action selection (use vectorized forward pass)
    USE_BATCH_ACTION_SELECTION = True

    # Parallel environment rollouts (if supported)
    USE_PARALLEL_ENVS = False
    NUM_PARALLEL_ENVS = 4

    # Mixed precision training
    USE_MIXED_PRECISION = False

    # ============================================================================
    # Helper Methods
    # ============================================================================

    @classmethod
    def get_phase(cls, episode: int) -> dict:
        """Get curriculum phase for given episode."""
        for phase in cls.CURRICULUM_PHASES:
            if phase['start_ep'] <= episode < phase['end_ep']:
                return phase

        # Return last phase if beyond curriculum
        return cls.CURRICULUM_PHASES[-1]

    @classmethod
    def get_map_type(cls, episode: int) -> str:
        """Sample map type for given episode."""
        import numpy as np

        phase = cls.get_phase(episode)
        map_dist = phase['map_distribution']

        map_types = list(map_dist.keys())
        probs = list(map_dist.values())

        return np.random.choice(map_types, p=probs)

    @classmethod
    def get_epsilon(cls, episode: int, base_epsilon: float) -> float:
        """Get epsilon for given episode."""
        phase = cls.get_phase(episode)
        epsilon_floor = phase['epsilon_floor']
        epsilon_decay = phase['epsilon_decay']

        # Decay epsilon within phase
        phase_progress = episode - phase['start_ep']
        decayed_epsilon = base_epsilon * (epsilon_decay ** phase_progress)

        return max(epsilon_floor, decayed_epsilon)

    @classmethod
    def print_config(cls):
        """Print multi-agent configuration summary."""
        print("=" * 70)
        print("MULTI-AGENT CONFIGURATION")
        print("=" * 70)
        print(f"  Num Agents: {cls.NUM_AGENTS}")
        print(f"  Grid Size: {cls.GRID_SIZE}")
        print(f"  Coordination: {cls.COORDINATION.value}")
        print(f"  Parameter Sharing: {cls.PARAMETER_SHARING}")
        print(f"  Shared Replay: {cls.SHARED_REPLAY}")
        print(f"  Team Reward Weight: {cls.TEAM_REWARD_WEIGHT}")
        print(f"  Total Episodes: {cls.TOTAL_EPISODES}")
        print(f"  Curriculum Phases: {len(cls.CURRICULUM_PHASES)}")
        print("=" * 70)


# Create global instance
ma_config = MultiAgentConfig()


if __name__ == "__main__":
    # Test configuration
    ma_config.print_config()

    print("\n✓ Curriculum Phases:")
    for phase in ma_config.CURRICULUM_PHASES:
        print(f"  {phase['name']}")
        print(f"    Episodes: {phase['start_ep']}-{phase['end_ep']}")
        print(f"    Agents: {phase['num_agents']}")
        print(f"    Coordination: {phase['coordination'].value}")
        print(f"    Expected Coverage: {phase['expected_coverage']*100:.0f}%")

    print("\n✓ Phase lookup test:")
    test_episodes = [0, 100, 400, 800]
    for ep in test_episodes:
        phase = ma_config.get_phase(ep)
        print(f"  Episode {ep}: {phase['name']}")

    print("\n✓ Map type sampling test:")
    for ep in [0, 300, 700]:
        map_type = ma_config.get_map_type(ep)
        print(f"  Episode {ep}: {map_type}")

    print("\n✓ Multi-agent config test complete")
