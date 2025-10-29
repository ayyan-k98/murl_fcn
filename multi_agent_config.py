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
    GRID_SIZE = 20

    # Sensor range (POMDP)
    SENSOR_RANGE = 3.0

    # Communication range between agents
    COMMUNICATION_RANGE = 5.0
    
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
    OVERLAP_PENALTY_SCALE = 2.0  # Penalty per overlapping cell
    
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
    PARAMETER_SHARING = True

    # Shared replay memory
    # True: Single replay buffer for all agents
    # False: Separate buffers per agent
    SHARED_REPLAY = True

    # Training episodes
    TOTAL_EPISODES = 1000

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
    # Progressive training: start simple, increase complexity

    CURRICULUM_PHASES = [
        {
            'name': 'Phase 1: Team Formation (2 agents, empty)',
            'start_ep': 0,
            'end_ep': 150,
            'num_agents': 2,
            'map_distribution': {'empty': 1.0},
            'coordination': CoordinationStrategy.INDEPENDENT,
            'expected_coverage': 0.75,
            'epsilon_floor': 0.1,
            'epsilon_decay': 0.98
        },
        {
            'name': 'Phase 2: Simple Coordination (2 agents, mixed)',
            'start_ep': 150,
            'end_ep': 300,
            'num_agents': 2,
            'map_distribution': {'empty': 0.6, 'random': 0.4},
            'coordination': CoordinationStrategy.INDEPENDENT,
            'expected_coverage': 0.80,
            'epsilon_floor': 0.1,
            'epsilon_decay': 0.98
        },
        {
            'name': 'Phase 3: Scale to 4 agents (empty)',
            'start_ep': 300,
            'end_ep': 450,
            'num_agents': 4,
            'map_distribution': {'empty': 0.7, 'random': 0.3},
            'coordination': CoordinationStrategy.INDEPENDENT,
            'expected_coverage': 0.82,
            'epsilon_floor': 0.08,
            'epsilon_decay': 0.98
        },
        {
            'name': 'Phase 4: Hierarchical Coordination (4 agents)',
            'start_ep': 450,
            'end_ep': 600,
            'num_agents': 4,
            'map_distribution': {'empty': 0.5, 'random': 0.3, 'maze': 0.2},
            'coordination': CoordinationStrategy.HIERARCHICAL,
            'expected_coverage': 0.85,
            'epsilon_floor': 0.08,
            'epsilon_decay': 0.98
        },
        {
            'name': 'Phase 5: Advanced Coordination (4 agents)',
            'start_ep': 600,
            'end_ep': 750,
            'num_agents': 4,
            'map_distribution': {'empty': 0.4, 'random': 0.3, 'maze': 0.3},
            'coordination': CoordinationStrategy.HIERARCHICAL,
            'expected_coverage': 0.87,
            'epsilon_floor': 0.05,
            'epsilon_decay': 0.98
        },
        {
            'name': 'Phase 6: Final Challenge (4 agents, all maps)',
            'start_ep': 750,
            'end_ep': 1000,
            'num_agents': 4,
            'map_distribution': {
                'empty': 0.2,
                'random': 0.25,
                'maze': 0.25,
                'office': 0.15,
                'warehouse': 0.15
            },
            'coordination': CoordinationStrategy.HIERARCHICAL,
            'expected_coverage': 0.90,
            'epsilon_floor': 0.05,
            'epsilon_decay': 0.98
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
    VALIDATION_MAP_TYPES = ['empty', 'random', 'maze', 'office', 'warehouse']

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
