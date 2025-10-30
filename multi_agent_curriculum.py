"""
Multi-Agent Curriculum for Coordination Learning

This curriculum is FUNDAMENTALLY DIFFERENT from single-agent curriculum.
Key differences:
1. Assumes coverage skills transferred from single-agent checkpoint
2. Focus on COORDINATION not coverage
3. Rooms introduced LATER (doorway bottlenecks harder for multi-agent)
4. Shorter duration (800-1000 episodes vs 2250)
5. Multiple metrics: coverage + overlap + spacing
6. Slower epsilon decay (joint exploration space is larger)

Design Philosophy:
- Single-agent learns: "How to cover space efficiently"
- Multi-agent learns: "How to coordinate without interfering"
"""

import config
from typing import Dict, Tuple

class MultiAgentCurriculumPhase:
    """Configuration for one curriculum phase."""
    
    def __init__(
        self,
        episode_start: int,
        episode_end: int,
        map_type_distribution: Dict[str, float],
        coverage_target: float,
        overlap_target: float,  # NEW: target overlap percentage
        epsilon_start: float,
        epsilon_end: float,
        description: str
    ):
        self.episode_start = episode_start
        self.episode_end = episode_end
        self.map_type_distribution = map_type_distribution
        self.coverage_target = coverage_target
        self.overlap_target = overlap_target
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.description = description
    
    def get_epsilon(self, episode: int) -> float:
        """Calculate epsilon for current episode within this phase."""
        if episode < self.episode_start or episode >= self.episode_end:
            return self.epsilon_end
        
        phase_progress = (episode - self.episode_start) / (self.episode_end - self.episode_start)
        epsilon = self.epsilon_start + (self.epsilon_end - self.epsilon_start) * phase_progress
        return epsilon
    
    def sample_map_type(self) -> str:
        """Sample map type from distribution."""
        import random
        map_types = list(self.map_type_distribution.keys())
        probabilities = list(self.map_type_distribution.values())
        return random.choices(map_types, weights=probabilities, k=1)[0]


# ============================================================================
# MULTI-AGENT CURRICULUM: 5 PHASES (800-1000 EPISODES)
# ============================================================================

MULTI_AGENT_PHASES = [
    # ========================================================================
    # PHASE 1: BASIC COORDINATION (Episodes 0-200)
    # ========================================================================
    # Goal: Learn basic coordination in empty space
    # - No obstacles to complicate movement
    # - Focus purely on spatial coordination
    # - Learn to spread out and avoid overlap
    # - High epsilon to explore joint action space
    MultiAgentCurriculumPhase(
        episode_start=0,
        episode_end=200,
        map_type_distribution={
            "empty": 1.0
        },
        coverage_target=0.50,  # Lower than single-agent (coordination takes time)
        overlap_target=0.20,    # Accept 20% overlap initially
        epsilon_start=1.0,
        epsilon_end=0.7,        # Slower decay than single-agent
        description="Basic Coordination - Empty Maps"
    ),
    
    # ========================================================================
    # PHASE 2: COORDINATION WITH OBSTACLES (Episodes 200-400)
    # ========================================================================
    # Goal: Coordinate while navigating obstacles
    # - Introduce sparse obstacles
    # - Learn to coordinate around obstacles
    # - Reduce overlap as coordination improves
    # - Continue high exploration
    MultiAgentCurriculumPhase(
        episode_start=200,
        episode_end=400,
        map_type_distribution={
            "empty": 0.7,
            "random": 0.3
        },
        coverage_target=0.55,
        overlap_target=0.18,    # Reduce overlap slightly
        epsilon_start=0.7,
        epsilon_end=0.5,
        description="Coordination with Obstacles"
    ),
    
    # ========================================================================
    # PHASE 3: MIXED OBSTACLES (Episodes 400-600)
    # ========================================================================
    # Goal: Maintain coordination under obstacle pressure
    # - Mixed obstacle densities
    # - Learn to coordinate in constrained spaces
    # - Further reduce overlap
    # - Moderate exploration
    MultiAgentCurriculumPhase(
        episode_start=400,
        episode_end=600,
        map_type_distribution={
            "empty": 0.4,
            "random": 0.6
        },
        coverage_target=0.58,
        overlap_target=0.15,    # Target more efficient coordination
        epsilon_start=0.5,
        epsilon_end=0.3,
        description="Mixed Obstacle Coordination"
    ),
    
    # ========================================================================
    # PHASE 4: DOORWAY COORDINATION (Episodes 600-800)
    # ========================================================================
    # Goal: Learn doorway negotiation (HARD for multi-agent)
    # - Introduce rooms with doorway bottlenecks
    # - Learn to queue at doorways
    # - Avoid doorway collisions
    # - This is HARDER than single-agent (coordination bottleneck)
    MultiAgentCurriculumPhase(
        episode_start=600,
        episode_end=800,
        map_type_distribution={
            "empty": 0.3,
            "random": 0.5,
            "room": 0.2  # Introduce rooms carefully
        },
        coverage_target=0.60,   # Don't expect high coverage (rooms are hard)
        overlap_target=0.12,
        epsilon_start=0.3,
        epsilon_end=0.15,
        description="Doorway Coordination - Rooms Introduced"
    ),
    
    # ========================================================================
    # PHASE 5: GENERALIZATION (Episodes 800-1000)
    # ========================================================================
    # Goal: Generalize coordination across all map types
    # - All map types including dense obstacles
    # - Consolidate coordination skills
    # - Minimize overlap
    # - Low exploration (exploit learned coordination)
    MultiAgentCurriculumPhase(
        episode_start=800,
        episode_end=1000,
        map_type_distribution={
            "empty": 0.2,
            "random": 0.5,
            "room": 0.15,
            "corridor": 0.15
        },
        coverage_target=0.62,   # Final target (with 0.85 threshold)
        overlap_target=0.10,    # Minimize overlap
        epsilon_start=0.15,
        epsilon_end=0.05,       # Lower final epsilon than single-agent
        description="Coordination Generalization"
    )
]


def get_multi_agent_curriculum_phase(episode: int) -> MultiAgentCurriculumPhase:
    """
    Get curriculum phase for current episode.
    
    Args:
        episode: Current episode number
        
    Returns:
        phase: Curriculum phase configuration
    """
    for phase in MULTI_AGENT_PHASES:
        if phase.episode_start <= episode < phase.episode_end:
            return phase
    
    # Return last phase if beyond curriculum
    return MULTI_AGENT_PHASES[-1]


def get_multi_agent_epsilon(episode: int) -> float:
    """
    Get epsilon for current episode.
    
    Multi-agent epsilon decays SLOWER than single-agent because:
    - Joint action space is exponentially larger
    - Need more exploration to find good coordination patterns
    - Example: 4 agents × 8 actions = 4096 joint actions vs 8 single actions
    
    Args:
        episode: Current episode number
        
    Returns:
        epsilon: Exploration rate
    """
    phase = get_multi_agent_curriculum_phase(episode)
    return phase.get_epsilon(episode)


def get_multi_agent_map_type(episode: int) -> str:
    """
    Sample map type for current episode.
    
    Args:
        episode: Current episode number
        
    Returns:
        map_type: One of ['empty', 'random', 'room', 'corridor']
    """
    phase = get_multi_agent_curriculum_phase(episode)
    return phase.sample_map_type()


def print_multi_agent_curriculum_summary():
    """Print summary of multi-agent curriculum."""
    print("\n" + "="*80)
    print("MULTI-AGENT COORDINATION CURRICULUM")
    print("="*80)
    print("\nKey Differences from Single-Agent:")
    print("- Duration: 1000 episodes (vs 2250 single-agent)")
    print("- Focus: Coordination learning (coverage skills transferred)")
    print("- Rooms: Introduced in Phase 4 (vs Phase 2 single-agent)")
    print("- Epsilon: Slower decay (larger joint action space)")
    print("- Metrics: Coverage + Overlap + Spacing")
    print("- Early Termination: Enabled (encourages efficiency)")
    print("\n" + "-"*80)
    
    for i, phase in enumerate(MULTI_AGENT_PHASES, 1):
        print(f"\nPhase {i}: {phase.description}")
        print(f"  Episodes: {phase.episode_start}-{phase.episode_end}")
        print(f"  Epsilon: {phase.epsilon_start:.2f} → {phase.epsilon_end:.2f}")
        print(f"  Coverage Target: {phase.coverage_target:.1%}")
        print(f"  Overlap Target: ≤{phase.overlap_target:.1%}")
        print(f"  Map Types:")
        for map_type, prob in phase.map_type_distribution.items():
            print(f"    - {map_type}: {prob:.0%}")
    
    print("\n" + "="*80)
    print(f"Total Episodes: {MULTI_AGENT_PHASES[-1].episode_end}")
    print(f"Estimated Training Time: ~6-8 hours (4 agents, early termination)")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Print curriculum summary
    print_multi_agent_curriculum_summary()
    
    # Test epsilon decay
    print("\nEpsilon Decay Schedule:")
    print("-" * 40)
    test_episodes = [0, 100, 200, 400, 600, 800, 1000]
    for ep in test_episodes:
        phase = get_multi_agent_curriculum_phase(ep)
        epsilon = get_multi_agent_epsilon(ep)
        print(f"Episode {ep:4d}: ε={epsilon:.3f} | {phase.description}")
    
    # Test map type distribution
    print("\nMap Type Sampling (Episode 700):")
    print("-" * 40)
    phase = get_multi_agent_curriculum_phase(700)
    samples = [get_multi_agent_map_type(700) for _ in range(100)]
    from collections import Counter
    distribution = Counter(samples)
    for map_type, count in sorted(distribution.items()):
        print(f"  {map_type}: {count}% (target: {phase.map_type_distribution.get(map_type, 0)*100:.0f}%)")
