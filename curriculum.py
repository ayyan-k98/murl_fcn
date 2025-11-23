"""
Curriculum Manager

Manages 13-phase curriculum learning with mastery gates.
Each phase has its own epsilon decay strategy for optimal exploration.
"""

from typing import List, Optional
from data_structures import CurriculumPhase
from config import config


class CurriculumManager:
    """
    Manages curriculum learning phases.

    13 progressive phases + 3 consolidation cycles for overlearning.
    """

    def __init__(self):
        self.phases = self._initialize_phases()
        self.current_phase_idx = 0
        self.phase_transitions = []

    def _initialize_phases(self) -> List[CurriculumPhase]:
        """
        Initialize all 13 curriculum phases + consolidation - 1500 EPISODES TOTAL.

        Design principles:
            - Gradual difficulty increase
            - 1200 episodes for core curriculum (13 phases)
            - 300 episodes for consolidation
            - Interleaving of map types
            - Mastery gates (expected_coverage thresholds adjusted for 0.85 threshold)
            - Phase-specific epsilon decay (fast for simple, slow for complex)
        
        Total: 1500 episodes (~8.3 hours @ 20s/ep)
        """
        phases = [
            # Phase 1: Foundation - Pure open environments (250 eps)
            CurriculumPhase(
                name="Phase1_Foundation_PureOpen",
                start_ep=0,
                end_ep=250,
                map_distribution={"empty": 1.0},
                expected_coverage=0.60,  # Reduced from 0.70 due to higher threshold
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE1
            ),

            # Phase 2: Introduce random obstacles (150 eps)
            CurriculumPhase(
                name="Phase2_IntroObstacles",
                start_ep=250,
                end_ep=400,
                map_distribution={"empty": 0.6, "random": 0.4},
                expected_coverage=0.60,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE2
            ),

            # Phase 3: More random obstacles (120 eps)
            CurriculumPhase(
                name="Phase3_MoreRandom",
                start_ep=400,
                end_ep=520,
                map_distribution={"random": 0.6, "empty": 0.4},
                expected_coverage=0.62,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE3
            ),

            # Phase 4: Consolidation 1 (40 eps)
            CurriculumPhase(
                name="Phase4_Consolidation1",
                start_ep=520,
                end_ep=560,
                map_distribution={"empty": 0.5, "random": 0.5},
                expected_coverage=0.65,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE4
            ),

            # Phase 5: Introduce rooms (120 eps)
            CurriculumPhase(
                name="Phase5_IntroRooms",
                start_ep=560,
                end_ep=680,
                map_distribution={"room": 0.4, "empty": 0.3, "random": 0.3},
                expected_coverage=0.62,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE5
            ),

            # Phase 6: More rooms (100 eps)
            CurriculumPhase(
                name="Phase6_MoreRooms",
                start_ep=680,
                end_ep=780,
                map_distribution={"room": 0.55, "random": 0.25, "empty": 0.20},
                expected_coverage=0.65,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE6
            ),

            # Phase 7: Consolidation 2 (40 eps)
            CurriculumPhase(
                name="Phase7_Consolidation2",
                start_ep=780,
                end_ep=820,
                map_distribution={"empty": 0.35, "random": 0.35, "room": 0.30},
                expected_coverage=0.68,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE7
            ),

            # Phase 8: Introduce corridors (100 eps)
            CurriculumPhase(
                name="Phase8_IntroCorridor",
                start_ep=820,
                end_ep=920,
                map_distribution={"room": 0.45, "corridor": 0.25, "random": 0.20, "empty": 0.10},
                expected_coverage=0.63,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE8
            ),

            # Phase 9: Introduce caves (100 eps)
            CurriculumPhase(
                name="Phase9_IntroCave",
                start_ep=920,
                end_ep=1020,
                map_distribution={"room": 0.35, "cave": 0.25, "corridor": 0.20, "random": 0.20},
                expected_coverage=0.60,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE9
            ),

            # Phase 10: Consolidation 3 (40 eps)
            CurriculumPhase(
                name="Phase10_Consolidation3",
                start_ep=1020,
                end_ep=1060,
                map_distribution={"room": 0.40, "corridor": 0.25, "cave": 0.20, "random": 0.15},
                expected_coverage=0.62,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE10
            ),

            # Phase 11: Introduce L-shapes (70 eps)
            CurriculumPhase(
                name="Phase11_IntroLShape",
                start_ep=1060,
                end_ep=1130,
                map_distribution={"room": 0.30, "cave": 0.20, "lshape": 0.20, "corridor": 0.15, "random": 0.15},
                expected_coverage=0.58,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE11
            ),

            # Phase 12: Complex mix (40 eps)
            CurriculumPhase(
                name="Phase12_ComplexMix",
                start_ep=1130,
                end_ep=1170,
                map_distribution={"room": 0.25, "cave": 0.20, "lshape": 0.20, "corridor": 0.20, "random": 0.15},
                expected_coverage=0.60,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE12
            ),

            # Phase 13: Final polish (30 eps)
            CurriculumPhase(
                name="Phase13_FinalPolish",
                start_ep=1170,
                end_ep=1200,
                map_distribution={"room": 0.25, "empty": 0.20, "random": 0.20, "cave": 0.15, "corridor": 0.10, "lshape": 0.10},
                expected_coverage=0.62,
                epsilon_floor=0.05,
                epsilon_decay=config.EPSILON_DECAY_PHASE13
            ),
            
            # Phase 14: Extended consolidation - All map types (300 eps)
            CurriculumPhase(
                name="Phase14_ExtendedConsolidation",
                start_ep=1200,
                end_ep=1500,
                map_distribution={
                    "room": 0.30,
                    "cave": 0.20,
                    "corridor": 0.15,
                    "random": 0.15,
                    "lshape": 0.10,
                    "empty": 0.10
                },
                expected_coverage=0.65,
                epsilon_floor=0.05,
                epsilon_decay=0.995  # Very slow decay for consolidation
            ),
        ]

        return phases

    def get_current_phase(self, episode: int) -> CurriculumPhase:
        """Get the active phase for given episode."""
        for phase in self.phases:
            if phase.is_active(episode):
                return phase

        # If past all phases, return last phase
        return self.phases[-1]

    def get_map_type(self, episode: int) -> str:
        """Sample a map type for current episode."""
        phase = self.get_current_phase(episode)
        return phase.sample_map_type()

    def get_epsilon_floor(self, episode: int) -> float:
        """Get minimum epsilon for current phase."""
        phase = self.get_current_phase(episode)
        return phase.epsilon_floor

    def get_epsilon_decay(self, episode: int) -> float:
        """Get epsilon decay rate for current phase."""
        phase = self.get_current_phase(episode)
        return phase.epsilon_decay

    def check_mastery(self, episode: int, avg_coverage: float) -> bool:
        """
        Check if agent has achieved mastery for current phase.

        Args:
            episode: Current episode
            avg_coverage: Average coverage over recent episodes

        Returns:
            True if mastery achieved
        """
        phase = self.get_current_phase(episode)
        return avg_coverage >= phase.expected_coverage

    def should_advance(self, episode: int, avg_coverage: float) -> bool:
        """Check if should advance to next phase."""
        # Can only advance if at end of current phase and mastery achieved
        phase = self.get_current_phase(episode)
        at_phase_end = episode >= phase.end_ep - 1
        has_mastery = self.check_mastery(episode, avg_coverage)

        return at_phase_end and has_mastery

    def get_summary(self) -> str:
        """Get curriculum summary."""
        summary = "=" * 80 + "\n"
        summary += "CURRICULUM OVERVIEW (1500 Episodes: 1200 Core + 300 Consolidation)\n"
        summary += "=" * 80 + "\n"
        summary += f"{'Phase':<6} {'Episodes':<15} {'ε Decay':<10} {'Map Mix':<30} {'Target'}\n"
        summary += "-" * 80 + "\n"

        for i, phase in enumerate(self.phases, 1):
            ep_range = f"{phase.start_ep}-{phase.end_ep}"
            decay_str = f"{phase.epsilon_decay:.4f}"

            # Format map distribution (abbreviated for space)
            map_mix = ", ".join([f"{k[:3]}:{int(v*100)}%" for k, v in sorted(phase.map_distribution.items())])

            target = f"{int(phase.expected_coverage*100)}%+"

            summary += f"{i:<6} {ep_range:<15} {decay_str:<10} {map_mix:<30} {target}\n"

        summary += "=" * 80 + "\n"
        summary += "Key Changes:\n"
        summary += "  - COVERAGE_THRESHOLD increased to 0.85 (cells need 85% coverage to count)\n"
        summary += "  - Expected coverage reduced ~10-15% due to higher threshold\n"
        summary += "  - Rotation penalties added for smoother trajectories\n"
        summary += "  - Phase 14: 300-episode consolidation for mastery\n"
        summary += "\nStrategy:\n"
        summary += "  - Phase 1 (0.98): Fast decay - Agent exploits by ep 50 (ε→0.36)\n"
        summary += "  - Phases 2,5,8,9 (0.985): Moderate decay - New environments\n"
        summary += "  - Phases 3,6,11 (0.987): Slower decay - Complex environments\n"
        summary += "  - Phases 4,7,10,12,13 (0.990-0.992): Slowest - Consolidation\n"
        summary += "  - Phase 14 (0.995): Minimal decay - Extended consolidation\n"
        summary += "=" * 80

        return summary


if __name__ == "__main__":
    # Test curriculum manager
    manager = CurriculumManager()

    print(manager.get_summary())
    print("\n\nPhase transitions:")

    for ep in [0, 200, 500, 1000, 1500, 1600]:
        phase = manager.get_current_phase(ep)
        map_type = manager.get_map_type(ep)
        print(f"  Episode {ep:4d}: {phase.name:30s} -> {map_type}")
