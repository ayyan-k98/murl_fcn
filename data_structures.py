"""
Data structures for GAT-MARL Coverage

Contains state representations, metrics, and curriculum phases.
"""

from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Set
import numpy as np
import networkx as nx

from config import config


@dataclass
class RobotState:
    """State of a single robot (POMDP-aware)."""
    position: Tuple[int, int]
    orientation: float  # Radians
    local_map: Dict[Tuple[int, int], Tuple[float, str]] = field(default_factory=dict)
    visited_positions: Set[Tuple[int, int]] = field(default_factory=set)
    last_action: int = 8  # STAY
    coverage_history: np.ndarray = None
    visit_heat: np.ndarray = None
    coverage_over_time: List[float] = field(default_factory=list)  # Track coverage progression

    def __post_init__(self):
        if self.coverage_history is None:
            self.coverage_history = np.zeros(
                (config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32
            )
        if self.visit_heat is None:
            self.visit_heat = np.zeros(
                (config.GRID_SIZE, config.GRID_SIZE), dtype=np.float32
            )

    def reset_maps(self):
        """Reset local maps for new episode."""
        self.local_map = {}
        self.visited_positions = set()
        self.coverage_history.fill(0.0)
        self.visit_heat.fill(0.0)
        self.last_action = 8
        self.coverage_over_time = []


@dataclass
class WorldState:
    """Ground truth world state (not fully observable by agents)."""
    grid_size: int
    graph: nx.Graph  # Ground truth spatial graph
    obstacles: Set[Tuple[int, int]]
    coverage_map: np.ndarray  # [grid_size, grid_size], values in [0,1]
    map_type: str = "empty"


@dataclass
class CoverageMetrics:
    """Training metrics tracker."""
    episode_rewards: List[float] = field(default_factory=list)
    episode_coverages: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    dqn_loss: List[float] = field(default_factory=list)
    epsilon_values: List[float] = field(default_factory=list)
    grad_norms: List[float] = field(default_factory=list)
    validation_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)
    grad_explosions: int = 0
    zero_gain_steps: int = 0

    def add_episode(self, reward: float, coverage: float, length: int, epsilon: float):
        """Add episode metrics."""
        self.episode_rewards.append(reward)
        self.episode_coverages.append(coverage)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)

    def add_loss(self, loss: float):
        """Add training loss."""
        self.dqn_loss.append(loss)

    def add_grad_norm(self, norm: float):
        """Add gradient norm."""
        self.grad_norms.append(norm)
        if norm > config.EXPLOSION_THRESHOLD:
            self.grad_explosions += 1

    def get_recent_avg(self, metric: str, window: int = 100) -> float:
        """Get recent average of a metric."""
        if metric == "reward":
            data = self.episode_rewards
        elif metric == "coverage":
            data = self.episode_coverages
        elif metric == "length":
            data = self.episode_lengths
        elif metric == "loss":
            data = self.dqn_loss
        else:
            return 0.0

        if len(data) == 0:
            return 0.0
        return np.mean(data[-window:])


@dataclass
class CurriculumPhase:
    """Single curriculum phase definition."""
    name: str
    start_ep: int
    end_ep: int
    map_distribution: Dict[str, float]  # map_type -> probability
    expected_coverage: float  # Mastery threshold
    epsilon_floor: float  # Min exploration for this phase
    epsilon_decay: float = 0.995  # Phase-specific decay rate

    def sample_map_type(self) -> str:
        """Sample a map type according to distribution."""
        map_types = list(self.map_distribution.keys())
        probs = list(self.map_distribution.values())
        return np.random.choice(map_types, p=probs)

    def is_active(self, episode: int) -> bool:
        """Check if this phase is active for given episode."""
        return self.start_ep <= episode < self.end_ep


if __name__ == "__main__":
    # Test data structures
    robot = RobotState(position=(10, 10), orientation=0.0)
    print(f"✓ RobotState created at {robot.position}")

    metrics = CoverageMetrics()
    metrics.add_episode(100.0, 0.75, 250, 0.5)
    print(f"✓ Metrics: avg reward = {metrics.get_recent_avg('reward')}")

    phase = CurriculumPhase(
        name="Test",
        start_ep=0,
        end_ep=100,
        map_distribution={"empty": 0.6, "random": 0.4},
        expected_coverage=0.70,
        epsilon_floor=0.2
    )
    print(f"✓ CurriculumPhase active(50)? {phase.is_active(50)}")
    print(f"  Sampled map: {phase.sample_map_type()}")
