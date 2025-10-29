"""
Coordination Metrics for Multi-Agent Coverage

Quantifies coordination quality beyond just coverage percentage.

Key Metrics:
1. Overlap Ratio - How much redundant coverage occurs
2. Exploration Efficiency - How well agents spread out
3. Communication Effectiveness - Impact of messages on decisions
4. Collision Rate - Agent-agent and agent-obstacle collisions
5. Load Balance - How evenly work is distributed
6. Frontier Sharing - How well agents coordinate on boundaries
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class CoordinationMetrics:
    """Container for coordination quality metrics."""
    
    # Overlap metrics (lower is better)
    overlap_ratio: float = 0.0          # cells visited by >1 agent / total visited
    redundant_visits: int = 0           # total redundant visits
    max_overlap_per_cell: int = 0      # worst case overlap
    
    # Exploration efficiency (higher is better)
    exploration_efficiency: float = 0.0  # unique_cells / total_visits
    spatial_dispersion: float = 0.0      # how spread out agents are (0-1)
    
    # Communication metrics
    comm_messages_sent: int = 0
    comm_messages_received: int = 0
    comm_influenced_decisions: int = 0   # decisions changed due to comm
    
    # Collision metrics (lower is better)
    agent_collisions: int = 0
    obstacle_collisions: int = 0
    near_misses: int = 0                # agents within 1 cell
    
    # Load balance (closer to 1.0 is better)
    load_balance_ratio: float = 0.0     # min_work / max_work
    coverage_per_agent: List[float] = field(default_factory=list)
    
    # Frontier coordination (higher is better)
    frontier_sharing_score: float = 0.0  # how well agents divide frontiers
    territory_overlap: float = 0.0       # overlap in explored territories
    
    # Timing metrics
    completion_time: int = 0             # steps to reach target coverage
    idle_steps: int = 0                  # steps where agents do nothing useful


class CoordinationAnalyzer:
    """
    Analyzes coordination quality during multi-agent episodes.
    
    Usage:
        analyzer = CoordinationAnalyzer(num_agents=4, grid_size=20)
        
        for step in episode:
            analyzer.update(positions, visited_maps, actions, messages)
        
        metrics = analyzer.get_metrics()
        print(f"Overlap ratio: {metrics.overlap_ratio:.2%}")
    """
    
    def __init__(self, num_agents: int, grid_size: int):
        self.num_agents = num_agents
        self.grid_size = grid_size
        
        # Tracking data
        self.visit_count_map = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.agent_visit_maps = [
            np.zeros((grid_size, grid_size), dtype=bool)
            for _ in range(num_agents)
        ]
        
        self.position_history = [[] for _ in range(num_agents)]
        self.collision_history = []
        self.near_miss_history = []
        
        self.messages_sent = 0
        self.messages_received = 0
        self.decisions_influenced = 0
        
        self.steps = 0
        self.coverage_per_agent = [0] * num_agents
    
    def update(self,
               positions: List[Tuple[int, int]],
               visited_maps: List[np.ndarray],
               actions: List[int] = None,
               messages: List[Dict] = None):
        """
        Update metrics with current step information.
        
        Args:
            positions: Current position of each agent [(x, y), ...]
            visited_maps: Boolean array of visited cells per agent
            actions: Actions taken by each agent (optional)
            messages: Communication messages exchanged (optional)
        """
        self.steps += 1
        
        # Update visit counts
        for i, (x, y) in enumerate(positions):
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                self.visit_count_map[y, x] += 1
                self.agent_visit_maps[i][y, x] = True
                self.position_history[i].append((x, y))
        
        # Track individual agent coverage
        for i in range(self.num_agents):
            self.coverage_per_agent[i] = np.sum(visited_maps[i]) if visited_maps else 0
        
        # Detect collisions (same position)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                if positions[i] == positions[j]:
                    self.collision_history.append((self.steps, i, j, positions[i]))
        
        # Detect near misses (adjacent positions)
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = abs(positions[i][0] - positions[j][0]) + abs(positions[i][1] - positions[j][1])
                if dist == 1:
                    self.near_miss_history.append((self.steps, i, j))
        
        # Track communication
        if messages:
            self.messages_sent += len(messages)
            # Each message potentially influences other agents
            self.messages_received += len(messages) * (self.num_agents - 1)
    
    def update_communication_influence(self, influenced: int):
        """
        Record that communication influenced decision-making.
        
        Args:
            influenced: Number of agents whose decisions were influenced
        """
        self.decisions_influenced += influenced
    
    def get_metrics(self) -> CoordinationMetrics:
        """
        Compute final coordination metrics.
        
        Returns:
            CoordinationMetrics object with all computed metrics
        """
        metrics = CoordinationMetrics()
        
        # === OVERLAP METRICS ===
        total_visited = np.sum(self.visit_count_map > 0)
        if total_visited > 0:
            # Cells visited by multiple agents
            overlapped_cells = np.sum(self.visit_count_map > 1)
            metrics.overlap_ratio = overlapped_cells / total_visited
            
            # Total redundant visits
            metrics.redundant_visits = int(np.sum(self.visit_count_map) - total_visited)
            
            # Maximum overlap per cell
            metrics.max_overlap_per_cell = int(np.max(self.visit_count_map))
        
        # === EXPLORATION EFFICIENCY ===
        total_visits = np.sum(self.visit_count_map)
        if total_visits > 0:
            metrics.exploration_efficiency = total_visited / total_visits
        
        # Spatial dispersion (how spread out agents are)
        metrics.spatial_dispersion = self._compute_spatial_dispersion()
        
        # === COMMUNICATION METRICS ===
        metrics.comm_messages_sent = self.messages_sent
        metrics.comm_messages_received = self.messages_received
        metrics.comm_influenced_decisions = self.decisions_influenced
        
        # === COLLISION METRICS ===
        metrics.agent_collisions = len(self.collision_history)
        metrics.near_misses = len(self.near_miss_history)
        
        # === LOAD BALANCE ===
        if len(self.coverage_per_agent) > 0 and max(self.coverage_per_agent) > 0:
            min_coverage = min(self.coverage_per_agent)
            max_coverage = max(self.coverage_per_agent)
            metrics.load_balance_ratio = min_coverage / max_coverage if max_coverage > 0 else 0.0
            metrics.coverage_per_agent = self.coverage_per_agent.copy()
        
        # === FRONTIER METRICS ===
        metrics.frontier_sharing_score = self._compute_frontier_sharing()
        metrics.territory_overlap = self._compute_territory_overlap()
        
        # === TIMING ===
        metrics.completion_time = self.steps
        
        return metrics
    
    def _compute_spatial_dispersion(self) -> float:
        """
        Compute how spread out agents are (0=clustered, 1=maximally dispersed).
        
        Uses average pairwise distance normalized by grid diagonal.
        """
        if self.steps == 0 or not any(self.position_history):
            return 0.0
        
        # Get most recent positions
        recent_positions = []
        for history in self.position_history:
            if history:
                recent_positions.append(history[-1])
        
        if len(recent_positions) < 2:
            return 0.0
        
        # Average pairwise distance
        total_dist = 0
        count = 0
        for i in range(len(recent_positions)):
            for j in range(i + 1, len(recent_positions)):
                x1, y1 = recent_positions[i]
                x2, y2 = recent_positions[j]
                dist = abs(x1 - x2) + abs(y1 - y2)
                total_dist += dist
                count += 1
        
        avg_dist = total_dist / count if count > 0 else 0
        
        # Normalize by maximum possible distance (grid diagonal)
        max_dist = 2 * (self.grid_size - 1)
        return min(avg_dist / max_dist, 1.0) if max_dist > 0 else 0.0
    
    def _compute_frontier_sharing(self) -> float:
        """
        Compute how well agents divide frontier exploration (0-1).
        
        Higher scores = less overlap in frontier regions.
        """
        # Find frontier cells for each agent
        frontiers = []
        for i in range(self.num_agents):
            visited = self.agent_visit_maps[i]
            frontier = self._find_frontier(visited)
            frontiers.append(frontier)
        
        if sum(len(f) for f in frontiers) == 0:
            return 1.0  # No frontiers = perfect sharing (trivial)
        
        # Count overlapping frontier cells
        frontier_overlap = 0
        total_frontier = 0
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                overlap = frontiers[i].intersection(frontiers[j])
                frontier_overlap += len(overlap)
            total_frontier += len(frontiers[i])
        
        if total_frontier == 0:
            return 1.0
        
        # Score: 1.0 - (overlap / total)
        return max(0.0, 1.0 - (frontier_overlap / total_frontier))
    
    def _compute_territory_overlap(self) -> float:
        """
        Compute overlap in explored territories (0-1).
        
        Lower scores = better territorial division.
        """
        total_overlap = 0
        total_cells = 0
        
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                overlap = np.logical_and(
                    self.agent_visit_maps[i],
                    self.agent_visit_maps[j]
                )
                total_overlap += np.sum(overlap)
            
            total_cells += np.sum(self.agent_visit_maps[i])
        
        if total_cells == 0:
            return 0.0
        
        return total_overlap / total_cells
    
    def _find_frontier(self, visited: np.ndarray) -> Set[Tuple[int, int]]:
        """
        Find frontier cells (unvisited cells adjacent to visited cells).
        
        Args:
            visited: Boolean array of visited cells
            
        Returns:
            Set of (x, y) frontier positions
        """
        frontier = set()
        
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                if visited[y, x]:
                    # Check 4-connected neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.grid_size and 
                            0 <= ny < self.grid_size and 
                            not visited[ny, nx]):
                            frontier.add((nx, ny))
        
        return frontier
    
    def reset(self):
        """Reset all tracking data for new episode."""
        self.visit_count_map.fill(0)
        for visit_map in self.agent_visit_maps:
            visit_map.fill(False)
        
        self.position_history = [[] for _ in range(self.num_agents)]
        self.collision_history.clear()
        self.near_miss_history.clear()
        
        self.messages_sent = 0
        self.messages_received = 0
        self.decisions_influenced = 0
        
        self.steps = 0
        self.coverage_per_agent = [0] * self.num_agents
    
    def print_summary(self, metrics: CoordinationMetrics = None):
        """Print human-readable summary of metrics."""
        if metrics is None:
            metrics = self.get_metrics()
        
        print("\n" + "="*70)
        print("COORDINATION METRICS SUMMARY")
        print("="*70)
        
        print("\n📊 Overlap & Efficiency:")
        print(f"  Overlap Ratio:           {metrics.overlap_ratio:.1%} (lower is better)")
        print(f"  Redundant Visits:        {metrics.redundant_visits}")
        print(f"  Exploration Efficiency:  {metrics.exploration_efficiency:.1%} (higher is better)")
        print(f"  Spatial Dispersion:      {metrics.spatial_dispersion:.1%} (higher is better)")
        
        print("\n💬 Communication:")
        print(f"  Messages Sent:           {metrics.comm_messages_sent}")
        print(f"  Messages Received:       {metrics.comm_messages_received}")
        print(f"  Influenced Decisions:    {metrics.comm_influenced_decisions}")
        
        print("\n⚠️  Collisions:")
        print(f"  Agent Collisions:        {metrics.agent_collisions}")
        print(f"  Near Misses:             {metrics.near_misses}")
        
        print("\n⚖️  Load Balance:")
        print(f"  Balance Ratio:           {metrics.load_balance_ratio:.2f} (closer to 1.0 is better)")
        if metrics.coverage_per_agent:
            print(f"  Coverage per Agent:      {[f'{c:.0f}' for c in metrics.coverage_per_agent]}")
        
        print("\n🎯 Frontier Coordination:")
        print(f"  Frontier Sharing:        {metrics.frontier_sharing_score:.1%} (higher is better)")
        print(f"  Territory Overlap:       {metrics.territory_overlap:.1%} (lower is better)")
        
        print("\n⏱️  Timing:")
        print(f"  Steps:                   {metrics.completion_time}")
        
        print("="*70 + "\n")


def coordination_score(metrics: CoordinationMetrics) -> float:
    """
    Compute overall coordination quality score (0-100).
    
    Weighted combination of metrics:
    - 30% Exploration efficiency (avoid redundancy)
    - 20% Load balance (fair work distribution)
    - 20% Spatial dispersion (spread out)
    - 15% Frontier sharing (coordinate boundaries)
    - 10% Low collisions (avoid conflicts)
    - 5% Communication effectiveness
    
    Args:
        metrics: CoordinationMetrics object
        
    Returns:
        score: Float in [0, 100] where higher is better
    """
    score = 0.0
    
    # Exploration efficiency (30 points)
    score += metrics.exploration_efficiency * 30
    
    # Load balance (20 points)
    score += metrics.load_balance_ratio * 20
    
    # Spatial dispersion (20 points)
    score += metrics.spatial_dispersion * 20
    
    # Frontier sharing (15 points)
    score += metrics.frontier_sharing_score * 15
    
    # Collision penalty (10 points)
    # Assume 0 collisions = 10 points, scale down with more collisions
    collision_penalty = min(metrics.agent_collisions * 0.5, 10)
    score += max(0, 10 - collision_penalty)
    
    # Communication effectiveness (5 points)
    if metrics.comm_messages_sent > 0:
        comm_effectiveness = metrics.comm_influenced_decisions / metrics.comm_messages_sent
        score += min(comm_effectiveness, 1.0) * 5
    
    return min(score, 100.0)


# ==============================================================================
# TEST CODE
# ==============================================================================

if __name__ == "__main__":
    print("Testing CoordinationAnalyzer...")
    
    # Scenario 1: Good coordination (agents spread out, no overlap)
    print("\n" + "="*70)
    print("SCENARIO 1: Good Coordination (4 agents, well-dispersed)")
    print("="*70)
    
    analyzer = CoordinationAnalyzer(num_agents=4, grid_size=20)
    
    # Simulate 100 steps with agents in different quadrants
    for step in range(100):
        positions = [
            (5, 5 + step // 10),   # Agent 0: top-left
            (15, 5 + step // 10),  # Agent 1: top-right
            (5, 15 + step // 10),  # Agent 2: bottom-left
            (15, 15 + step // 10)  # Agent 3: bottom-right
        ]
        
        # Create simple visited maps
        visited_maps = [
            np.random.rand(20, 20) < 0.3 for _ in range(4)
        ]
        
        analyzer.update(positions, visited_maps)
    
    metrics1 = analyzer.get_metrics()
    analyzer.print_summary(metrics1)
    print(f"Overall Coordination Score: {coordination_score(metrics1):.1f}/100\n")
    
    # Scenario 2: Poor coordination (agents clustered, lots of overlap)
    print("\n" + "="*70)
    print("SCENARIO 2: Poor Coordination (4 agents, clustered)")
    print("="*70)
    
    analyzer.reset()
    
    # Simulate 100 steps with agents clustered together
    for step in range(100):
        positions = [
            (10, 10),              # Agent 0: center
            (10, 11),              # Agent 1: near center
            (11, 10),              # Agent 2: near center
            (11, 11)               # Agent 3: near center
        ]
        
        visited_maps = [
            np.random.rand(20, 20) < 0.3 for _ in range(4)
        ]
        
        analyzer.update(positions, visited_maps)
    
    metrics2 = analyzer.get_metrics()
    analyzer.print_summary(metrics2)
    print(f"Overall Coordination Score: {coordination_score(metrics2):.1f}/100\n")
    
    print("\n✓ CoordinationAnalyzer test complete!")
