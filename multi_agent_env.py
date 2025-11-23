"""
Multi-Agent Coverage Environment

Extension of the single-agent coverage system for multi-robot coordination with QMIX.

Key Features:
- Multiple agents with independent POMDP observations
- Full state sharing within communication range (for coordination)
- Agent-agent collision detection and penalties
- Joint reward structure (cooperative)
- CTDE (Centralized Training, Decentralized Execution) support

Reward Function:
1. Joint Coverage Reward: +10 per new cell covered (shared by all agents)
2. Collision Penalty: -2.0 for agent-agent collision
3. Separation Incentive: -0.5 for being adjacent to another agent
4. Redundancy Penalty: -0.1 per cell covered by multiple agents
"""

import math
import random
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum
from dataclasses import dataclass, field
import numpy as np
import networkx as nx

from config import config
from data_structures import RobotState, WorldState
from environment import CoverageEnvironment


class CoordinationStrategy(Enum):
    """Multi-agent coordination strategies."""
    INDEPENDENT = "independent"  # No coordination, purely independent agents
    HIERARCHICAL = "hierarchical"  # Leader assigns tasks to followers


@dataclass
class AgentState:
    """State of a single agent in multi-agent setting."""
    agent_id: int
    robot_state: RobotState
    assigned_region: Optional[Set[Tuple[int, int]]] = None  # For Hierarchical
    task_assignment: Optional[Tuple[int, int]] = None  # For Hierarchical
    communication_range: float = 5.0

    def reset_assignment(self):
        """Reset region and task assignments."""
        self.assigned_region = None
        self.task_assignment = None


@dataclass
class MultiAgentState:
    """Complete multi-agent system state."""
    agents: List[AgentState]
    world_state: WorldState
    coordination: CoordinationStrategy
    step_count: int = 0

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def get_agent_positions(self) -> List[Tuple[int, int]]:
        """Get all agent positions."""
        return [agent.robot_state.position for agent in self.agents]

    def get_nearby_agents(self, agent_id: int) -> List[int]:
        """Get IDs of agents within communication range."""
        agent = self.agents[agent_id]
        pos = agent.robot_state.position
        comm_range = agent.communication_range

        nearby = []
        for other in self.agents:
            if other.agent_id == agent_id:
                continue

            other_pos = other.robot_state.position
            distance = math.sqrt(
                (pos[0] - other_pos[0])**2 + (pos[1] - other_pos[1])**2
            )

            if distance <= comm_range:
                nearby.append(other.agent_id)

        return nearby


class MultiAgentCoverageEnv:
    """
    Multi-agent coverage environment with coordination.

    Supports 2-8 agents with various coordination strategies.

    Key Methods:
        reset() -> MultiAgentState
        step(actions: List[int]) -> (next_state, rewards, done, info)
        get_observations() -> List[observation]
    """

    def __init__(
        self,
        num_agents: int = 4,
        grid_size: int = 20,
        sensor_range: float = 3.0,
        communication_range: float = 5.0,
        coordination: CoordinationStrategy = CoordinationStrategy.INDEPENDENT,
        map_type: str = "empty",
        collision_penalty: float = None,
        team_reward_weight: float = 0.5
    ):
        """
        Initialize multi-agent environment.

        Args:
            num_agents: Number of agents [2-8]
            grid_size: Grid dimension
            sensor_range: POMDP sensor range
            communication_range: Agent communication range
            coordination: Coordination strategy
            map_type: Map type (empty, random, maze, office, warehouse)
            collision_penalty: Penalty for agent-agent collision
            team_reward_weight: Weight for team reward vs individual [0-1]
        """
        assert 2 <= num_agents <= 8, "num_agents must be in [2, 8]"
        assert 0 <= team_reward_weight <= 1.0, "team_reward_weight must be in [0, 1]"

        self.num_agents = num_agents
        self.grid_size = grid_size
        self.sensor_range = sensor_range
        self.communication_range = communication_range
        self.coordination = coordination
        self.map_type = map_type
        self.collision_penalty = collision_penalty if collision_penalty is not None else config.COLLISION_PENALTY
        self.team_reward_weight = team_reward_weight

        # Create base single-agent environments for reusing logic
        self.base_env = CoverageEnvironment(
            grid_size=grid_size,
            sensor_range=sensor_range,
            map_type=map_type
        )

        # Multi-agent state
        self.state: Optional[MultiAgentState] = None

        # Episode tracking
        self.max_steps = config.MAX_EPISODE_STEPS

        # Coordination state (strategy-specific)
        self.voronoi_regions: Optional[Dict[int, Set[Tuple[int, int]]]] = None
        self.market_bids: Optional[Dict[int, Dict[Tuple[int, int], float]]] = None
        self.leader_id: Optional[int] = None
        
        # Track last actions for rotation penalty (per agent)
        self.last_actions: List[Optional[int]] = [None] * num_agents

        # Action angle mapping (in degrees, 0° = North)
        self.action_angles = {
            0: 0,    # N
            1: 45,   # NE
            2: 90,   # E
            3: 135,  # SE
            4: 180,  # S
            5: 225,  # SW
            6: 270,  # W
            7: 315,  # NW
            8: None  # STAY (no direction)
        }

        # Communication system (optional)
        # Will be initialized in set_communication_protocol()
        self.comm_system = None
        self.use_communication = False

    def set_communication_protocol(self, comm_protocol):
        """
        Set communication protocol for the environment.

        Args:
            comm_protocol: Communication protocol instance (from communication.py)
        """
        from communication import PositionCommunication

        self.comm_system = comm_protocol
        self.use_communication = isinstance(comm_protocol, PositionCommunication)

        if self.use_communication:
            print(f"✓ Communication enabled: {comm_protocol.__class__.__name__}")
            print(f"  Range: {comm_protocol.comm_range:.1f} cells")
            print(f"  Frequency: Every {comm_protocol.comm_freq} steps")
        else:
            print("✓ Communication disabled (NoCommunciation)")

    def reset(self, map_type: Optional[str] = None) -> MultiAgentState:
        """
        Reset environment for new multi-agent episode.

        Args:
            map_type: Override default map type

        Returns:
            Initial multi-agent state
        """
        if map_type is not None:
            self.map_type = map_type

        # Generate shared world (obstacles, graph)
        graph, obstacles = self.base_env.map_generator.generate(self.map_type)

        # Initialize shared coverage map
        coverage_map = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        world_state = WorldState(
            grid_size=self.grid_size,
            graph=graph,
            obstacles=obstacles,
            coverage_map=coverage_map,
            map_type=self.map_type
        )

        # Initialize agents at spatially distributed start positions
        start_positions = self._generate_start_positions(obstacles)

        agents = []
        for i in range(self.num_agents):
            robot_state = RobotState(
                position=start_positions[i],
                orientation=random.uniform(0, 2 * math.pi),
                coverage_history=np.zeros((self.grid_size, self.grid_size), dtype=np.float32),
                visit_heat=np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            )

            agent_state = AgentState(
                agent_id=i,
                robot_state=robot_state,
                communication_range=self.communication_range
            )

            agents.append(agent_state)

        # Create multi-agent state
        self.state = MultiAgentState(
            agents=agents,
            world_state=world_state,
            coordination=self.coordination,
            step_count=0
        )

        # Initialize coordination strategy
        self._initialize_coordination()

        # Reset last actions for rotation penalty
        self.last_actions = [None] * self.num_agents

        # Reset communication system (if enabled)
        if self.use_communication and self.comm_system is not None:
            self.comm_system.reset()

        # Perform initial sensing for all agents
        for agent in self.state.agents:
            self._update_agent_sensing(agent)

        return self.state

    def step(
        self,
        actions: List[int]
    ) -> Tuple[MultiAgentState, List[float], bool, Dict]:
        """
        Execute multi-agent step.

        Args:
            actions: List of actions [action_0, action_1, ..., action_n-1]

        Returns:
            next_state: Updated multi-agent state
            rewards: List of rewards [reward_0, ..., reward_n-1]
            done: Episode termination flag
            info: Additional information dict
        """
        assert len(actions) == self.num_agents, f"Expected {self.num_agents} actions"

        self.state.step_count += 1

        # Store previous coverage for reward calculation
        prev_coverage_map = self.state.world_state.coverage_map.copy()
        prev_local_map_sizes = [
            len(agent.robot_state.local_map) for agent in self.state.agents
        ]

        # Execute actions simultaneously (detect collisions)
        collision_info = self._execute_actions_parallel(actions)

        # Communication step (if enabled)
        if self.use_communication and self.comm_system is not None:
            # Broadcast positions at configured frequency
            if self.comm_system.should_communicate(self.state.step_count):
                for agent in self.state.agents:
                    self.comm_system.broadcast(
                        agent.agent_id,
                        agent.robot_state.position,
                        velocity=(0.0, 0.0)  # Simplified: velocity not tracked
                    )
            # Increment comm system step counter
            self.comm_system.step()

        # Update coordination strategy
        self._update_coordination()

        # Update sensing for all agents
        for agent in self.state.agents:
            self._update_agent_sensing(agent)

        # Calculate individual rewards
        individual_rewards = []
        coverage_gains = []
        knowledge_gains = []

        for i, agent in enumerate(self.state.agents):
            coverage_gain = self._calculate_agent_coverage_gain(
                agent, prev_coverage_map
            )
            knowledge_gain = len(agent.robot_state.local_map) - prev_local_map_sizes[i]

            # Individual reward components
            reward = self._calculate_agent_reward(
                agent=agent,
                action=actions[i],
                coverage_gain=coverage_gain,
                knowledge_gain=knowledge_gain,
                collision=collision_info['collisions'][i]
            )

            individual_rewards.append(reward)
            coverage_gains.append(coverage_gain)
            knowledge_gains.append(knowledge_gain)

        # Calculate team reward
        team_coverage_gain = self._calculate_total_coverage_gain(prev_coverage_map)
        team_reward = team_coverage_gain * config.COVERAGE_REWARD

        # Blend individual and team rewards
        final_rewards = [
            (1 - self.team_reward_weight) * ind_r + self.team_reward_weight * team_reward
            for ind_r in individual_rewards
        ]
        
        # Apply multi-agent reward normalization (CRITICAL for QMIX stability)
        # Prevents gradient explosion when team rewards get very large
        final_rewards = self._normalize_rewards(final_rewards)
        
        # Update last actions for rotation penalty tracking
        for i, action in enumerate(actions):
            self.last_actions[i] = action

        # Check termination
        done, termination_reason = self._check_done()
        
        # Calculate and apply completion bonus
        completion_bonus = self._calculate_completion_bonus(
            self.state.step_count, 
            termination_reason
        )
        
        if completion_bonus > 0:
            # Apply completion bonus to all agents equally
            final_rewards = [r + completion_bonus for r in final_rewards]

        # Info dictionary
        info = {
            'team_coverage_gain': team_coverage_gain,
            'individual_coverage_gains': coverage_gains,
            'knowledge_gains': knowledge_gains,
            'collisions': collision_info['collisions'],
            'agent_collisions': collision_info['agent_collisions'],
            'coverage_pct': self._get_coverage_percentage(),
            'steps': self.state.step_count,
            'coordination': self.coordination.value,
            'termination_reason': termination_reason,
            'completion_bonus': completion_bonus
        }

        return self.state, final_rewards, done, info

    def _generate_start_positions(
        self,
        obstacles: Set[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        """
        Generate spatially distributed start positions for agents.

        Uses grid-based placement to ensure agents start spread out.
        """
        positions = []

        # Divide grid into regions (rough grid)
        grid_div = int(math.ceil(math.sqrt(self.num_agents)))
        region_size = self.grid_size // grid_div

        # Attempt to place one agent per region
        for i in range(self.num_agents):
            region_x = (i % grid_div) * region_size
            region_y = (i // grid_div) * region_size

            # Search for valid position in this region
            max_attempts = 100
            for _ in range(max_attempts):
                x = region_x + random.randint(0, region_size - 1)
                y = region_y + random.randint(0, region_size - 1)

                # Clamp to grid
                x = max(1, min(x, self.grid_size - 2))
                y = max(1, min(y, self.grid_size - 2))

                pos = (x, y)

                # Check valid (not obstacle, not occupied)
                if pos not in obstacles and pos not in positions:
                    positions.append(pos)
                    break
            else:
                # Fallback: find any valid position
                pos = self._find_valid_position(obstacles, positions)
                positions.append(pos)

        return positions

    def _find_valid_position(
        self,
        obstacles: Set[Tuple[int, int]],
        occupied: List[Tuple[int, int]]
    ) -> Tuple[int, int]:
        """Find a valid position (not obstacle, not occupied)."""
        for _ in range(1000):
            x = random.randint(1, self.grid_size - 2)
            y = random.randint(1, self.grid_size - 2)
            pos = (x, y)

            if pos not in obstacles and pos not in occupied:
                return pos

        # Last resort: search systematically
        for x in range(1, self.grid_size - 1):
            for y in range(1, self.grid_size - 1):
                pos = (x, y)
                if pos not in obstacles and pos not in occupied:
                    return pos

        # Emergency fallback
        return (self.grid_size // 2, self.grid_size // 2)

    def _execute_actions_parallel(
        self,
        actions: List[int]
    ) -> Dict:
        """
        Execute all agent actions simultaneously.

        Detects:
        - Obstacle collisions
        - Boundary collisions
        - Agent-agent collisions

        Returns:
            collision_info: Dict with 'collisions' and 'agent_collisions'
        """
        # Calculate intended positions
        intended_positions = []
        for i, agent in enumerate(self.state.agents):
            action = actions[i]
            dx, dy = config.ACTION_DELTAS[action]

            current_pos = agent.robot_state.position
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy
            intended_pos = (new_x, new_y)

            intended_positions.append(intended_pos)

        # Check collisions
        collisions = []
        agent_collisions = []

        for i, agent in enumerate(self.state.agents):
            action = actions[i]
            intended_pos = intended_positions[i]
            current_pos = agent.robot_state.position

            # Update last action
            agent.robot_state.last_action = action

            collision = False
            agent_collision = False

            # Check boundary
            if not (0 <= intended_pos[0] < self.grid_size and
                    0 <= intended_pos[1] < self.grid_size):
                collision = True

            # Check obstacle
            elif intended_pos in self.state.world_state.obstacles:
                collision = True

            # Check agent-agent collision
            else:
                for j, other_intended_pos in enumerate(intended_positions):
                    if i != j and intended_pos == other_intended_pos:
                        collision = True
                        agent_collision = True
                        break

            # Execute or block movement
            if not collision:
                # Valid move
                agent.robot_state.position = intended_pos

                # Update orientation
                dx, dy = config.ACTION_DELTAS[action]
                if dx != 0 or dy != 0:
                    agent.robot_state.orientation = math.atan2(dy, dx)

                # Mark as visited
                agent.robot_state.visited_positions.add(intended_pos)
                agent.robot_state.visit_heat[intended_pos[0], intended_pos[1]] += 1

                # Update coverage map based on distance from agent
                if config.USE_PROBABILISTIC_ENV:
                    # Probabilistic coverage: distance-based sensor model
                    # P_cov(cell | robot_pos) = 1 / (1 + e^(k*(r - r0)))
                    distance = 0.0  # Agent is at this position
                    
                    # Sigmoid parameters
                    r0 = config.PROBABILISTIC_COVERAGE_MIDPOINT
                    k = config.PROBABILISTIC_COVERAGE_STEEPNESS
                    
                    # Coverage probability at agent position (distance=0)
                    p_cov = 1.0 / (1.0 + np.exp(k * (distance - r0)))
                    
                    # Update coverage (take maximum)
                    new_coverage = max(self.state.world_state.coverage_map[intended_pos[0], intended_pos[1]], p_cov)
                    self.state.world_state.coverage_map[intended_pos[0], intended_pos[1]] = new_coverage
                    agent.robot_state.coverage_history[intended_pos[0], intended_pos[1]] = new_coverage
                else:
                    # Binary coverage: instant 100%
                    self.state.world_state.coverage_map[intended_pos[0], intended_pos[1]] = 1.0
                    agent.robot_state.coverage_history[intended_pos[0], intended_pos[1]] = 1.0

            collisions.append(collision)
            agent_collisions.append(agent_collision)

        return {
            'collisions': collisions,
            'agent_collisions': agent_collisions
        }

    def _update_agent_sensing(self, agent: AgentState):
        """Update agent's local map via ray-cast sensing."""
        sensed_cells = self._raycast_sensing(
            agent.robot_state.position,
            agent.robot_state.orientation
        )

        # Update local map and coverage
        for cell in sensed_cells:
            if cell in self.state.world_state.obstacles:
                agent.robot_state.local_map[cell] = (0.0, "obstacle")
            else:
                # Update coverage based on distance (probabilistic mode)
                if config.USE_PROBABILISTIC_ENV:
                    # Distance-based coverage probability
                    distance = np.sqrt((cell[0] - agent.robot_state.position[0])**2 + 
                                     (cell[1] - agent.robot_state.position[1])**2)
                    
                    r0 = config.PROBABILISTIC_COVERAGE_MIDPOINT
                    k = config.PROBABILISTIC_COVERAGE_STEEPNESS
                    
                    # Coverage probability decreases with distance
                    p_cov = 1.0 / (1.0 + np.exp(k * (distance - r0)))
                    
                    # Update coverage (take maximum of current and new)
                    current_cov = self.state.world_state.coverage_map[cell[0], cell[1]]
                    new_coverage = max(current_cov, p_cov)
                    self.state.world_state.coverage_map[cell[0], cell[1]] = new_coverage
                    agent.robot_state.coverage_history[cell[0], cell[1]] = new_coverage
                    
                    # Update local map with new coverage
                    agent.robot_state.local_map[cell] = (new_coverage, "free")
                else:
                    # Binary mode: instant 100% coverage for sensed cells
                    self.state.world_state.coverage_map[cell[0], cell[1]] = 1.0
                    agent.robot_state.coverage_history[cell[0], cell[1]] = 1.0
                    agent.robot_state.local_map[cell] = (1.0, "free")

    def _raycast_sensing(
        self,
        position: Tuple[int, int],
        orientation: float
    ) -> Set[Tuple[int, int]]:
        """Ray-cast sensing (reuse from base environment)."""
        sensed = set()
        px, py = position

        sensed.add(position)

        # Vectorized ray-casting
        angles = np.linspace(0, 2 * np.pi, config.NUM_RAYS, endpoint=False)
        cos_angles = np.cos(angles)
        sin_angles = np.sin(angles)

        radii = np.linspace(0, self.sensor_range, config.SAMPLES_PER_RAY)[1:]

        for i in range(config.NUM_RAYS):
            cos_a = cos_angles[i]
            sin_a = sin_angles[i]

            for r in radii:
                cx = int(round(px + r * cos_a))
                cy = int(round(py + r * sin_a))

                if not (0 <= cx < self.grid_size and 0 <= cy < self.grid_size):
                    break

                cell = (cx, cy)
                sensed.add(cell)

                if cell in self.state.world_state.obstacles:
                    break

        return sensed

    def _calculate_agent_coverage_gain(
        self,
        agent: AgentState,
        prev_coverage_map: np.ndarray
    ) -> int:
        """Calculate newly covered cells by this agent."""
        # This is approximate - in multi-agent, multiple agents may cover same cell
        # We attribute coverage to all agents who sensed it
        current_coverage = self.state.world_state.coverage_map
        newly_covered = np.sum((current_coverage >= config.COVERAGE_THRESHOLD) & 
                              (prev_coverage_map < config.COVERAGE_THRESHOLD))

        # Divide by number of agents (approximate attribution)
        return int(newly_covered / self.num_agents)

    def _calculate_total_coverage_gain(self, prev_coverage_map: np.ndarray) -> int:
        """Calculate total newly covered cells (team metric)."""
        current_coverage = self.state.world_state.coverage_map
        newly_covered = np.sum((current_coverage >= config.COVERAGE_THRESHOLD) &
                              (prev_coverage_map < config.COVERAGE_THRESHOLD))
        return int(newly_covered)

    def _count_overlapping_cells(self, agent_id: int) -> int:
        """
        Count cells this agent has visited that other agents have also visited.

        FIXED: Only count each overlapping cell ONCE, not once per other agent.
        Previously with 4 agents visiting same cell: counted as 3 overlaps (wrong!)
        Now with 4 agents visiting same cell: counted as 1 overlap (correct!)

        Args:
            agent_id: ID of the agent to check

        Returns:
            Number of overlapping cells (counted once each)
        """
        from multi_agent_config import ma_config

        if not ma_config.USE_OVERLAP_PENALTY:
            return 0

        agent = self.state.agents[agent_id]
        agent_visits = agent.robot_state.coverage_history >= config.COVERAGE_THRESHOLD

        # FIXED: Use logical OR to find any cell visited by this agent AND any other agent
        # This counts each overlapping cell only ONCE instead of once per other agent
        any_other_visits = np.zeros_like(agent_visits, dtype=bool)
        for other in self.state.agents:
            if other.agent_id != agent_id:
                other_visits = other.robot_state.coverage_history >= config.COVERAGE_THRESHOLD
                any_other_visits = np.logical_or(any_other_visits, other_visits)

        # Count cells visited by this agent that were also visited by at least one other agent
        overlap = np.logical_and(agent_visits, any_other_visits)
        overlap_count = np.sum(overlap)

        return int(overlap_count)

    def _get_min_distance_to_other_agents(self, agent_id: int) -> float:
        """
        Get minimum Manhattan distance to nearest other agent.

        Args:
            agent_id: ID of the agent

        Returns:
            Minimum distance to nearest agent (in grid cells)
        """
        from multi_agent_config import ma_config

        if not ma_config.USE_DIVERSITY_BONUS:
            return 0.0

        agent = self.state.agents[agent_id]
        pos = agent.robot_state.position

        min_dist = float('inf')
        for other in self.state.agents:
            if other.agent_id != agent_id:
                other_pos = other.robot_state.position
                dist = abs(pos[0] - other_pos[0]) + abs(pos[1] - other_pos[1])
                min_dist = min(min_dist, dist)

        return min_dist if min_dist != float('inf') else 0.0

    def _get_agent_efficiency(self, agent_id: int) -> float:
        """
        Calculate agent's exploration efficiency (unique_cells / total_visits).

        Args:
            agent_id: ID of the agent

        Returns:
            Efficiency ratio in [0, 1]
        """
        from multi_agent_config import ma_config

        if not ma_config.USE_EFFICIENCY_BONUS:
            return 0.0

        agent = self.state.agents[agent_id]

        # Count unique cells visited
        unique_cells = np.sum(agent.robot_state.coverage_history >= config.COVERAGE_THRESHOLD)

        # Count total visits
        total_visits = np.sum(agent.robot_state.visit_heat)

        if total_visits == 0:
            return 0.0

        return unique_cells / total_visits

    def _compute_rotation_penalty(self, agent_id: int, current_action: int) -> float:
        """
        Compute rotation penalty for specific agent.
        
        Args:
            agent_id: ID of the agent
            current_action: Current action [0-8]
        
        Returns:
            Negative reward proportional to rotation angle (0 to -0.15)
        """
        if not config.USE_ROTATION_PENALTY:
            return 0.0
        
        # No penalty on first move or after STAY
        if self.last_actions[agent_id] is None or self.last_actions[agent_id] == 8:
            return 0.0
        
        # STAY action has no rotation
        if current_action == 8:
            return 0.0
        
        # Get angles for both actions
        last_angle = self.action_angles[self.last_actions[agent_id]]
        current_angle = self.action_angles[current_action]
        
        # Calculate minimum rotation (accounting for 360° wrap-around)
        angle_diff = abs(current_angle - last_angle)
        angle_diff = min(angle_diff, 360 - angle_diff)
        
        # Apply graduated penalties based on rotation magnitude
        if angle_diff == 0:
            return 0.0  # No rotation
        elif angle_diff <= 45:
            return config.ROTATION_PENALTY_SMALL   # -0.05
        elif angle_diff <= 90:
            return config.ROTATION_PENALTY_MEDIUM  # -0.10
        else:  # 135° or 180°
            return config.ROTATION_PENALTY_LARGE   # -0.15

    def _calculate_agent_reward(
        self,
        agent: AgentState,
        action: int,
        coverage_gain: int,
        knowledge_gain: int,
        collision: bool
    ) -> float:
        """Calculate individual agent reward with coordination components."""
        from multi_agent_config import ma_config

        reward = 0.0

        # === INDIVIDUAL REWARDS ===

        # Coverage reward (scale for probabilistic mode)
        coverage_reward_scale = config.COVERAGE_REWARD
        if config.USE_PROBABILISTIC_ENV:
            coverage_reward_scale *= config.PROBABILISTIC_REWARD_SCALE
        reward += coverage_gain * coverage_reward_scale

        # Exploration reward
        reward += knowledge_gain * config.EXPLORATION_REWARD

        # Rotation penalty (encourages smooth paths)
        rotation_penalty = self._compute_rotation_penalty(agent.agent_id, action)
        reward += rotation_penalty

        # Collision penalty (stronger for agent-agent collisions)
        if collision:
            reward += self.collision_penalty

        # Step penalty
        reward += config.STEP_PENALTY

        # Stay penalty
        if action == 8:
            reward += config.STAY_PENALTY

        # === COORDINATION REWARDS (NEW!) ===

        # Overlap penalty: Discourage redundant coverage
        if ma_config.USE_OVERLAP_PENALTY:
            overlap_count = self._count_overlapping_cells(agent.agent_id)
            overlap_penalty = overlap_count * ma_config.OVERLAP_PENALTY_SCALE
            reward -= overlap_penalty

        # Diversity bonus: Encourage spatial separation
        if ma_config.USE_DIVERSITY_BONUS:
            min_distance = self._get_min_distance_to_other_agents(agent.agent_id)
            # Normalize to [0, 1]: max useful distance is 10 cells
            diversity = min(min_distance / 10.0, 1.0)
            reward += diversity * ma_config.DIVERSITY_BONUS_SCALE

        # Efficiency bonus: Reward high unique_coverage / total_visits ratio
        if ma_config.USE_EFFICIENCY_BONUS:
            efficiency = self._get_agent_efficiency(agent.agent_id)
            reward += efficiency * ma_config.EFFICIENCY_BONUS_SCALE

        return reward

    def _initialize_coordination(self):
        """Initialize coordination strategy."""
        if self.coordination == CoordinationStrategy.INDEPENDENT:
            pass  # No coordination

        elif self.coordination == CoordinationStrategy.HIERARCHICAL:
            # Agent 0 is leader
            self.leader_id = 0

    def _update_coordination(self):
        """Update coordination strategy (called every step)."""
        if self.coordination == CoordinationStrategy.HIERARCHICAL:
            # Leader assigns tasks every N steps
            if self.state.step_count % 15 == 0:
                self._hierarchical_assignment()

    def _hierarchical_assignment(self):
        """Hierarchical task assignment (leader assigns tasks)."""
        if self.leader_id is None:
            return

        # Leader identifies frontiers
        frontier_cells = self._identify_frontier_cells()

        if len(frontier_cells) == 0:
            return

        # Leader assigns tasks to followers (round-robin)
        frontier_list = list(frontier_cells)

        for i, agent in enumerate(self.state.agents):
            if i == self.leader_id:
                continue  # Skip leader

            # Assign frontier
            if len(frontier_list) > 0:
                assigned_frontier = frontier_list[i % len(frontier_list)]
                agent.task_assignment = assigned_frontier

    def _identify_frontier_cells(self) -> Set[Tuple[int, int]]:
        """Identify frontier cells (team knowledge)."""
        # Merge all agent local maps
        team_knowledge = set()
        for agent in self.state.agents:
            team_knowledge.update(agent.robot_state.local_map.keys())

        # Find frontiers (known cells adjacent to unknown)
        frontiers = set()
        for cell in team_knowledge:
            x, y = cell
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                neighbor = (x + dx, y + dy)

                if (0 <= neighbor[0] < self.grid_size and
                    0 <= neighbor[1] < self.grid_size):

                    if neighbor not in team_knowledge:
                        frontiers.add(cell)
                        break

        return frontiers

    def _check_done(self) -> Tuple[bool, str]:
        """
        Check if episode should terminate.
        
        Returns:
            done: Whether episode is complete
            reason: Termination reason ('max_steps', 'early_completion', 'high_coverage', or 'incomplete')
        """
        # Max steps reached
        if self.state.step_count >= self.max_steps:
            return True, 'max_steps'

        # Early termination (multi-agent only)
        if config.ENABLE_EARLY_TERMINATION_MULTI:
            coverage_pct = self._get_coverage_percentage()
            
            # Check minimum steps constraint first
            if self.state.step_count >= config.EARLY_TERM_MIN_STEPS_MULTI:
                # Check if coverage target reached
                if coverage_pct >= config.EARLY_TERM_COVERAGE_TARGET_MULTI:
                    return True, 'early_completion'
                
                # Legacy high coverage termination (kept for backwards compatibility)
                if coverage_pct > 0.95:
                    return True, 'high_coverage'
        else:
            # Without early termination, only check legacy high coverage
            coverage_pct = self._get_coverage_percentage()
            if coverage_pct > 0.95:
                return True, 'high_coverage'

        return False, 'incomplete'
    
    def _calculate_completion_bonus(self, steps_used: int, termination_reason: str) -> float:
        """
        Calculate completion bonus for early termination.
        
        Encourages agents to complete coverage efficiently by rewarding
        earlier completions with larger bonuses.
        
        Args:
            steps_used: Number of steps taken in episode
            termination_reason: Why episode terminated
            
        Returns:
            bonus: Completion bonus (0 if no early completion)
        """
        if termination_reason != 'early_completion':
            return 0.0
        
        # Calculate steps saved
        steps_saved = self.max_steps - steps_used
        
        # Flat bonus + per-step bonus
        flat_bonus = config.EARLY_TERM_COMPLETION_BONUS
        time_bonus = steps_saved * config.EARLY_TERM_TIME_BONUS_PER_STEP
        
        total_bonus = flat_bonus + time_bonus
        
        return total_bonus
    
    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """
        Normalize rewards for QMIX stability.
        
        Prevents gradient explosion when team rewards scale with number of agents.
        Uses hybrid approach from QMIX (Rashid et al., 2018) and R2D2 (Pohlen et al., 2018).
        
        Args:
            rewards: Raw rewards per agent
            
        Returns:
            normalized_rewards: Scaled rewards in manageable range
        """
        if not config.MULTI_AGENT_REWARD_NORMALIZE_BY_N:
            # No normalization - use raw rewards
            normalized = rewards
        else:
            # Step 1: Normalize by number of agents
            # This keeps total team reward same scale as single-agent
            normalized = [r / self.num_agents for r in rewards]
        
        # Step 2: Apply scale factor
        # Maps typical per-step rewards (0-20) to smaller range (0-2)
        if config.MULTI_AGENT_REWARD_SCALE_FACTOR != 1.0:
            normalized = [r / config.MULTI_AGENT_REWARD_SCALE_FACTOR for r in normalized]
        
        # Step 3: Optional clipping
        if config.MULTI_AGENT_REWARD_CLIP_MIN is not None or config.MULTI_AGENT_REWARD_CLIP_MAX is not None:
            clip_min = config.MULTI_AGENT_REWARD_CLIP_MIN if config.MULTI_AGENT_REWARD_CLIP_MIN is not None else -float('inf')
            clip_max = config.MULTI_AGENT_REWARD_CLIP_MAX if config.MULTI_AGENT_REWARD_CLIP_MAX is not None else float('inf')
            normalized = [np.clip(r, clip_min, clip_max) for r in normalized]
        
        # Step 4: Optional value rescaling (R2D2 style)
        if config.MULTI_AGENT_USE_VALUE_RESCALING:
            eps = config.MULTI_AGENT_VALUE_RESCALE_EPS
            normalized = [
                np.sign(r) * (np.sqrt(abs(r) + 1) - 1) + eps * r
                for r in normalized
            ]
        
        return normalized

    def _get_coverage_percentage(self) -> float:
        """Calculate coverage percentage using threshold from config."""
        total_free_cells = (
            self.grid_size * self.grid_size -
            len(self.state.world_state.obstacles)
        )

        if total_free_cells == 0:
            return 0.0

        covered_cells = np.sum(self.state.world_state.coverage_map >= config.COVERAGE_THRESHOLD)
        return covered_cells / total_free_cells

    def get_global_state(self) -> np.ndarray:
        """
        Get global state for QMIX mixing network.

        Global state includes team-level information that is not available to
        individual agents but can be used during centralized training:
        - Team coverage map (1600 values for 40x40)
        - All agent positions (8 values for 4 agents)
        - Total coverage percentage (1 value)
        - Episode progress (1 value)

        Total: 1610 dimensions

        This enables CTDE (Centralized Training, Decentralized Execution):
        - Training: Use global state for mixing network to optimize team strategy
        - Execution: Each agent acts based on local observations only

        Returns:
            global_state: [1610] array with team-level information
        """
        # Coverage map (flattened)
        # Shape: [grid_size * grid_size] = [1600]
        coverage_map = self.state.world_state.coverage_map.flatten()

        # All agent positions (normalized to [0, 1])
        # Shape: [num_agents * 2] = [8] for 4 agents
        positions = []
        for agent in self.state.agents:
            positions.extend([
                agent.robot_state.position[0] / self.grid_size,  # normalized x
                agent.robot_state.position[1] / self.grid_size   # normalized y
            ])
        positions = np.array(positions, dtype=np.float32)

        # Coverage percentage
        # Shape: [1]
        coverage_pct = np.array([self._get_coverage_percentage()], dtype=np.float32)

        # Episode progress (normalized to [0, 1])
        # Shape: [1]
        progress = np.array([self.state.step_count / self.max_steps], dtype=np.float32)

        # Concatenate all components
        global_state = np.concatenate([
            coverage_map,
            positions,
            coverage_pct,
            progress
        ])

        return global_state

    def get_observations(self) -> List[Dict]:
        """
        Get POMDP observations for all agents.

        Returns:
            observations: List of observation dicts (one per agent)
        """
        observations = []

        for agent in self.state.agents:
            obs = {
                'robot_state': agent.robot_state,
                'world_state': self.state.world_state,
                'agent_id': agent.agent_id,
                'nearby_agents': self.state.get_nearby_agents(agent.agent_id),
                'assigned_region': agent.assigned_region,
                'task_assignment': agent.task_assignment
            }
            observations.append(obs)

        return observations

    def render(self):
        """Render environment (text-based)."""
        coverage_pct = self._get_coverage_percentage()

        print(f"\n=== Multi-Agent Coverage ({self.coordination.value}) ===")
        print(f"Step {self.state.step_count}/{self.max_steps}")
        print(f"Coverage: {coverage_pct*100:.1f}%")

        for i, agent in enumerate(self.state.agents):
            pos = agent.robot_state.position
            sensed = len(agent.robot_state.local_map)
            visited = len(agent.robot_state.visited_positions)
            print(f"  Agent {i}: pos={pos}, sensed={sensed}, visited={visited}")


if __name__ == "__main__":
    print("Testing MultiAgentCoverageEnv...")
    print("\n" + "="*80)
    print("MULTI-AGENT REWARD FUNCTION")
    print("="*80)
    print("""
The multi-agent reward function has several components:

1. JOINT COVERAGE REWARD (Primary objective)
   - Reward: +10.0 per NEW cell covered by ANY agent
   - Shared by ALL agents (cooperative)
   - Encourages teamwork and efficient exploration
   
   Example: If agent 1 covers 5 new cells and agent 2 covers 3 new cells,
            ALL agents receive reward = (5 + 3) * 10 = 80

2. COLLISION PENALTY
   - Penalty: -2.0 for agent-agent collision
   - Applied ONLY to colliding agents
   - Encourages collision avoidance
   
   Example: Agent 1 and Agent 2 try to move to same cell
            → Both get -2.0 penalty

3. SEPARATION INCENTIVE (Near-miss penalty)
   - Penalty: -0.5 for being adjacent to another agent
   - Encourages agents to spread out
   - Prevents clustering
   
   Example: Agent 1 at (5,5), Agent 2 at (5,6) (adjacent)
            → Both get -0.5 penalty

4. REDUNDANCY PENALTY (Optional)
   - Penalty: -0.1 per cell covered by multiple agents
   - Discourages overlapping coverage
   - Promotes efficient space partitioning
   
   Example: Both agents sense same 10 cells
            → Each gets -0.1 * 10 = -1.0 penalty

TOTAL REWARD per agent:
reward_i = joint_coverage_reward 
           + collision_penalty_i 
           + separation_penalty_i 
           + redundancy_penalty_i

REWARD MODES:
- 'joint': All agents share total coverage (fully cooperative)
- 'individual': Each agent only rewarded for own coverage
- 'mixed': 70% individual + 30% joint (balance)

COMMUNICATION (Full State Sharing within comm_range):
- Agents within comm_range share complete local_map
- Agents merge coverage_history from neighbors
- Enables coordinated exploration
- No bandwidth cost (perfect communication)

WHY THIS REWARD STRUCTURE?
1. Joint reward → cooperation (agents work together)
2. Collision penalty → safety (avoid crashes)
3. Separation incentive → efficiency (don't cluster)
4. Redundancy penalty → optimal coverage (minimize overlap)

This creates emergent behavior:
- Agents spread out to maximize coverage
- Agents avoid each other to prevent collisions
- Agents communicate to share knowledge
- Team performance >> individual agents
""")
    print("="*80)

    # Test independent strategy
    env = MultiAgentCoverageEnv(
        num_agents=4,
        grid_size=20,
        coordination=CoordinationStrategy.INDEPENDENT
    )

    state = env.reset()
    print(f"\n✓ Reset complete")
    print(f"  Num agents: {state.num_agents}")
    print(f"  Agent positions: {state.get_agent_positions()}")
    print(f"  Coordination: {state.coordination.value}")

    # Test episode
    total_rewards = [0.0] * env.num_agents
    for step in range(10):
        actions = [random.randint(0, 8) for _ in range(env.num_agents)]
        next_state, rewards, done, info = env.step(actions)

        for i in range(env.num_agents):
            total_rewards[i] += rewards[i]

        if step == 0:
            print(f"\n✓ First step complete")
            print(f"  Actions: {actions}")
            print(f"  Rewards: {[f'{r:.2f}' for r in rewards]}")
            print(f"  Team coverage gain: {info['team_coverage_gain']}")
            print(f"  Agent collisions: {info['agent_collisions']}")

        if done:
            print(f"\n✓ Episode terminated at step {step}")
            break

    print(f"\n✓ MultiAgentCoverageEnv test complete")
    print(f"  Total rewards: {[f'{r:.2f}' for r in total_rewards]}")
    print(f"  Final coverage: {env._get_coverage_percentage()*100:.1f}%")
