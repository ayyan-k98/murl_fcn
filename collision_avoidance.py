"""
Collision Avoidance for Multi-Agent Systems

Implements several collision avoidance strategies:
1. Reactive avoidance (filter unsafe actions)
2. Predictive avoidance (anticipate future collisions)
3. Learned avoidance (through reward shaping)
4. Communication-based (agents share intentions)

Collision Types:
- Agent-Agent: Two agents try to occupy same cell
- Agent-Obstacle: Agent tries to move into obstacle
- Agent-Wall: Agent tries to move outside grid
"""

import torch
import numpy as np
from typing import List, Tuple, Set, Dict, Optional
from dataclasses import dataclass

from config import config


@dataclass
class CollisionEvent:
    """Record of a collision."""
    step: int
    agent_ids: Tuple[int, int]  # Which agents collided
    position: Tuple[int, int]    # Where collision occurred
    collision_type: str          # 'agent-agent', 'agent-obstacle', 'agent-wall'


class CollisionDetector:
    """
    Detects potential collisions before they happen.
    
    Supports:
    - Static collision detection (current positions)
    - Dynamic collision detection (predicted positions)
    - Multi-step lookahead
    """
    
    def __init__(self, grid_size: int, num_agents: int):
        self.grid_size = grid_size
        self.num_agents = num_agents
        
        # Collision history (for analysis)
        self.collision_history: List[CollisionEvent] = []
        self.total_collisions = 0
    
    def check_agent_agent_collision(self, 
                                    positions: List[Tuple[int, int]]) -> List[bool]:
        """
        Check if any agents occupy same position.
        
        Args:
            positions: List of (x, y) positions for each agent
        
        Returns:
            collisions: [num_agents] bool array (True if agent collides)
        """
        collisions = [False] * self.num_agents
        
        # Check pairwise collisions
        for i in range(self.num_agents):
            for j in range(i+1, self.num_agents):
                if positions[i] == positions[j]:
                    collisions[i] = True
                    collisions[j] = True
                    
                    self.collision_history.append(CollisionEvent(
                        step=len(self.collision_history),
                        agent_ids=(i, j),
                        position=positions[i],
                        collision_type='agent-agent'
                    ))
                    self.total_collisions += 1
        
        return collisions
    
    def check_wall_collision(self, position: Tuple[int, int]) -> bool:
        """Check if position is outside grid bounds."""
        x, y = position
        return x < 0 or x >= self.grid_size or y < 0 or y >= self.grid_size
    
    def check_obstacle_collision(self, position: Tuple[int, int],
                                 obstacles: Set[Tuple[int, int]]) -> bool:
        """Check if position intersects obstacle."""
        return position in obstacles
    
    def predict_collision(self,
                         current_positions: List[Tuple[int, int]],
                         actions: List[int],
                         obstacles: Set[Tuple[int, int]]) -> List[bool]:
        """
        Predict if actions will cause collisions.
        
        Args:
            current_positions: Current agent positions
            actions: Proposed actions
            obstacles: Set of obstacle positions
        
        Returns:
            will_collide: [num_agents] bool (True if action causes collision)
        """
        # Compute next positions
        next_positions = []
        for pos, action in zip(current_positions, actions):
            next_pos = self._apply_action(pos, action)
            next_positions.append(next_pos)
        
        # Check collisions
        will_collide = [False] * self.num_agents
        
        for i in range(self.num_agents):
            # Wall collision
            if self.check_wall_collision(next_positions[i]):
                will_collide[i] = True
            
            # Obstacle collision
            if self.check_obstacle_collision(next_positions[i], obstacles):
                will_collide[i] = True
            
            # Agent-agent collision
            for j in range(self.num_agents):
                if i != j and next_positions[i] == next_positions[j]:
                    will_collide[i] = True
                    break
        
        return will_collide
    
    def _apply_action(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Apply action to get next position."""
        dx, dy = config.ACTION_DELTAS[action]
        return (position[0] + dx, position[1] + dy)
    
    def get_collision_rate(self) -> float:
        """Get collision rate (collisions per step)."""
        if len(self.collision_history) == 0:
            return 0.0
        return self.total_collisions / len(self.collision_history)
    
    def reset(self):
        """Reset collision history."""
        self.collision_history = []
        self.total_collisions = 0


class CollisionAvoider:
    """
    Filters actions to avoid collisions.
    
    Strategies:
    1. Remove unsafe actions from action space
    2. Assign priorities to agents (higher priority moves first)
    3. Replan if collision detected
    """
    
    def __init__(self, grid_size: int, num_agents: int, 
                 strategy: str = 'filter'):
        self.grid_size = grid_size
        self.num_agents = num_agents
        self.strategy = strategy
        
        self.detector = CollisionDetector(grid_size, num_agents)
        
        # Agent priorities (for conflict resolution)
        self.priorities = list(range(num_agents))  # Default: agent 0 has highest priority
    
    def filter_actions(self,
                      agent_id: int,
                      current_positions: List[Tuple[int, int]],
                      obstacles: Set[Tuple[int, int]],
                      other_actions: Optional[List[int]] = None) -> List[int]:
        """
        Get list of safe actions for agent.
        
        Args:
            agent_id: Agent to filter actions for
            current_positions: All agents' current positions
            obstacles: Obstacle positions
            other_actions: Actions already chosen by other agents (for sequential decision)
        
        Returns:
            valid_actions: List of action indices that are safe
        """
        valid_actions = []
        
        for action in range(config.N_ACTIONS):
            # Compute next position
            next_pos = self._apply_action(current_positions[agent_id], action)
            
            # Check wall collision
            if self.detector.check_wall_collision(next_pos):
                continue
            
            # Check obstacle collision
            if self.detector.check_obstacle_collision(next_pos, obstacles):
                continue
            
            # Check agent-agent collision
            collision = False
            for other_id in range(self.num_agents):
                if other_id == agent_id:
                    continue
                
                # If other agent's action is known, check collision
                if other_actions and other_actions[other_id] is not None:
                    other_next_pos = self._apply_action(
                        current_positions[other_id], 
                        other_actions[other_id]
                    )
                    if next_pos == other_next_pos:
                        collision = True
                        break
                else:
                    # Other agent's action unknown, check if next_pos is other's current pos
                    if next_pos == current_positions[other_id]:
                        collision = True
                        break
            
            if not collision:
                valid_actions.append(action)
        
        # If no valid actions, allow STAY action (safest)
        if len(valid_actions) == 0:
            stay_action = 8  # Assuming STAY is action 8
            if stay_action < config.N_ACTIONS:
                valid_actions.append(stay_action)
        
        return valid_actions
    
    def sequential_action_selection(self,
                                   current_positions: List[Tuple[int, int]],
                                   obstacles: Set[Tuple[int, int]],
                                   action_selector) -> List[int]:
        """
        Select actions sequentially by priority to avoid collisions.
        
        Args:
            current_positions: All agents' positions
            obstacles: Obstacle set
            action_selector: Function that selects action given valid actions
                           Signature: action_selector(agent_id, valid_actions) -> action
        
        Returns:
            actions: List of selected actions
        """
        actions = [None] * self.num_agents
        
        # Select actions in priority order
        for agent_id in self.priorities:
            # Get valid actions (considering already-selected actions)
            valid_actions = self.filter_actions(
                agent_id,
                current_positions,
                obstacles,
                other_actions=actions
            )
            
            # Select action
            action = action_selector(agent_id, valid_actions)
            actions[agent_id] = action
        
        return actions
    
    def resolve_collision(self,
                         actions: List[int],
                         current_positions: List[Tuple[int, int]],
                         obstacles: Set[Tuple[int, int]]) -> List[int]:
        """
        Resolve collisions in proposed actions.
        
        If collision detected, lower-priority agent changes action.
        
        Args:
            actions: Proposed actions
            current_positions: Current positions
            obstacles: Obstacles
        
        Returns:
            resolved_actions: Collision-free actions
        """
        resolved = actions.copy()
        
        # Predict collisions
        will_collide = self.detector.predict_collision(
            current_positions, actions, obstacles
        )
        
        # Resolve by priority
        for agent_id in reversed(self.priorities):  # Lower priority first
            if will_collide[agent_id]:
                # Find alternative action
                valid_actions = self.filter_actions(
                    agent_id,
                    current_positions,
                    obstacles,
                    other_actions=resolved
                )
                
                if len(valid_actions) > 0:
                    # Choose safest action (e.g., STAY)
                    stay_action = 8 if 8 < config.N_ACTIONS else valid_actions[0]
                    if stay_action in valid_actions:
                        resolved[agent_id] = stay_action
                    else:
                        resolved[agent_id] = valid_actions[0]
        
        return resolved
    
    def _apply_action(self, position: Tuple[int, int], action: int) -> Tuple[int, int]:
        """Apply action to position."""
        dx, dy = config.ACTION_DELTAS[action]
        return (position[0] + dx, position[1] + dy)
    
    def set_priorities(self, priorities: List[int]):
        """Set agent priorities (for conflict resolution)."""
        assert len(priorities) == self.num_agents
        assert set(priorities) == set(range(self.num_agents))
        self.priorities = priorities
    
    def randomize_priorities(self):
        """Randomize priorities (for fairness)."""
        import random
        self.priorities = list(range(self.num_agents))
        random.shuffle(self.priorities)


class CollisionRewardShaper:
    """
    Shapes rewards to encourage collision avoidance.
    
    Reward components:
    - Collision penalty (immediate)
    - Near-miss penalty (getting close to other agents)
    - Separation bonus (maintaining safe distance)
    """
    
    def __init__(self, collision_penalty: float = -2.0,
                 near_miss_penalty: float = -0.5,
                 safe_distance: int = 2):
        self.collision_penalty = collision_penalty
        self.near_miss_penalty = near_miss_penalty
        self.safe_distance = safe_distance
    
    def compute_collision_reward(self,
                                 agent_id: int,
                                 position: Tuple[int, int],
                                 other_positions: List[Tuple[int, int]],
                                 collided: bool) -> float:
        """
        Compute collision-related reward.
        
        Args:
            agent_id: Agent ID
            position: Agent's position
            other_positions: Other agents' positions
            collided: Whether collision occurred
        
        Returns:
            reward: Collision-related reward
        """
        reward = 0.0
        
        # Direct collision penalty
        if collided:
            reward += self.collision_penalty
        
        # Near-miss penalty (getting too close)
        for i, other_pos in enumerate(other_positions):
            if i == agent_id:
                continue
            
            distance = abs(position[0] - other_pos[0]) + abs(position[1] - other_pos[1])
            
            if distance == 1:  # Adjacent
                reward += self.near_miss_penalty
            elif distance == 0:  # Same position (collision)
                # Already handled above
                pass
        
        return reward
    
    def compute_separation_bonus(self,
                                positions: List[Tuple[int, int]]) -> float:
        """
        Bonus for maintaining good separation.
        
        Encourages agents to spread out.
        """
        # Compute pairwise distances
        distances = []
        num_agents = len(positions)
        
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                dist = abs(positions[i][0] - positions[j][0]) + \
                       abs(positions[i][1] - positions[j][1])
                distances.append(dist)
        
        if len(distances) == 0:
            return 0.0
        
        # Bonus if average distance is above threshold
        avg_distance = np.mean(distances)
        
        if avg_distance >= self.safe_distance:
            return 0.1  # Small bonus for good separation
        else:
            return 0.0


def get_collision_avoidance_strategy(strategy: str, 
                                     grid_size: int,
                                     num_agents: int):
    """
    Factory function to create collision avoidance strategy.
    
    Args:
        strategy: 'none', 'filter', 'sequential', 'resolve'
        grid_size: Grid size
        num_agents: Number of agents
    
    Returns:
        CollisionAvoider instance or None
    """
    if strategy == 'none':
        return None
    elif strategy in ['filter', 'sequential', 'resolve']:
        return CollisionAvoider(grid_size, num_agents, strategy)
    else:
        raise ValueError(f"Unknown collision avoidance strategy: {strategy}")
