"""
Agent occupancy probability channel computation.

Computes P(other agent at cell (x,y)) based on:
1. Last communicated positions
2. Time decay (uncertainty grows with time)
3. Probabilistic union (multiple agents)

This provides a soft representation of where other agents are likely to be,
enabling proactive coordination and collision avoidance.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import torch


class AgentOccupancyComputer:
    """Compute agent occupancy probability channel for multi-agent coordination.
    
    The occupancy channel represents P(another agent occupies cell (x,y)) based on:
    - Communicated positions from other agents
    - Time since last communication (uncertainty grows)
    - Movement model (agents can move with max_velocity)
    
    The resulting probability map is used as the 6th input channel to the FCN,
    allowing agents to anticipate and avoid collisions proactively.
    """
    
    def __init__(self, 
                 grid_size: int,
                 base_sigma: float = 0.5,
                 max_velocity: float = 1.0,
                 time_decay_rate: float = 0.1):
        """
        Initialize agent occupancy computer.
        
        Args:
            grid_size: Size of the grid (assumes square grid)
            base_sigma: Base standard deviation for Gaussian (in grid cells)
            max_velocity: Maximum agent velocity (cells per timestep)
            time_decay_rate: Rate at which uncertainty grows with time
        """
        self.grid_size = grid_size
        self.base_sigma = base_sigma
        self.max_velocity = max_velocity
        self.time_decay_rate = time_decay_rate
        
        # Cache for Gaussian computation (optimization)
        self._gaussian_cache = {}
        
        # Pre-compute coordinate grids for vectorized operations
        self.y_grid, self.x_grid = np.meshgrid(
            np.arange(grid_size), 
            np.arange(grid_size),
            indexing='ij'
        )
    
    def compute(self,
                agent_id: int,
                messages: List[Dict],
                current_time: int) -> np.ndarray:
        """
        Compute occupancy probability map for OTHER agents.
        
        Args:
            agent_id: ID of agent computing occupancy (exclude self)
            messages: List of recent messages from other agents
                     Each message should have:
                     - 'sender_id': int
                     - 'position': (x, y) tuple or [x, y] array
                     - 'timestamp': int (optional, defaults to current_time)
            current_time: Current timestep
            
        Returns:
            occupancy: [H, W] array of probabilities in [0, 1]
                      Higher values = more likely another agent is there
        """
        if not messages:
            # No messages = no other agents known
            return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        occupancy = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        for msg in messages:
            # Skip messages from self
            if msg.get('sender_id') == agent_id:
                continue
            
            # Extract position
            position = msg.get('position')
            if position is None:
                continue
            
            # Handle both tuple and array formats
            if isinstance(position, (list, np.ndarray, tuple)):
                x, y = float(position[0]), float(position[1])
            else:
                continue  # Invalid position format
            
            # Get timestamp (default to current if not provided)
            timestamp = msg.get('timestamp', current_time)
            delta_t = current_time - timestamp
            
            # Compute uncertainty that grows with time
            # sigma = base_sigma + velocity * time + decay_rate * time^2
            # Quadratic growth models increasing uncertainty over time
            sigma = (self.base_sigma + 
                    self.max_velocity * delta_t + 
                    self.time_decay_rate * delta_t**2)
            
            # Clamp sigma to reasonable range
            sigma = max(0.1, min(sigma, self.grid_size / 2))
            
            # Compute Gaussian probability distribution centered at position
            prob_map = self._gaussian_at((x, y), sigma)
            
            # Probabilistic union: P(A or B) = 1 - (1-P(A))(1-P(B))
            # This correctly handles multiple agents without double-counting
            occupancy = 1.0 - (1.0 - occupancy) * (1.0 - prob_map)
        
        return occupancy.astype(np.float32)
    
    def compute_batch(self,
                     agent_ids: List[int],
                     messages: List[Dict],
                     current_time: int) -> List[np.ndarray]:
        """
        Compute occupancy maps for multiple agents (batch version).
        
        Args:
            agent_ids: List of agent IDs
            messages: List of messages from all agents
            current_time: Current timestep
            
        Returns:
            occupancies: List of [H, W] occupancy maps, one per agent
        """
        return [
            self.compute(agent_id, messages, current_time)
            for agent_id in agent_ids
        ]
    
    def _gaussian_at(self, position: Tuple[float, float], 
                     sigma: float) -> np.ndarray:
        """
        Compute 2D Gaussian probability distribution centered at position.
        
        Uses vectorized operations for efficiency.
        
        Args:
            position: (x, y) center of Gaussian
            sigma: Standard deviation
            
        Returns:
            gaussian: [H, W] probability map
        """
        x, y = position
        
        # Vectorized distance computation
        dist_sq = (self.x_grid - x)**2 + (self.y_grid - y)**2
        
        # Gaussian formula: exp(-d^2 / (2*sigma^2))
        gaussian = np.exp(-dist_sq / (2 * sigma**2))
        
        # Normalize so max probability is 1.0 at center
        max_val = np.max(gaussian)
        if max_val > 0:
            gaussian = gaussian / max_val
        
        return gaussian
    
    def compute_from_positions(self,
                              agent_id: int,
                              positions: List[Tuple[int, Tuple[float, float]]],
                              current_time: int) -> np.ndarray:
        """
        Compute occupancy from simple position list (convenience method).
        
        Args:
            agent_id: ID of agent computing occupancy
            positions: List of (other_agent_id, (x, y)) tuples
            current_time: Current timestep
            
        Returns:
            occupancy: [H, W] probability map
        """
        # Convert positions to message format
        messages = [
            {
                'sender_id': other_id,
                'position': pos,
                'timestamp': current_time
            }
            for other_id, pos in positions
        ]
        
        return self.compute(agent_id, messages, current_time)
    
    def visualize(self, occupancy: np.ndarray) -> str:
        """
        Create ASCII visualization of occupancy map (for debugging).
        
        Args:
            occupancy: [H, W] occupancy map
            
        Returns:
            visualization: ASCII string representation
        """
        # Map probabilities to characters
        chars = ' .:-=+*#@'
        
        lines = []
        for row in occupancy:
            line = ''
            for prob in row:
                idx = min(int(prob * len(chars)), len(chars) - 1)
                line += chars[idx]
            lines.append(line)
        
        return '\n'.join(lines)


def create_dummy_occupancy(grid_size: int) -> np.ndarray:
    """
    Create dummy occupancy map (all zeros) for single-agent training.
    
    This allows using 6-channel networks in single-agent mode by providing
    a dummy 6th channel that is always zero.
    
    Args:
        grid_size: Size of grid
        
    Returns:
        dummy: [H, W] array of zeros
    """
    return np.zeros((grid_size, grid_size), dtype=np.float32)


if __name__ == "__main__":
    # Test the agent occupancy computer
    print("Testing AgentOccupancyComputer...")
    
    computer = AgentOccupancyComputer(grid_size=20)
    
    # Test case 1: Single agent at center
    messages = [
        {'sender_id': 1, 'position': (10.0, 10.0), 'timestamp': 0}
    ]
    occupancy = computer.compute(agent_id=0, messages=messages, current_time=0)
    
    print("\nTest 1: Single agent at center (10, 10)")
    print(f"Occupancy shape: {occupancy.shape}")
    print(f"Max probability: {occupancy.max():.3f}")
    print(f"Sum of probabilities: {occupancy.sum():.2f}")
    print("\nVisualization:")
    print(computer.visualize(occupancy))
    
    # Test case 2: Multiple agents
    messages = [
        {'sender_id': 1, 'position': (5.0, 5.0), 'timestamp': 0},
        {'sender_id': 2, 'position': (15.0, 15.0), 'timestamp': 0}
    ]
    occupancy = computer.compute(agent_id=0, messages=messages, current_time=0)
    
    print("\n\nTest 2: Two agents at (5,5) and (15,15)")
    print(f"Max probability: {occupancy.max():.3f}")
    print(f"Sum of probabilities: {occupancy.sum():.2f}")
    
    # Test case 3: Old message (uncertainty grows)
    messages = [
        {'sender_id': 1, 'position': (10.0, 10.0), 'timestamp': 0}
    ]
    occupancy_old = computer.compute(agent_id=0, messages=messages, current_time=5)
    
    print("\n\nTest 3: Old message (5 timesteps ago)")
    print(f"Max probability: {occupancy_old.max():.3f}")
    print(f"Sum of probabilities: {occupancy_old.sum():.2f}")
    print("(More spread out due to uncertainty)")
    
    # Test case 4: Self-filtering
    messages = [
        {'sender_id': 0, 'position': (10.0, 10.0), 'timestamp': 0},  # Self
        {'sender_id': 1, 'position': (15.0, 15.0), 'timestamp': 0}   # Other
    ]
    occupancy = computer.compute(agent_id=0, messages=messages, current_time=0)
    
    print("\n\nTest 4: Self-filtering (agent 0 ignores own message)")
    print(f"Max probability at (10,10): {occupancy[10, 10]:.3f} (should be low)")
    print(f"Max probability at (15,15): {occupancy[15, 15]:.3f} (should be high)")
    
    print("\n\nAll tests completed!")
