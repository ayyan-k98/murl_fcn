"""
Potential-Based Reward Shaping (PBRS) for Multi-Agent Coverage

This module implements optional potential-based reward shaping to accelerate learning.

WHY START WITHOUT PBRS (RECOMMENDED):
====================================

1. QMIX is Already Powerful
   - Monotonic value factorization handles credit assignment well
   - Proven to work on complex coordination tasks
   - Don't add complexity until you know you need it

2. Coverage Rewards Are Already Dense
   - Every new cell visited gives immediate +1 reward
   - N-step returns (n=3) provide credit assignment over short horizons
   - Unlike sparse tasks (e.g., "reach goal"), we get frequent feedback

3. Simpler Debugging
   - If training fails with PBRS, you won't know if it's:
     * Bad potential function design
     * QMIX implementation issues
     * Environment bugs
     * Hyperparameter problems
   - Validate the base system first!

4. Potential Functions Can Be Tricky
   - Wrong potential can actively hurt learning
   - Need domain knowledge to design good potentials
   - Requires tuning the shaping coefficient β

5. You Can Always Add It Later
   - If you see specific problems (clustering, slow exploration, etc.)
   - Then you know exactly what behavior to shape
   - Targeted fixes are better than premature optimization

WHEN TO ADD PBRS:
================
Only add PBRS if you observe these specific issues after training with base QMIX:

1. Slow Early Learning (episodes 0-200)
   - Agents wander aimlessly for many episodes
   - Coverage % stays near zero for too long
   → Use frontier_distance or expected_coverage_potential

2. Agent Clustering
   - All agents follow each other
   - Redundant coverage, low efficiency
   → Use coordination_potential with separation term

3. Ignoring Far Regions
   - Agents stay near spawn, don't explore full map
   - Coverage plateaus below 70-80%
   → Use frontier_distance with large β

4. Action Oscillation
   - Agents flip between actions frequently
   - High action entropy, unstable policies
   → Use smoother potential like local_coverage_potential

THEORY:
=======
PBRS adds F(s, a, s') = γ * Φ(s') - Φ(s) to the reward, where:
- Φ(s) is a potential function measuring "goodness" of state s
- γ is the discount factor
- F is the shaping reward

Key property: PBRS is guaranteed to preserve optimal policy if Φ is arbitrary
(Ng, Harada, & Russell, 1999: "Policy Invariance Under Reward Transformations")

However, it CAN change:
- Learning speed (good potentials accelerate)
- Sample efficiency
- Exploration patterns
- Convergence stability

"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.ndimage import distance_transform_edt
from config import config


class PotentialFunction:
    """Base class for potential functions."""
    
    def __call__(self, state: Dict, agent_positions: np.ndarray, 
                 coverage_map: np.ndarray) -> float:
        """
        Compute potential value Φ(s) for current state.
        
        Args:
            state: Full state dict from environment
            agent_positions: (num_agents, 2) array of positions
            coverage_map: (grid_size, grid_size) binary coverage map
            
        Returns:
            Potential value (higher = better state)
        """
        raise NotImplementedError


class FrontierDistancePotential(PotentialFunction):
    """
    Φ(s) = -min_distance_to_uncovered_cell
    
    Encourages agents to move toward nearest unexplored regions.
    Good for: Slow initial exploration, agents ignoring far regions
    """
    
    def __call__(self, state: Dict, agent_positions: np.ndarray, 
                 coverage_map: np.ndarray) -> float:
        uncovered = (coverage_map == 0).astype(np.float32)
        if uncovered.sum() == 0:
            return 0.0  # Fully covered
        
        # Distance transform: each cell = distance to nearest uncovered
        dist_to_uncovered = distance_transform_edt(1 - uncovered)
        
        # Average distance from each agent to nearest frontier
        total_dist = 0.0
        for pos in agent_positions:
            x, y = int(pos[0]), int(pos[1])
            if 0 <= x < coverage_map.shape[0] and 0 <= y < coverage_map.shape[1]:
                total_dist += dist_to_uncovered[x, y]
        
        avg_dist = total_dist / len(agent_positions)
        return -avg_dist  # Negative so closer to frontier = higher potential


class LocalCoveragePotential(PotentialFunction):
    """
    Φ(s) = Σ_agents (coverage in local window around agent)
    
    Rewards being in areas with good local coverage progress.
    Good for: Encouraging systematic sweeping patterns
    """
    
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
    
    def __call__(self, state: Dict, agent_positions: np.ndarray, 
                 coverage_map: np.ndarray) -> float:
        total_local_coverage = 0.0
        half_window = self.window_size // 2
        
        for pos in agent_positions:
            x, y = int(pos[0]), int(pos[1])
            x_min = max(0, x - half_window)
            x_max = min(coverage_map.shape[0], x + half_window + 1)
            y_min = max(0, y - half_window)
            y_max = min(coverage_map.shape[1], y + half_window + 1)
            
            local_window = coverage_map[x_min:x_max, y_min:y_max]
            total_local_coverage += local_window.sum()
        
        return total_local_coverage


class ExpectedCoveragePotential(PotentialFunction):
    """
    Φ(s) = Σ_cells P(cell will be covered soon | agent positions)
    
    Uses distance-weighted coverage expectation.
    Good for: General acceleration, works well across different scenarios
    """
    
    def __init__(self, decay_rate: float = 0.3):
        self.decay_rate = decay_rate
    
    def __call__(self, state: Dict, agent_positions: np.ndarray, 
                 coverage_map: np.ndarray) -> float:
        grid_size = coverage_map.shape[0]
        expected_coverage = np.zeros_like(coverage_map, dtype=np.float32)
        
        for pos in agent_positions:
            # Create distance map from this agent
            y_grid, x_grid = np.ogrid[:grid_size, :grid_size]
            distances = np.sqrt((x_grid - pos[0])**2 + (y_grid - pos[1])**2)
            
            # Coverage probability decays with distance
            prob = np.exp(-self.decay_rate * distances)
            expected_coverage += prob
        
        # Weight by uncovered cells (don't reward revisiting)
        uncovered_mask = (coverage_map == 0).astype(np.float32)
        potential = (expected_coverage * uncovered_mask).sum()
        
        return potential


class CoordinationPotential(PotentialFunction):
    """
    Φ(s) = coverage_potential - separation_penalty
    
    Balances coverage progress with agent separation.
    Good for: Agent clustering, redundant coverage
    """
    
    def __init__(self, min_separation: float = 5.0, separation_weight: float = 0.5):
        self.min_separation = min_separation
        self.separation_weight = separation_weight
    
    def __call__(self, state: Dict, agent_positions: np.ndarray, 
                 coverage_map: np.ndarray) -> float:
        # Coverage potential: simple coverage percentage
        coverage_potential = coverage_map.sum()
        
        # Separation penalty: sum of pairwise distances below threshold
        separation_penalty = 0.0
        num_agents = len(agent_positions)
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                dist = np.linalg.norm(agent_positions[i] - agent_positions[j])
                if dist < self.min_separation:
                    separation_penalty += (self.min_separation - dist)
        
        return coverage_potential - self.separation_weight * separation_penalty


class PotentialBasedRewardShaper:
    """
    Applies potential-based reward shaping: F(s, a, s') = γ * Φ(s') - Φ(s)
    
    Usage in training loop:
        shaper = PotentialBasedRewardShaper(potential_fn, gamma=0.99, beta=0.1)
        
        # After environment step:
        shaped_reward = shaper.shape_reward(
            base_reward=reward,
            prev_state=prev_state,
            next_state=next_state,
            agent_positions=positions,
            coverage_map=coverage_map
        )
    """
    
    def __init__(self, potential_fn: PotentialFunction, 
                 gamma: float = config.GAMMA,
                 beta: float = 0.1,
                 enabled: bool = True):
        """
        Args:
            potential_fn: The potential function Φ
            gamma: Discount factor
            beta: Shaping coefficient (0 = no shaping, 1 = full potential difference)
            enabled: Whether shaping is active (for easy ablation studies)
        """
        self.potential_fn = potential_fn
        self.gamma = gamma
        self.beta = beta
        self.enabled = enabled
        
        self.prev_potential = None  # Cache for efficiency
    
    def reset(self):
        """Call at start of each episode."""
        self.prev_potential = None
    
    def shape_reward(self, base_reward: float, prev_state: Dict, next_state: Dict,
                     agent_positions: np.ndarray, next_positions: np.ndarray,
                     coverage_map: np.ndarray, next_coverage_map: np.ndarray) -> float:
        """
        Add shaping reward to base reward.
        
        Returns:
            shaped_reward = base_reward + β * [γ * Φ(s') - Φ(s)]
        """
        if not self.enabled:
            return base_reward
        
        # Compute potentials
        if self.prev_potential is None:
            prev_potential = self.potential_fn(prev_state, agent_positions, coverage_map)
        else:
            prev_potential = self.prev_potential
        
        next_potential = self.potential_fn(next_state, next_positions, next_coverage_map)
        
        # Cache for next call
        self.prev_potential = next_potential
        
        # Shaping reward: F = γ * Φ(s') - Φ(s)
        shaping_reward = self.gamma * next_potential - prev_potential
        shaped_reward = base_reward + self.beta * shaping_reward
        
        return shaped_reward
    
    def get_stats(self) -> Dict[str, float]:
        """Return statistics for logging."""
        return {
            'potential_value': self.prev_potential if self.prev_potential is not None else 0.0,
            'shaping_enabled': float(self.enabled),
            'shaping_beta': self.beta
        }


# Example configurations for different scenarios
def get_shaper_config(scenario: str = 'none') -> Optional[PotentialBasedRewardShaper]:
    """
    Factory function for creating shapers.
    
    Args:
        scenario: One of ['none', 'frontier', 'local', 'expected', 'coordination']
    
    Returns:
        PotentialBasedRewardShaper or None
    """
    if scenario == 'none':
        return None
    
    elif scenario == 'frontier':
        # Good for: Slow exploration, agents ignoring far regions
        potential = FrontierDistancePotential()
        return PotentialBasedRewardShaper(potential, beta=0.1, enabled=True)
    
    elif scenario == 'local':
        # Good for: Encouraging systematic sweeping
        potential = LocalCoveragePotential(window_size=7)
        return PotentialBasedRewardShaper(potential, beta=0.05, enabled=True)
    
    elif scenario == 'expected':
        # Good for: General acceleration, balanced
        potential = ExpectedCoveragePotential(decay_rate=0.2)
        return PotentialBasedRewardShaper(potential, beta=0.15, enabled=True)
    
    elif scenario == 'coordination':
        # Good for: Agent clustering, redundant coverage
        potential = CoordinationPotential(min_separation=5.0, separation_weight=0.5)
        return PotentialBasedRewardShaper(potential, beta=0.1, enabled=True)
    
    else:
        raise ValueError(f"Unknown scenario: {scenario}")


# Recommended usage in train_marl.py:
"""
# At top of training script:
from potential_based_shaping import get_shaper_config

# In main():
USE_PBRS = False  # START WITH FALSE!
PBRS_SCENARIO = 'expected'  # Only matters if USE_PBRS = True

if USE_PBRS:
    reward_shaper = get_shaper_config(PBRS_SCENARIO)
    print(f"Using PBRS with scenario: {PBRS_SCENARIO}")
else:
    reward_shaper = None
    print("Training without PBRS (recommended for first run)")

# In training loop, after env.step():
if reward_shaper is not None:
    reward = reward_shaper.shape_reward(
        base_reward=reward,
        prev_state=prev_state,
        next_state=next_state,
        agent_positions=prev_positions,
        next_positions=positions,
        coverage_map=prev_coverage_map,
        next_coverage_map=coverage_map
    )

# At episode reset:
if reward_shaper is not None:
    reward_shaper.reset()
"""
