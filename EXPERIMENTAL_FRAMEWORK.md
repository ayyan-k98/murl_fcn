# Multi-Agent Coverage: Experimental Framework

## Overview

Systematic comparison of:
1. **Architecture variants**: Single-agent baseline, Independent agents, QMIX
2. **Communication protocols**: None, Full-state, Attention-based
3. **Input channels**: 5 channels (baseline) vs 6 channels (+agent occupancy)

**Total experiments**: 3 architectures × 3 comm protocols × 2 input variants = 18 configurations

---

## Experimental Matrix

### Dimension 1: Architecture
```
A1. Single-Agent Baseline (trained on 20×20)
    - FCN with 5 channels
    - Test: Deploy 4 copies independently in multi-agent env
    - Coordination: None (reactive collision avoidance only)

A2. Independent Multi-Agent (CTDE with independent Q-learning)
    - Each agent: Separate FCN Q-network
    - Shared replay buffer (optional)
    - Coordination: Through environment interactions only

A3. QMIX (Value Function Factorization)
    - Individual FCN Q-networks + Mixing network
    - Centralized training, decentralized execution
    - Coordination: Learned through mixing network
```

### Dimension 2: Communication
```
C1. No Communication
    - Agents act on local observations only
    - No message passing
    - Baseline for measuring communication benefit

C2. Full-State Sharing (within comm_range)
    - Share: position, coverage_map, visited_positions
    - Update observations with neighbor information
    - Realistic bandwidth (simplified)

C3. Attention-Based Communication
    - Learned messages via attention mechanism
    - Agents decide what to communicate
    - Most flexible but needs more training
```

### Dimension 3: Input Channels
```
I1. Baseline (5 channels)
    Ch0: Visited map
    Ch1: Coverage map
    Ch2: Self position (one-hot)
    Ch3: Frontier map
    Ch4: Obstacle map
    (+2 CoordConv channels)

I2. With Agent Occupancy (6 channels)
    Ch0-4: Same as baseline
    Ch5: Other agents occupancy probability
    - P(other agent at cell (x,y))
    - Based on last communication + decay
    - Enables proactive coordination
    (+2 CoordConv channels)
```

---

## Implementation Checklist

### Phase 0: Infrastructure ✅ DONE
- [x] FCN architecture supports variable input channels
- [x] Reward normalization for QMIX
- [x] Communication protocols implemented
- [x] Multi-agent environment working

### Phase 1: Single-Agent Baseline (Reference Point)
- [ ] **1.1** Train single-agent FCN (5 channels, 20×20)
  - Episodes: 1000
  - Expected coverage: ~80%
  - Save checkpoint: `single_agent_5ch.pth`
  
- [ ] **1.2** Train single-agent FCN (6 channels, dummy zeros)
  - Same training, but ch5 = all zeros
  - Network learns to ignore extra channel
  - Expected coverage: ~80% (same as 5ch)
  - Save checkpoint: `single_agent_6ch_dummy.pth`
  
- [ ] **1.3** Verify both achieve similar performance
  - Compare coverage, learning curves
  - Verify 6ch doesn't hurt single-agent

### Phase 2: Multi-Agent Infrastructure
- [ ] **2.1** Modify FCN to accept 6 channels
  - Change `input_channels` parameter default
  - Add channel computation in observation encoder
  - Test forward pass works
  
- [ ] **2.2** Implement agent occupancy channel computation
  - `compute_agent_occupancy_channel(messages, current_time)`
  - Gaussian decay over time
  - Probabilistic union aggregation
  
- [ ] **2.3** Update observation encoding
  - Single-agent: ch5 = zeros
  - Multi-agent: ch5 = agent_occupancy
  - Seamless switching

### Phase 3: Independent Agents Experiments
**Group 1: No Communication**
- [ ] **3.1.1** Independent agents, 5ch, no comm
  - Baseline: 4 independent agents
  - Each uses single_agent_5ch.pth (frozen or fine-tune)
  - 400 episodes training
  
- [ ] **3.1.2** Independent agents, 6ch, no comm
  - ch5 = zeros (no other agents known)
  - Should perform similar to 3.1.1
  - 400 episodes training

**Group 2: Full-State Communication**
- [ ] **3.2.1** Independent agents, 5ch, full-state comm
  - Share coverage maps within range
  - Agents update their observations
  - 400 episodes training
  
- [ ] **3.2.2** Independent agents, 6ch, full-state comm
  - ch5 = agent occupancy from messages
  - Should outperform 3.2.1 (explicit coordination)
  - 400 episodes training

**Group 3: Attention Communication**
- [ ] **3.3.1** Independent agents, 5ch, attention comm
  - Learned messages
  - 400 episodes training
  
- [ ] **3.3.2** Independent agents, 6ch, attention comm
  - ch5 = agent occupancy + learned messages
  - 400 episodes training

### Phase 4: QMIX Experiments
**Group 1: No Communication**
- [ ] **4.1.1** QMIX, 5ch, no comm
  - Mixing network learns coordination
  - 400 episodes training
  
- [ ] **4.1.2** QMIX, 6ch, no comm (dummy)
  - ch5 = zeros
  - Should perform similar to 4.1.1
  - 400 episodes training

**Group 2: Full-State Communication**
- [ ] **4.2.1** QMIX, 5ch, full-state comm
  - Messages + mixing network
  - 400 episodes training
  
- [ ] **4.2.2** QMIX, 6ch, full-state comm
  - ch5 = agent occupancy
  - Expected: BEST performance
  - 400 episodes training

**Group 3: Attention Communication**
- [ ] **4.3.1** QMIX, 5ch, attention comm
  - 400 episodes training
  
- [ ] **4.3.2** QMIX, 6ch, attention comm
  - 400 episodes training

---

## Metrics to Track

### Primary Metrics
```python
metrics = {
    'coverage_pct': float,        # Final coverage percentage
    'episode_length': int,         # Steps to completion
    'team_reward': float,          # Cumulative team reward
    'efficiency': float,           # coverage / episode_length
}
```

### Coordination Metrics
```python
coordination_metrics = {
    'overlap_cells': int,          # Cells visited by multiple agents
    'overlap_pct': float,          # overlap / total_covered
    'collisions': int,             # Agent-agent collisions
    'separation_avg': float,       # Mean distance between agents
    'frontier_conflicts': int,     # Multiple agents chase same frontier
}
```

### Communication Metrics (when applicable)
```python
comm_metrics = {
    'messages_sent': int,          # Total messages
    'messages_per_agent': float,   # Average messages per agent
    'comm_frequency': float,       # Messages per step
    'bandwidth_used': float,       # If measuring message size
}
```

### Learning Metrics
```python
learning_metrics = {
    'loss': List[float],           # Training loss over episodes
    'epsilon': List[float],        # Exploration rate
    'grad_norm': List[float],      # Gradient magnitudes
    'q_values': List[float],       # Q-value statistics
}
```

---

## Expected Results

### Hypothesis Matrix

| Config | Coverage | Overlap | Collisions | Notes |
|--------|----------|---------|------------|-------|
| Single-agent × 4 | 75-80% | 30-35% | 15-20 | Baseline, no coordination |
| Indep, 5ch, no comm | 78-82% | 28-32% | 12-18 | Learning collision avoidance |
| Indep, 6ch, no comm | 78-82% | 28-32% | 12-18 | Same (ch5 unused without comm) |
| Indep, 5ch, full comm | 82-85% | 20-25% | 8-12 | Reactive coordination |
| **Indep, 6ch, full comm** | **85-88%** | **15-20%** | **5-8** | **Proactive coordination** ⭐ |
| QMIX, 5ch, no comm | 83-86% | 18-22% | 6-10 | Mixing network coordination |
| QMIX, 6ch, no comm | 83-86% | 18-22% | 6-10 | Same (ch5 = zeros) |
| QMIX, 5ch, full comm | 86-89% | 15-18% | 4-7 | Comm + mixing |
| **QMIX, 6ch, full comm** | **88-92%** | **10-15%** | **2-5** | **Best expected** 🏆 |

**Key predictions:**
1. 6th channel helps ONLY when communication provides position info
2. QMIX > Independent > Single-agent baseline
3. Communication always helps (5-10% coverage improvement)
4. 6th channel gives 3-5% additional improvement with comm

---

## Implementation Plan

### Step 1: Prepare FCN for 6 Channels

**File: `fcn_spatial_network.py`**
```python
# Current: input_channels=5
# Change to: input_channels=5 (default, backward compatible)
# But support input_channels=6 when needed

# No changes needed! Already parameterized ✓
```

**File: `fcn_agent.py`**
```python
def _encode_state(self, robot_state, world_state, 
                  agent_occupancy=None):
    """
    Encode state as tensor for FCN.
    
    Args:
        robot_state: RobotState
        world_state: WorldState
        agent_occupancy: Optional [H, W] array of other agent probs
    
    Returns:
        state_tensor: [5 or 6, H, W] depending on agent_occupancy
    """
    channels = []
    
    # Ch0-4: Standard channels
    channels.append(visited_map)
    channels.append(coverage_map)
    channels.append(agent_position)
    channels.append(frontier_map)
    channels.append(obstacle_map)
    
    # Ch5: Optional agent occupancy
    if agent_occupancy is not None:
        channels.append(agent_occupancy)
    
    return torch.stack(channels, dim=0)
```

### Step 2: Implement Agent Occupancy Computation

**File: `agent_occupancy.py` (NEW)**
```python
"""
Agent occupancy probability channel computation.

Computes P(other agent at cell (x,y)) based on:
1. Last communicated positions
2. Time decay (uncertainty grows)
3. Probabilistic union (multiple agents)
"""

import numpy as np
from typing import List, Tuple
import torch

class AgentOccupancyComputer:
    """Compute agent occupancy probability channel."""
    
    def __init__(self, 
                 grid_size: int,
                 base_sigma: float = 0.5,
                 max_velocity: float = 1.0,
                 comm_range: float = 10.0):
        self.grid_size = grid_size
        self.base_sigma = base_sigma
        self.max_velocity = max_velocity
        self.comm_range = comm_range
        
        # Cache for Gaussian computation
        self._gaussian_cache = {}
    
    def compute(self,
                agent_id: int,
                messages: List[Dict],
                current_time: int) -> np.ndarray:
        """
        Compute occupancy probability for OTHER agents.
        
        Args:
            agent_id: ID of agent computing occupancy (exclude self)
            messages: List of recent messages from other agents
            current_time: Current timestep
            
        Returns:
            occupancy: [H, W] array of probabilities [0, 1]
        """
        occupancy = np.zeros((self.grid_size, self.grid_size))
        
        for msg in messages:
            if msg['sender_id'] == agent_id:
                continue  # Don't include self
            
            position = msg['position']
            timestamp = msg['timestamp']
            delta_t = current_time - timestamp
            
            # Uncertainty grows with time
            sigma = self.base_sigma + self.max_velocity * delta_t
            
            # Gaussian probability distribution
            prob_map = self._gaussian_at(position, sigma)
            
            # Probabilistic union: P(A or B) = 1 - (1-P(A))(1-P(B))
            occupancy = 1 - (1 - occupancy) * (1 - prob_map)
        
        return occupancy
    
    def _gaussian_at(self, position: Tuple[float, float], 
                     sigma: float) -> np.ndarray:
        """Compute 2D Gaussian centered at position."""
        # Use cached version if available
        cache_key = (position[0], position[1], sigma)
        if cache_key in self._gaussian_cache:
            return self._gaussian_cache[cache_key]
        
        # Compute Gaussian
        y, x = np.ogrid[:self.grid_size, :self.grid_size]
        dist_sq = (x - position[0])**2 + (y - position[1])**2
        gaussian = np.exp(-dist_sq / (2 * sigma**2))
        
        # Cache (limit cache size)
        if len(self._gaussian_cache) < 1000:
            self._gaussian_cache[cache_key] = gaussian
        
        return gaussian
```

### Step 3: Modify Training Scripts

**File: `train_fcn.py` (single-agent baseline)**
```python
# Add flag for 6-channel dummy training
parser.add_argument('--dummy-6ch', action='store_true',
                   help='Train with 6 channels (ch5=zeros)')

if args.dummy_6ch:
    # Create agent with 6 channels
    agent = FCNAgent(grid_size=20, input_channels=6)
    # During encoding, always append zeros as ch5
else:
    agent = FCNAgent(grid_size=20, input_channels=5)
```

**File: `train_multi_agent.py` (independent agents)**
```python
# Add flags
parser.add_argument('--use-6ch', action='store_true',
                   help='Use 6-channel input with agent occupancy')
parser.add_argument('--comm-protocol', type=str, default='none',
                   choices=['none', 'full_state', 'attention'])

# Initialize occupancy computer if using 6ch
if args.use_6ch:
    occupancy_computer = AgentOccupancyComputer(grid_size=20)
else:
    occupancy_computer = None

# In training loop:
for step in range(max_steps):
    # Get messages (if communication enabled)
    messages = comm_manager.communicate(observations, state)
    
    # Compute agent occupancy (if using 6ch)
    if occupancy_computer:
        agent_occupancies = [
            occupancy_computer.compute(i, messages, step)
            for i in range(num_agents)
        ]
    else:
        agent_occupancies = [None] * num_agents
    
    # Select actions (with optional occupancy)
    actions = [
        agent.select_action(obs, occupancy=agent_occupancies[i])
        for i, (agent, obs) in enumerate(zip(agents, observations))
    ]
```

**File: `train_qmix.py`**
```python
# Same modifications as train_multi_agent.py
# QMIX agent uses same FCN architecture
# Just add --use-6ch flag and occupancy computation
```

### Step 4: Experiment Runner Script

**File: `run_experiments.py` (NEW)**
```python
"""
Run all experiments in the experimental matrix.

Usage:
    python run_experiments.py --phase 1  # Single-agent baselines
    python run_experiments.py --phase 2  # Independent agents
    python run_experiments.py --phase 3  # QMIX
    python run_experiments.py --all      # Run everything
"""

import subprocess
import argparse
from datetime import datetime

experiments = {
    'phase1': [
        {
            'name': '1.1_single_5ch',
            'cmd': 'python train_fcn.py --episodes 1000',
            'description': 'Single-agent baseline, 5 channels'
        },
        {
            'name': '1.2_single_6ch_dummy',
            'cmd': 'python train_fcn.py --episodes 1000 --dummy-6ch',
            'description': 'Single-agent with dummy 6th channel'
        }
    ],
    
    'phase2': [
        # Independent, no comm
        {
            'name': '3.1.1_indep_5ch_nocomm',
            'cmd': 'python train_multi_agent.py --episodes 400 --agents 4 --comm-protocol none',
            'description': 'Independent agents, 5ch, no communication'
        },
        {
            'name': '3.1.2_indep_6ch_nocomm',
            'cmd': 'python train_multi_agent.py --episodes 400 --agents 4 --comm-protocol none --use-6ch',
            'description': 'Independent agents, 6ch, no communication'
        },
        
        # Independent, full-state comm
        {
            'name': '3.2.1_indep_5ch_fullcomm',
            'cmd': 'python train_multi_agent.py --episodes 400 --agents 4 --comm-protocol full_state',
            'description': 'Independent agents, 5ch, full-state comm'
        },
        {
            'name': '3.2.2_indep_6ch_fullcomm',
            'cmd': 'python train_multi_agent.py --episodes 400 --agents 4 --comm-protocol full_state --use-6ch',
            'description': 'Independent agents, 6ch, full-state comm ⭐'
        },
        
        # Independent, attention comm
        {
            'name': '3.3.1_indep_5ch_attncomm',
            'cmd': 'python train_multi_agent.py --episodes 400 --agents 4 --comm-protocol attention',
            'description': 'Independent agents, 5ch, attention comm'
        },
        {
            'name': '3.3.2_indep_6ch_attncomm',
            'cmd': 'python train_multi_agent.py --episodes 400 --agents 4 --comm-protocol attention --use-6ch',
            'description': 'Independent agents, 6ch, attention comm'
        }
    ],
    
    'phase3': [
        # QMIX, no comm
        {
            'name': '4.1.1_qmix_5ch_nocomm',
            'cmd': 'python train_qmix.py --episodes 400 --agents 4 --comm-protocol none',
            'description': 'QMIX, 5ch, no communication'
        },
        {
            'name': '4.1.2_qmix_6ch_nocomm',
            'cmd': 'python train_qmix.py --episodes 400 --agents 4 --comm-protocol none --use-6ch',
            'description': 'QMIX, 6ch, no communication (dummy)'
        },
        
        # QMIX, full-state comm
        {
            'name': '4.2.1_qmix_5ch_fullcomm',
            'cmd': 'python train_qmix.py --episodes 400 --agents 4 --comm-protocol full_state',
            'description': 'QMIX, 5ch, full-state comm'
        },
        {
            'name': '4.2.2_qmix_6ch_fullcomm',
            'cmd': 'python train_qmix.py --episodes 400 --agents 4 --comm-protocol full_state --use-6ch',
            'description': 'QMIX, 6ch, full-state comm 🏆 BEST EXPECTED'
        },
        
        # QMIX, attention comm
        {
            'name': '4.3.1_qmix_5ch_attncomm',
            'cmd': 'python train_qmix.py --episodes 400 --agents 4 --comm-protocol attention',
            'description': 'QMIX, 5ch, attention comm'
        },
        {
            'name': '4.3.2_qmix_6ch_attncomm',
            'cmd': 'python train_qmix.py --episodes 400 --agents 4 --comm-protocol attention --use-6ch',
            'description': 'QMIX, 6ch, attention comm'
        }
    ]
}

def run_experiment(exp):
    """Run single experiment."""
    print(f"\n{'='*70}")
    print(f"Running: {exp['name']}")
    print(f"Description: {exp['description']}")
    print(f"Command: {exp['cmd']}")
    print(f"{'='*70}\n")
    
    start_time = datetime.now()
    result = subprocess.run(exp['cmd'], shell=True)
    end_time = datetime.now()
    
    duration = (end_time - start_time).total_seconds() / 3600
    
    if result.returncode == 0:
        print(f"\n✓ {exp['name']} completed in {duration:.2f}h")
        return True
    else:
        print(f"\n✗ {exp['name']} FAILED")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=int, choices=[1, 2, 3])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--experiment', type=str, help='Run specific experiment by name')
    args = parser.parse_args()
    
    if args.all:
        to_run = experiments['phase1'] + experiments['phase2'] + experiments['phase3']
    elif args.phase:
        phase_key = f'phase{args.phase}'
        to_run = experiments[phase_key]
    elif args.experiment:
        # Find experiment by name
        to_run = []
        for phase_exps in experiments.values():
            for exp in phase_exps:
                if args.experiment in exp['name']:
                    to_run.append(exp)
        if not to_run:
            print(f"Experiment '{args.experiment}' not found")
            return
    else:
        print("Specify --phase, --all, or --experiment")
        return
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENTAL RUN")
    print(f"Total experiments: {len(to_run)}")
    print(f"{'='*70}")
    
    results = []
    for exp in to_run:
        success = run_experiment(exp)
        results.append((exp['name'], success))
    
    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    for name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status}: {name}")

if __name__ == "__main__":
    main()
```

---

## Timeline Estimate

**Phase 1 (Single-Agent Baselines):** 2-3 days
- Train 5ch: ~8-10 hours
- Train 6ch dummy: ~8-10 hours
- Analysis: ~4-6 hours

**Phase 2 (Independent Agents):** 5-7 days
- 6 experiments × 8-10 hours each
- Parallel training possible (3 experiments at once on different GPUs)
- Analysis: ~1 day

**Phase 3 (QMIX):** 5-7 days
- 6 experiments × 8-10 hours each
- Most important comparisons
- Detailed analysis: ~1-2 days

**Total:** 12-17 days for complete experimental matrix

---

## Success Criteria

### Must-Have Results:
1. ✅ Single-agent 5ch and 6ch perform similarly (~80% coverage)
2. ✅ Multi-agent > single-agent (any config)
3. ✅ Communication helps (5-10% improvement)
4. ✅ 6ch + comm > 5ch + comm (3-5% improvement)
5. ✅ QMIX > Independent (3-8% improvement)

### Key Comparisons:
- **Indep 6ch full comm** vs **Indep 5ch full comm**: Validates position channel
- **QMIX 6ch full comm** vs **QMIX 5ch full comm**: Best performance check
- **QMIX any** vs **Indep same config**: Validates QMIX benefit

### Ablation Insights:
- 6ch without comm: Should equal 5ch (validates ch5 unused when no position info)
- Comm without 6ch: Should still help (validates comm benefit)
- 6ch + comm: Should be best (validates synergy)

---

## Next Steps

**Immediate (1-2 days):**
1. Create `agent_occupancy.py`
2. Modify `fcn_agent.py` to support optional 6th channel
3. Add `--use-6ch` flag to training scripts
4. Test end-to-end with dummy data

**Short-term (3-5 days):**
1. Train Phase 1 (single-agent baselines)
2. Verify 5ch and 6ch perform similarly
3. Begin Phase 2 (independent agents)

**Medium-term (1-2 weeks):**
1. Complete Phase 2 experiments
2. Analyze results, tune if needed
3. Begin Phase 3 (QMIX)

**Long-term (2-3 weeks):**
1. Complete all experiments
2. Comprehensive analysis and plots
3. Paper/report writing

---

## Current Status

Infrastructure:
- ✅ FCN supports variable input channels
- ✅ Reward normalization implemented
- ✅ Communication protocols ready
- ✅ Multi-agent environment working
- ❌ Agent occupancy computation (need to implement)
- ❌ 6-channel integration (need to add)
- ❌ Experiment runner (need to create)

**READY TO START Phase 1 after implementing agent_occupancy.py**
