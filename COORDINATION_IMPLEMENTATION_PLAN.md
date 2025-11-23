# Multi-Agent Coordination Implementation Plan

**Goal**: Transform independent agents into coordinated team
**Current Status**: 4 independent agents with visual collision avoidance (35-40% overlap)
**Target Status**: Coordinated multi-agent system with QMIX (12-18% overlap)

---

## Phase 1: Add Communication System (Priority: CRITICAL)

**Implementation Time**: ~1-2 hours
**Files to Modify**: `communication.py`, `agent_occupancy.py`, `multi_agent_env.py`

### 1.1 Enable Position-Based Communication

**File**: `communication.py`

```python
class PositionCommunication:
    """
    Position-based communication protocol.
    Agents broadcast (position, velocity, timestamp) every N steps.
    """

    def __init__(self, num_agents: int, comm_range: float = 15.0, comm_freq: int = 5):
        self.num_agents = num_agents
        self.comm_range = comm_range
        self.comm_freq = comm_freq
        self.last_messages = {}  # agent_id -> (pos, vel, timestamp)
        self.step_count = 0

    def broadcast(self, agent_id: int, position: Tuple[float, float],
                  velocity: Tuple[float, float]) -> None:
        """Agent broadcasts its position and velocity."""
        self.last_messages[agent_id] = (position, velocity, self.step_count)

    def receive(self, agent_id: int, agent_position: Tuple[float, float]) -> List[Dict]:
        """
        Receive messages from other agents within communication range.

        Returns:
            List of messages from agents within range:
            [{'agent_id': int, 'position': (x,y), 'velocity': (vx,vy),
              'age': int, 'distance': float}, ...]
        """
        messages = []
        for other_id, (pos, vel, timestamp) in self.last_messages.items():
            if other_id == agent_id:
                continue

            # Check if within communication range
            distance = np.sqrt((pos[0] - agent_position[0])**2 +
                             (pos[1] - agent_position[1])**2)

            if distance <= self.comm_range:
                age = self.step_count - timestamp
                messages.append({
                    'agent_id': other_id,
                    'position': pos,
                    'velocity': vel,
                    'age': age,
                    'distance': distance
                })

        return messages

    def step(self) -> None:
        """Increment step counter."""
        self.step_count += 1
```

### 1.2 Update Agent Occupancy Channel from Communication

**File**: `agent_occupancy.py`

Modify `update_from_communication()` method:

```python
def update_from_communication(self, agent_id: int, messages: List[Dict]) -> None:
    """
    Update agent occupancy map from communication messages.

    Args:
        agent_id: This agent's ID
        messages: List of messages from other agents
    """
    for msg in messages:
        other_id = msg['agent_id']
        position = msg['position']
        age = msg['age']

        # Calculate uncertainty growth
        # σ(t) = σ_base + σ_growth * Δt
        uncertainty = self.uncertainty_base + self.uncertainty_growth * age

        # Update probability map for this agent
        x_grid, y_grid = np.meshgrid(
            np.arange(self.grid_size),
            np.arange(self.grid_size)
        )

        # Gaussian centered at communicated position
        dx = x_grid - position[0]
        dy = y_grid - position[1]
        distance_sq = dx**2 + dy**2

        # P(agent at cell) = exp(-distance²/2σ²)
        probability = np.exp(-distance_sq / (2 * uncertainty**2))

        # Update occupancy for this agent
        self.agent_occupancy_maps[other_id] = probability
```

### 1.3 Integrate Communication into Environment

**File**: `multi_agent_env.py`

```python
def __init__(...):
    # ... existing code ...

    # Initialize communication
    if ma_config.USE_COMMUNICATION:
        self.comm_system = PositionCommunication(
            num_agents=num_agents,
            comm_range=ma_config.COMMUNICATION_RANGE,
            comm_freq=ma_config.COMMUNICATION_FREQUENCY
        )
    else:
        self.comm_system = None

def step(self, actions):
    # ... execute actions ...

    # Communication step
    if self.comm_system is not None:
        # Broadcast positions
        if self.state.step_count % self.comm_system.comm_freq == 0:
            for agent in self.state.agents:
                self.comm_system.broadcast(
                    agent.agent_id,
                    agent.robot_state.position,
                    agent.robot_state.velocity
                )

        # Update occupancy from messages
        for agent in self.state.agents:
            messages = self.comm_system.receive(
                agent.agent_id,
                agent.robot_state.position
            )
            self.occupancy_computer.update_from_communication(
                agent.agent_id,
                messages
            )

        self.comm_system.step()

    # ... rest of step logic ...
```

### 1.4 Enable in Configuration

**File**: `multi_agent_config.py`

```python
# Communication settings
USE_COMMUNICATION = True  # CRITICAL: Enable communication!
COMMUNICATION_RANGE = 15.0  # Already set correctly
COMMUNICATION_FREQUENCY = 5  # Already set correctly
```

**Expected Impact of Phase 1:**
- Overlap: 35-40% → 25-30% (agents can anticipate each other)
- Coverage: 52% → 58-62% (better coordination)
- Validation: Starting to improve
- 6th channel now USEFUL (persistent position information)

---

## Phase 2: Implement QMIX Value Decomposition (Priority: CRITICAL)

**Implementation Time**: ~2-3 hours
**Files to Create**: `qmix.py`
**Files to Modify**: `multi_agent_trainer.py`, `train_multi_agent.py`

### 2.1 QMIX Architecture

**File**: `qmix.py` (new file)

```python
"""
QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL
Reference: Rashid et al., ICML 2018
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixingNetwork(nn.Module):
    """
    Mixing network for QMIX.

    Combines individual Q-values into joint Q_tot with monotonicity constraint.
    Uses hypernetworks to generate weights conditioned on global state.
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_embed: int = 64
    ):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork for layer 1 weights
        # Generates weights of shape (num_agents, embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, num_agents * embed_dim)
        )

        # Hypernetwork for layer 1 bias
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for layer 2 weights
        # Generates weights of shape (embed_dim, 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim)
        )

        # Hypernetwork for layer 2 bias
        # This is state-dependent scalar
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into joint Q_tot.

        Args:
            agent_qs: Individual Q-values [batch_size, num_agents]
            state: Global state [batch_size, state_dim]

        Returns:
            q_tot: Joint Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)

        # Generate mixing network weights from state
        # All weights are non-negative (via abs()) to ensure monotonicity

        # Layer 1: [num_agents] -> [embed_dim]
        w1 = torch.abs(self.hyper_w1(state))  # [batch, num_agents * embed_dim]
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)  # [batch, agents, embed]

        b1 = self.hyper_b1(state)  # [batch, embed_dim]
        b1 = b1.view(batch_size, 1, self.embed_dim)  # [batch, 1, embed]

        # Layer 2: [embed_dim] -> [1]
        w2 = torch.abs(self.hyper_w2(state))  # [batch, embed_dim]
        w2 = w2.view(batch_size, self.embed_dim, 1)  # [batch, embed, 1]

        b2 = self.hyper_b2(state)  # [batch, 1]

        # Mix individual Q-values
        # Q_tot = b2 + w2^T * (ReLU(w1^T * agent_qs + b1))
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)  # [batch, 1, agents]

        # Layer 1: [batch, 1, agents] @ [batch, agents, embed] -> [batch, 1, embed]
        hidden = torch.bmm(agent_qs, w1) + b1
        hidden = F.elu(hidden)  # ELU works better than ReLU for mixing

        # Layer 2: [batch, 1, embed] @ [batch, embed, 1] -> [batch, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2

        return q_tot.view(batch_size, 1)


class QMIXLoss(nn.Module):
    """
    QMIX training loss with TD-error.
    """

    def __init__(self, gamma: float = 0.99):
        super().__init__()
        self.gamma = gamma

    def forward(
        self,
        q_tot: torch.Tensor,
        target_q_tot: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute QMIX TD-error loss.

        Args:
            q_tot: Current Q_tot [batch_size, 1]
            target_q_tot: Target Q_tot from next state [batch_size, 1]
            rewards: Team rewards [batch_size, 1]
            dones: Episode termination flags [batch_size, 1]

        Returns:
            loss: Mean squared TD-error
        """
        # TD target: r + γ * (1 - done) * Q_tot(s', a')
        td_target = rewards + self.gamma * (1 - dones) * target_q_tot

        # TD error
        td_error = q_tot - td_target.detach()

        # Mean squared loss
        loss = (td_error ** 2).mean()

        return loss
```

### 2.2 Global State Representation

**File**: `multi_agent_env.py`

```python
def get_global_state(self) -> np.ndarray:
    """
    Get global state for QMIX mixing network.

    Global state includes:
    - Team coverage map (1600 values for 40x40)
    - All agent positions (8 values for 4 agents)
    - Total coverage percentage (1 value)
    - Episode progress (1 value)

    Total: 1610 dimensions
    """
    # Coverage map (flattened)
    coverage_map = self.state.world_state.coverage_map.flatten()  # [1600]

    # All agent positions
    positions = []
    for agent in self.state.agents:
        positions.extend([
            agent.robot_state.position[0] / self.grid_size,  # normalized x
            agent.robot_state.position[1] / self.grid_size   # normalized y
        ])
    positions = np.array(positions)  # [8] for 4 agents

    # Coverage percentage
    coverage_pct = np.array([self._get_coverage_percentage()])  # [1]

    # Episode progress
    progress = np.array([self.state.step_count / config.MAX_EPISODE_STEPS])  # [1]

    # Concatenate
    global_state = np.concatenate([
        coverage_map,
        positions,
        coverage_pct,
        progress
    ])

    return global_state  # [1610]
```

### 2.3 Integrate QMIX into Trainer

**File**: `multi_agent_trainer.py`

```python
def __init__(..., use_qmix: bool = False):
    # ... existing initialization ...

    self.use_qmix = use_qmix

    if self.use_qmix:
        # Initialize QMIX mixing network
        from qmix import QMixingNetwork, QMIXLoss

        state_dim = 1610  # From get_global_state()
        self.mixing_net = QMixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim,
            embed_dim=32,
            hypernet_embed=64
        ).to(self.device)

        self.target_mixing_net = QMixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim,
            embed_dim=32,
            hypernet_embed=64
        ).to(self.device)
        self.target_mixing_net.load_state_dict(self.mixing_net.state_dict())

        # Optimizer includes mixing network parameters
        all_params = []
        for agent in self.agents:
            all_params.extend(agent.network.parameters())
        all_params.extend(self.mixing_net.parameters())

        self.optimizer = torch.optim.Adam(all_params, lr=self.learning_rate)

        # QMIX loss
        self.qmix_loss_fn = QMIXLoss(gamma=config.GAMMA)

def train_step(self, batch):
    """Training step with QMIX."""
    if not self.use_qmix:
        return self._train_step_independent(batch)
    else:
        return self._train_step_qmix(batch)

def _train_step_qmix(self, batch):
    """
    QMIX training step.

    Key difference from independent:
    - Compute individual Q-values for chosen actions
    - Mix them with global state to get Q_tot
    - Compute TD-error on Q_tot (not individual Q-values)
    - Backprop through mixing network to all agents
    """
    states, actions, rewards, next_states, dones, global_states, next_global_states = batch

    batch_size = len(states)

    # === Compute current Q-values ===
    # Individual Q-values for chosen actions
    agent_qs = []
    for i, agent in enumerate(self.agents):
        state_batch = torch.stack([s[i] for s in states]).to(self.device)
        action_batch = torch.tensor([a[i] for a in actions]).to(self.device)

        q_values = agent.network(state_batch)  # [batch, num_actions]
        q_chosen = q_values.gather(1, action_batch.unsqueeze(1))  # [batch, 1]
        agent_qs.append(q_chosen)

    agent_qs = torch.cat(agent_qs, dim=1)  # [batch, num_agents]

    # Mix individual Q-values
    global_state_batch = torch.stack(global_states).to(self.device)
    q_tot = self.mixing_net(agent_qs, global_state_batch)  # [batch, 1]

    # === Compute target Q-values ===
    with torch.no_grad():
        target_agent_qs = []
        for i, agent in enumerate(self.agents):
            next_state_batch = torch.stack([s[i] for s in next_states]).to(self.device)
            target_q_values = agent.target_network(next_state_batch)
            target_q_max = target_q_values.max(dim=1, keepdim=True)[0]
            target_agent_qs.append(target_q_max)

        target_agent_qs = torch.cat(target_agent_qs, dim=1)

        # Mix target Q-values
        next_global_state_batch = torch.stack(next_global_states).to(self.device)
        target_q_tot = self.target_mixing_net(target_agent_qs, next_global_state_batch)

    # === Compute loss and update ===
    # Team reward (sum or mean of individual rewards)
    team_rewards = torch.tensor([sum(r) for r in rewards]).unsqueeze(1).to(self.device)
    dones_tensor = torch.tensor(dones).unsqueeze(1).to(self.device)

    loss = self.qmix_loss_fn(q_tot, target_q_tot, team_rewards, dones_tensor)

    self.optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(self.mixing_net.parameters(), config.GRAD_CLIP_NORM)
    for agent in self.agents:
        torch.nn.utils.clip_grad_norm_(agent.network.parameters(), config.GRAD_CLIP_NORM)

    self.optimizer.step()

    return {
        'loss': loss.item(),
        'q_tot_mean': q_tot.mean().item(),
        'target_q_tot_mean': target_q_tot.mean().item()
    }
```

### 2.4 Update Replay Buffer for QMIX

**File**: `multi_agent_trainer.py`

```python
def store_transition(self, states, actions, rewards, next_states, done):
    """
    Store transition in replay buffer.
    For QMIX, also store global states.
    """
    if self.use_qmix:
        # Get global states
        global_state = self.env.get_global_state()
        # Store with global state
        self.replay_buffer.append((
            states, actions, rewards, next_states, done, global_state
        ))
    else:
        # Independent training (existing)
        self.replay_buffer.append((states, actions, rewards, next_states, done))
```

**Expected Impact of Phase 2:**
- Overlap: 25-30% → 15-20% (joint optimization!)
- Coverage: 58-62% → 72-78% (team strategy)
- Validation: Significant improvement
- Coordination score: 42 → 70+

---

## Phase 3: Training Configuration

**File**: `multi_agent_config.py`

```python
# Enable all coordination features
USE_COMMUNICATION = True
USE_QMIX = True
PARAMETER_SHARING = True  # OK with QMIX (shares mixing network)

# Team reward components (already set)
USE_OVERLAP_PENALTY = True
OVERLAP_PENALTY_SCALE = 0.01  # Keep current (working)

USE_DIVERSITY_BONUS = True
DIVERSITY_BONUS_SCALE = 0.5

USE_EFFICIENCY_BONUS = True
EFFICIENCY_BONUS_SCALE = 5.0

# QMIX-specific settings
QMIX_EMBED_DIM = 32
QMIX_HYPERNET_EMBED = 64

# Training episodes
TOTAL_EPISODES = 600  # Reduce from 800 (should learn faster with QMIX)
```

---

## Phase 4: Training Command

```bash
# Full multi-agent coordination training
python train_multi_agent.py \
    --episodes 600 \
    --agents 4 \
    --coordination hierarchical \
    --use-6ch \
    --use-qmix \
    --probabilistic \
    --experiment-name "ma4_qmix_comm_600ep"
```

---

## Expected Results Timeline

### After Phase 1 Only (Communication, no QMIX)
**Episode 100:**
- Coverage: 58-62% (validation)
- Overlap: 28-32%
- Coordination: 50/100

**Episode 200:**
- Coverage: 62-68%
- Overlap: 22-26%
- Coordination: 55/100

**Episode 400:**
- Coverage: 68-72%
- Overlap: 18-24%
- Coordination: 60/100

### After Phase 1 + 2 (Communication + QMIX)
**Episode 100:**
- Coverage: 65-70% (validation)
- Overlap: 22-28%
- Coordination: 58/100

**Episode 200:**
- Coverage: 72-76%
- Overlap: 16-20%
- Coordination: 68/100

**Episode 400:**
- Coverage: 78-82%
- Overlap: 14-18%
- Coordination: 75/100

**Episode 600 (Final):**
- Coverage: 80-85%
- Overlap: 12-16%
- Coordination: 78-82/100

**Per-map validation (Episode 600):**
- Empty: 90-92%
- Random: 82-86%
- Room: 78-82%
- Corridor: 70-75%
- Cave: 72-78%

---

## Implementation Checklist

### Phase 1: Communication (Critical)
- [ ] Implement `PositionCommunication` class in `communication.py`
- [ ] Update `agent_occupancy.py` to use communication messages
- [ ] Integrate communication into `multi_agent_env.py` step function
- [ ] Enable `USE_COMMUNICATION = True` in config
- [ ] Test: Verify 6th channel is populated from messages (not just vision)
- [ ] Test: Run 10 episodes, check occupancy channel is non-zero

### Phase 2: QMIX (Critical)
- [ ] Create `qmix.py` with `QMixingNetwork` and `QMIXLoss`
- [ ] Implement `get_global_state()` in `multi_agent_env.py`
- [ ] Add QMIX training loop in `multi_agent_trainer.py`
- [ ] Update replay buffer to store global states
- [ ] Enable `USE_QMIX = True` in config
- [ ] Test: Verify mixing network forward pass works
- [ ] Test: Run 10 episodes, check Q_tot is computed

### Phase 3: Training
- [ ] Set configuration for 600 episodes
- [ ] Start training with `--use-qmix` flag
- [ ] Monitor overlap decreasing to <20%
- [ ] Monitor validation coverage improving to >75%
- [ ] Save checkpoints every 100 episodes

### Phase 4: Validation
- [ ] Run validation on all map types
- [ ] Verify overlap <18%, coverage >78%
- [ ] Compare to baseline (current 52%, 35-40% overlap)
- [ ] Document results

---

## Alternative: Quick Fix (If Time Constrained)

If you need faster results and can accept suboptimal performance:

**Just Phase 1 (Communication only):**
- Time: 2 hours (implementation + 200 episodes training)
- Result: 65-70% coverage, 22-28% overlap
- Better than current (52%, 35-40%) but not optimal
- Skip QMIX implementation

**Use this if:**
- Need results in 2-3 hours
- Can accept "decent" not "publication-quality"
- Want to validate communication works before QMIX

**DON'T use this if:**
- Need state-of-the-art results
- Want true multi-agent coordination
- Have 8 hours available (do full implementation)

---

## Bottom Line

**Current system limitations:**
- ❌ No communication = 6th channel useless
- ❌ Independent Q-learning = no team optimization
- ❌ Result: 52% coverage, 40% overlap (poor)

**After full implementation:**
- ✅ Communication = persistent position information
- ✅ QMIX = joint value function, monotonic mixing
- ✅ Result: 80-85% coverage, 12-18% overlap (excellent)

**Time investment:**
- Phase 1: 1-2 hours
- Phase 2: 2-3 hours
- Training: 3-4 hours (600 episodes)
- **Total: 6-9 hours to working multi-agent system**

**This is the correct architecture for multi-agent coordination. Current approach will never achieve <20% overlap without these changes.**
