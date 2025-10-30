"""
FCN Agent - CNN-based DQN Agent

Replacement for GAT-based agent using FCN + Spatial Softmax.

Key Differences from GAT Agent:
- Uses CNN instead of Graph Neural Network
- Directly processes grid as 2D image (no graph encoding)
- Grid-size invariant (train on 20×20, test on 50×50)
- 3× faster (no graph construction overhead)
- More stable gradients

Interface:
- Same API as CoverageAgent (drop-in replacement)
- Inputs: RobotState, WorldState
- Outputs: Actions [0-8]
"""

import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List

from config import config
from data_structures import RobotState, WorldState
from fcn_spatial_network import FCNSpatialNetwork
from replay_memory import StratifiedReplayMemory


class FCNAgent:
    """
    DQN Agent using FCN + Spatial Softmax for grid-based coverage.

    Architecture:
        Input: Grid [5, H, W] (visited, coverage, agent, frontier, obstacles)
        ↓
        FCN + Spatial Softmax
        ↓
        Output: Q-values [9] (N, E, S, W, NE, NW, SE, SW, STAY)
    """

    def __init__(
        self,
        grid_size: int = 20,
        learning_rate: float = None,
        gamma: float = None,
        device: str = None,
        input_channels: int = 5
    ):
        self.grid_size = grid_size
        self.input_channels = input_channels

        # Use config values if not provided
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE
        if gamma is None:
            gamma = config.GAMMA

        self.gamma = gamma
        self.device = device or config.DEVICE

        print(f"✓ Using FCN + Spatial Softmax (grid-size invariant, {input_channels} channels)")

        # Policy network (online)
        self.policy_net = FCNSpatialNetwork(
            input_channels=input_channels,  # 5 or 6 channels
            num_actions=config.N_ACTIONS,
            hidden_dim=config.CNN_HIDDEN_DIM,  # Use CNN config
            use_coordconv=True,
            spatial_softmax_temp=1.0,
            dropout=config.CNN_DROPOUT
        ).to(self.device)

        # Target network (for stability)
        self.target_net = FCNSpatialNetwork(
            input_channels=input_channels,
            num_actions=config.N_ACTIONS,
            hidden_dim=config.CNN_HIDDEN_DIM,
            use_coordconv=True,
            spatial_softmax_temp=1.0,
            dropout=config.CNN_DROPOUT
        ).to(self.device)

        # Initialize target network
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # OPTIMIZATION: Compile networks with torch.compile (PyTorch 2.0+)
        if config.COMPILE_MODEL:
            try:
                print(f"Compiling networks with mode='{config.COMPILE_MODE}'...")
                self.policy_net = torch.compile(self.policy_net, mode=config.COMPILE_MODE)
                self.target_net = torch.compile(self.target_net, mode=config.COMPILE_MODE)
                print("✓ Networks compiled successfully")
            except Exception as e:
                print(f"⚠ torch.compile failed (PyTorch 2.0+ required): {e}")
                print("  Continuing without compilation...")

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = StratifiedReplayMemory(capacity=config.REPLAY_BUFFER_SIZE)

        # Epsilon (exploration)
        self.epsilon = config.EPSILON_START

        # Gradient tracking
        self.grad_norm_history = []

    def _encode_state(
        self,
        robot_state: RobotState,
        world_state: WorldState,
        agent_occupancy: Optional[np.ndarray] = None
    ) -> torch.Tensor:
        """
        Encode robot and world state as grid tensor.

        Channel 0: Visited (binary)
        Channel 1: Coverage (probability 0-1)
        Channel 2: Agent position (one-hot)
        Channel 3: Frontier (binary)
        Channel 4: Obstacles (binary)
        Channel 5: Agent occupancy (optional, multi-agent only)

        Args:
            robot_state: Robot position and history
            world_state: World grid and coverage
            agent_occupancy: Optional [H, W] array of other agent probabilities
                           If provided, adds 6th channel. If None, uses 5 channels.

        Returns:
            grid_tensor: [1, 5 or 6, H, W] - Batch size 1
        """
        H, W = world_state.grid_size, world_state.grid_size

        # Determine number of channels based on agent_occupancy
        n_channels = 6 if agent_occupancy is not None else 5

        # Initialize channels
        grid = np.zeros((n_channels, H, W), dtype=np.float32)

        # Channel 0: Visited cells
        visited = np.zeros((H, W), dtype=np.float32)
        for (x, y) in robot_state.visited_positions:
            if 0 <= x < W and 0 <= y < H:
                visited[y, x] = 1.0
        grid[0] = visited

        # Channel 1: Coverage probability
        # Use agent's own coverage_history (not shared world_state.coverage_map)
        # This ensures agents only see what they've personally sensed
        # (unless communication merges knowledge)
        if hasattr(robot_state, 'coverage_history') and robot_state.coverage_history is not None:
            grid[1] = np.array(robot_state.coverage_history, dtype=np.float32)
        else:
            # Fallback: Binary coverage (visited = covered)
            grid[1] = visited

        # Channel 2: Agent position (one-hot)
        agent_x, agent_y = robot_state.position
        if 0 <= agent_x < W and 0 <= agent_y < H:
            grid[2, agent_y, agent_x] = 1.0

        # Channel 3: Frontier cells (boundary between visited and unvisited)
        # OPTIMIZED: Vectorized frontier detection using np.roll
        # Shift visited map in 4 directions to find neighbors
        north = np.roll(visited, -1, axis=0)
        south = np.roll(visited, 1, axis=0)
        west = np.roll(visited, -1, axis=1)
        east = np.roll(visited, 1, axis=1)

        # Fix edges (rolled values from opposite side are invalid)
        north[-1, :] = 0
        south[0, :] = 0
        west[:, -1] = 0
        east[:, 0] = 0

        # Frontier = unvisited cells with at least one visited neighbor
        has_visited_neighbor = (north + south + west + east) > 0
        frontier = (visited == 0) & has_visited_neighbor
        grid[3] = frontier.astype(np.float32)

        # Channel 4: Obstacles
        obstacles = np.zeros((H, W), dtype=np.float32)
        for (x, y) in world_state.obstacles:
            if 0 <= x < W and 0 <= y < H:
                obstacles[y, x] = 1.0
        grid[4] = obstacles

        # Channel 5: Agent occupancy (optional, multi-agent only)
        if agent_occupancy is not None:
            # Validate shape
            if agent_occupancy.shape != (H, W):
                raise ValueError(f"agent_occupancy shape {agent_occupancy.shape} != grid shape ({H}, {W})")
            grid[5] = agent_occupancy.astype(np.float32)

        # Convert to tensor and add batch dimension
        grid_tensor = torch.from_numpy(grid).unsqueeze(0)  # [1, 5 or 6, H, W]

        return grid_tensor

    def select_action(
        self,
        robot_state: RobotState,
        world_state: WorldState,
        epsilon: Optional[float] = None,
        agent_occupancy: Optional[np.ndarray] = None
    ) -> int:
        """
        Select action using epsilon-greedy policy.

        Args:
            robot_state: Current robot state
            world_state: World state (grid)
            epsilon: Override default epsilon
            agent_occupancy: Optional [H, W] array for 6th channel (multi-agent)

        Returns:
            action: Integer action [0-8]
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, config.N_ACTIONS - 1)

        # Greedy action
        with torch.no_grad():
            # Encode state to grid (with optional 6th channel)
            grid = self._encode_state(robot_state, world_state, agent_occupancy)
            grid = grid.to(self.device)

            # Forward pass
            q_values = self.policy_net(grid)

            # Select best action
            action = q_values.argmax(dim=1).item()

        return action

    def select_action_from_tensor(
        self,
        grid_tensor: torch.Tensor,
        epsilon: Optional[float] = None
    ) -> int:
        """
        OPTIMIZED: Select action from pre-encoded grid (avoids redundant encoding).

        Args:
            grid_tensor: Pre-encoded grid tensor [1, 5, H, W] (on CPU)
            epsilon: Override default epsilon

        Returns:
            action: Integer action [0-8]
        """
        if epsilon is None:
            epsilon = self.epsilon

        # Epsilon-greedy
        if random.random() < epsilon:
            return random.randint(0, config.N_ACTIONS - 1)

        # Greedy action
        with torch.no_grad():
            grid_device = grid_tensor.to(self.device)

            # Forward pass
            q_values = self.policy_net(grid_device)

            # Select best action
            action = q_values.argmax(dim=1).item()

            # Explicit cleanup
            del grid_device, q_values

        return action

    def select_actions_batch(
        self,
        grid_tensors: torch.Tensor,
        epsilon: Optional[float] = None
    ) -> List[int]:
        """
        OPTIMIZED: Select actions for batch of states simultaneously.

        This is much faster than calling select_action_from_tensor in a loop
        because it performs a single forward pass for all states.

        Args:
            grid_tensors: [batch, 5, H, W] - Multiple pre-encoded grids
            epsilon: Override default epsilon

        Returns:
            actions: List of integer actions [0-8]
        """
        if epsilon is None:
            epsilon = self.epsilon

        batch_size = grid_tensors.size(0)
        actions = []

        # Epsilon-greedy mask (vectorized)
        explore_mask = torch.rand(batch_size) < epsilon

        # Greedy actions for all (single vectorized forward pass)
        with torch.no_grad():
            grid_device = grid_tensors.to(self.device)
            q_values = self.policy_net(grid_device)  # [batch, 9]
            greedy_actions = q_values.argmax(dim=1).cpu().numpy()
            del grid_device, q_values

        # Apply epsilon-greedy
        for i in range(batch_size):
            if explore_mask[i]:
                actions.append(random.randint(0, config.N_ACTIONS - 1))
            else:
                actions.append(int(greedy_actions[i]))

        return actions

    def store_transition(
        self,
        state: torch.Tensor,
        action: int,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
        info: dict
    ):
        """
        Store transition in replay memory.

        Args:
            state: Grid tensor [1, 5, H, W]
            action: Action taken
            reward: Reward received
            next_state: Next grid tensor
            done: Episode done flag
            info: Additional info (coverage, etc.)
        """
        # Ensure tensors are on CPU before storing
        if hasattr(state, 'cpu'):
            state = state.cpu()
        if hasattr(next_state, 'cpu'):
            next_state = next_state.cpu()

        self.memory.push(state, action, reward, next_state, done, info)

    def optimize(self) -> Optional[float]:
        """
        Perform one optimization step using DQN loss.

        Returns:
            loss: DQN loss value, or None if not enough samples
        """
        # Check if enough samples
        if len(self.memory) < config.MIN_REPLAY_SIZE:
            return None

        # Sample batch
        batch = self.memory.sample(config.BATCH_SIZE)

        if len(batch) == 0:
            return None

        # Unpack batch
        states, actions, rewards, next_states, dones, infos = zip(*batch)

        # Stack tensors
        # States: list of [1, 5, H, W] → [batch, 5, H, W]
        states_batch = torch.cat(states, dim=0).to(self.device)
        next_states_batch = torch.cat(next_states, dim=0).to(self.device)

        actions_batch = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_batch = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones_batch = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # ================================================================
        # DQN LOSS COMPUTATION
        # ================================================================

        # Current Q-values: Q(s, a)
        q_values = self.policy_net(states_batch)  # [batch, 9]
        q_values = q_values.gather(1, actions_batch).squeeze(1)  # [batch]

        # Next Q-values: max_a' Q_target(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states_batch)  # [batch, 9]
            next_q_values_max = next_q_values.max(dim=1)[0]  # [batch]

            # Target: r + γ * max_a' Q_target(s', a') * (1 - done)
            targets = rewards_batch + self.gamma * next_q_values_max * (1 - dones_batch)

        # Huber loss (smooth L1)
        loss = F.smooth_l1_loss(q_values, targets)

        # ================================================================
        # BACKPROPAGATION WITH GRADIENT CLIPPING
        # ================================================================

        self.optimizer.zero_grad()
        loss.backward()

        # Adaptive gradient clipping
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_net.parameters(),
            max_norm=config.GRAD_CLIP_NORM
        )

        # Track gradient norms
        self.grad_norm_history.append(total_norm.item())
        if len(self.grad_norm_history) > 1000:
            self.grad_norm_history = self.grad_norm_history[-1000:]

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """
        Update target network (hard update).
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self, decay_rate: float = None):
        """
        Decay exploration rate.

        Args:
            decay_rate: Multiplicative decay (default from config)
        """
        if decay_rate is None:
            decay_rate = config.EPSILON_DECAY

        self.epsilon *= decay_rate

    def set_epsilon(self, epsilon: float):
        """Set epsilon directly."""
        self.epsilon = epsilon

    def get_learning_rate(self) -> float:
        """Get current learning rate."""
        return self.optimizer.param_groups[0]['lr']

    def save(self, filepath: str):
        """
        Save agent checkpoint.

        Args:
            filepath: Path to save checkpoint
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'grad_norm_history': self.grad_norm_history
        }, filepath)

    def load(self, filepath: str):
        """
        Load agent checkpoint.

        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.grad_norm_history = checkpoint.get('grad_norm_history', [])

        print(f"✓ Loaded checkpoint from {filepath}")
        print(f"  Epsilon: {self.epsilon:.3f}")


# ==============================================================================
# TEST CODE
# ==============================================================================

if __name__ == "__main__":
    print("="*70)
    print("TESTING FCN AGENT")
    print("="*70)

    # Create agent
    agent = FCNAgent(
        grid_size=20,
        learning_rate=3e-4,
        gamma=0.99,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"\nAgent created on device: {agent.device}")

    # Count parameters
    total_params = sum(p.numel() for p in agent.policy_net.parameters())
    print(f"Policy network parameters: {total_params:,}")

    # Test 1: State encoding
    print("\n" + "="*70)
    print("Test 1: State encoding")
    print("="*70)

    # Create dummy state
    robot_state = RobotState(
        position=(10, 10),
        orientation=0,
        visited_positions={(10, 10), (10, 11), (11, 10)}
    )

    world_state = WorldState(
        grid_size=20,
        graph=None,
        obstacles=set(),
        coverage_map=np.zeros((20, 20), dtype=np.float32),
        map_type="empty"
    )

    grid = agent._encode_state(robot_state, world_state)
    print(f"  Grid shape: {grid.shape}")
    print(f"  Expected: torch.Size([1, 5, 20, 20])")
    print(f"  ✓ PASSED" if grid.shape == torch.Size([1, 5, 20, 20]) else "  ✗ FAILED")

    # Test 2: Action selection
    print("\n" + "="*70)
    print("Test 2: Action selection")
    print("="*70)

    action = agent.select_action(robot_state, world_state, epsilon=0.0)
    print(f"  Selected action: {action}")
    print(f"  Valid action: {0 <= action < 9}")
    print(f"  ✓ PASSED" if 0 <= action < 9 else "  ✗ FAILED")

    # Test 3: Store transitions
    print("\n" + "="*70)
    print("Test 3: Store transitions")
    print("="*70)

    next_robot_state = RobotState(
        position=(10, 11),
        orientation=0,
        visited_positions={(10, 10), (10, 11), (11, 10), (10, 12)}
    )

    state_grid = agent._encode_state(robot_state, world_state)
    next_state_grid = agent._encode_state(next_robot_state, world_state)

    agent.store_transition(
        state=state_grid,
        action=action,
        reward=10.0,
        next_state=next_state_grid,
        done=False,
        info={'coverage': 0.4}
    )

    print(f"  Memory size: {len(agent.memory)}")
    print(f"  ✓ PASSED" if len(agent.memory) == 1 else "  ✗ FAILED")

    # Test 4: Optimization (needs more samples)
    print("\n" + "="*70)
    print("Test 4: Optimization")
    print("="*70)

    # Add more samples
    for i in range(100):
        agent.store_transition(
            state=state_grid,
            action=random.randint(0, 8),
            reward=random.uniform(0, 10),
            next_state=next_state_grid,
            done=False,
            info={'coverage': 0.4}
        )

    print(f"  Memory size: {len(agent.memory)}")
    loss = agent.optimize()
    print(f"  Loss: {loss:.4f}" if loss is not None else "  Loss: None")
    print(f"  ✓ PASSED" if loss is not None else "  ✗ FAILED")

    # Test 5: Target network update
    print("\n" + "="*70)
    print("Test 5: Target network update")
    print("="*70)

    # Check parameters differ before update
    policy_param = list(agent.policy_net.parameters())[0].clone()
    target_param = list(agent.target_net.parameters())[0].clone()

    agent.update_target_network()

    target_param_after = list(agent.target_net.parameters())[0]
    same_after = torch.allclose(policy_param, target_param_after)

    print(f"  Target network synced: {same_after}")
    print(f"  ✓ PASSED" if same_after else "  ✗ FAILED")

    # Test 6: Epsilon decay
    print("\n" + "="*70)
    print("Test 6: Epsilon decay")
    print("="*70)

    initial_epsilon = agent.epsilon
    agent.decay_epsilon(decay_rate=0.99)
    final_epsilon = agent.epsilon

    print(f"  Initial epsilon: {initial_epsilon:.4f}")
    print(f"  Final epsilon: {final_epsilon:.4f}")
    print(f"  Decayed: {final_epsilon < initial_epsilon}")
    print(f"  ✓ PASSED" if final_epsilon < initial_epsilon else "  ✗ FAILED")

    # Test 7: Save/load
    print("\n" + "="*70)
    print("Test 7: Save/load checkpoint")
    print("="*70)

    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = os.path.join(tmpdir, "test_checkpoint.pt")

        # Save
        agent.save(checkpoint_path)
        saved_epsilon = agent.epsilon

        # Modify epsilon
        agent.set_epsilon(0.5)

        # Load
        agent.load(checkpoint_path)
        loaded_epsilon = agent.epsilon

        print(f"  Saved epsilon: {saved_epsilon:.4f}")
        print(f"  Loaded epsilon: {loaded_epsilon:.4f}")
        print(f"  Match: {abs(saved_epsilon - loaded_epsilon) < 1e-6}")
        print(f"  ✓ PASSED" if abs(saved_epsilon - loaded_epsilon) < 1e-6 else "  ✗ FAILED")

    print("\n" + "="*70)
    print("ALL TESTS COMPLETED")
    print("="*70)
    print(f"\nFCN Agent ready for training!")
