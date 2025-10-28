"""
QMIX Agent for Multi-Agent Coverage

Implements QMIX (Monotonic Value Function Factorisation) for coordinated coverage.
Extends FCN architecture with centralized mixing network.

Key Features:
- Individual FCN Q-networks per agent (decentralized execution)
- Centralized mixing network (learns joint Q-function)
- Monotonicity constraint (ensures consistency)
- Grid-size invariant (uses Spatial Softmax)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from collections import deque

from fcn_spatial_network import FCNSpatialNetwork
from config import config


class QMixingNetwork(nn.Module):
    """
    QMIX Mixing Network
    
    Combines individual Q-values into joint Q-value using hypernetworks.
    Ensures monotonicity: ∂Q_tot/∂Q_i ≥ 0 for all i.
    """
    
    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.embed_dim = embed_dim
        
        # State encoder (processes global coverage map)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )
        
        # Hypernetwork for first mixing layer
        self.hyper_w1 = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents * 32)
        )
        self.hyper_b1 = nn.Linear(embed_dim, 32)
        
        # Hypernetwork for second mixing layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, individual_qs: torch.Tensor, global_state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values using global state.
        
        Args:
            individual_qs: [batch, num_agents] individual Q-values
            global_state: [batch, state_dim] flattened global coverage map
        
        Returns:
            q_tot: [batch] joint Q-value
        """
        batch_size = global_state.shape[0]
        
        # Encode global state
        state_embed = self.state_encoder(global_state)  # [batch, embed_dim]
        
        # Generate first layer weights (absolute value ensures monotonicity)
        w1 = torch.abs(self.hyper_w1(state_embed))
        w1 = w1.view(batch_size, self.num_agents, 32)
        b1 = self.hyper_b1(state_embed).view(batch_size, 1, 32)
        
        # First mixing layer: [batch, 1, num_agents] @ [batch, num_agents, 32] -> [batch, 1, 32]
        hidden = F.elu(torch.bmm(individual_qs.unsqueeze(1), w1) + b1)
        
        # Generate second layer weights (absolute value ensures monotonicity)
        w2 = torch.abs(self.hyper_w2(state_embed)).view(batch_size, 32, 1)
        b2 = self.hyper_b2(state_embed).view(batch_size, 1)
        
        # Second mixing layer: [batch, 1, 32] @ [batch, 32, 1] -> [batch, 1]
        q_tot = torch.bmm(hidden, w2).squeeze() + b2.squeeze()
        
        return q_tot


class MultiAgentReplayBuffer:
    """
    Replay buffer for multi-agent transitions.
    
    Stores joint experiences with both local and global information.
    """
    
    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, 
             local_obs: List[torch.Tensor],
             global_state: torch.Tensor,
             actions: List[int],
             reward: float,
             next_local_obs: List[torch.Tensor],
             next_global_state: torch.Tensor,
             done: bool):
        """Store a multi-agent transition."""
        self.buffer.append({
            'local_obs': local_obs,
            'global_state': global_state,
            'actions': actions,
            'reward': reward,
            'next_local_obs': next_local_obs,
            'next_global_state': next_global_state,
            'done': done
        })
    
    def sample(self, batch_size: int) -> dict:
        """Sample a batch of transitions."""
        batch = random.sample(self.buffer, batch_size)
        
        # Stack observations
        num_agents = len(batch[0]['local_obs'])
        
        local_obs = []
        next_local_obs = []
        for i in range(num_agents):
            local_obs.append(torch.stack([b['local_obs'][i] for b in batch]))
            next_local_obs.append(torch.stack([b['next_local_obs'][i] for b in batch]))
        
        return {
            'local_obs': local_obs,  # List of [batch, C, H, W]
            'global_state': torch.stack([b['global_state'] for b in batch]),
            'actions': torch.tensor([[b['actions'][i] for i in range(num_agents)] 
                                     for b in batch], dtype=torch.long),
            'rewards': torch.tensor([b['reward'] for b in batch], dtype=torch.float32),
            'next_local_obs': next_local_obs,
            'next_global_state': torch.stack([b['next_global_state'] for b in batch]),
            'dones': torch.tensor([b['done'] for b in batch], dtype=torch.float32)
        }
    
    def __len__(self):
        return len(self.buffer)


class QMIXAgent:
    """
    QMIX Agent for Multi-Agent Coverage
    
    Implements QMIX with FCN-based individual Q-networks.
    """
    
    def __init__(self, 
                 num_agents: int, 
                 grid_size: int = 20, 
                 input_channels: int = 5,
                 learning_rate: float = None,
                 gamma: float = None,
                 device: str = None):
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.input_channels = input_channels
        
        # Use config values if not provided
        if learning_rate is None:
            learning_rate = config.LEARNING_RATE
        if gamma is None:
            gamma = config.GAMMA
        if device is None:
            device = config.DEVICE
        
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.device = torch.device(device if isinstance(device, str) else 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Individual Q-networks (one per agent, same architecture as FCNAgent)
        self.agent_qnets = nn.ModuleList([
            FCNSpatialNetwork(
                input_channels=input_channels,  # 5 or 6 (with agent occupancy)
                num_actions=config.N_ACTIONS,
                hidden_dim=128
            )
            for _ in range(num_agents)
        ]).to(self.device)
        
        # QMIX mixing network (uses global state)
        state_dim = grid_size * grid_size * 3  # 3 channels: obstacles, coverage, visit_heat
        self.mixer = QMixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim
        ).to(self.device)
        
        # Target networks (for stable learning)
        self.target_qnets = nn.ModuleList([
            FCNSpatialNetwork(
                input_channels=input_channels,
                num_actions=config.N_ACTIONS,
                hidden_dim=128
            )
            for _ in range(num_agents)
        ]).to(self.device)
        self.target_mixer = QMixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim
        ).to(self.device)
        
        # Copy parameters to target networks
        self.update_target_networks(tau=1.0)
        
        # Replay buffer
        self.memory = MultiAgentReplayBuffer(capacity=config.REPLAY_BUFFER_SIZE)
        
        # Optimizer (optimizes all agent Q-networks + mixer jointly)
        self.optimizer = optim.Adam(
            list(self.agent_qnets.parameters()) + list(self.mixer.parameters()),
            lr=learning_rate
        )
        
        # Exploration
        self.epsilon = config.EPSILON_START
        
        # Training stats
        self.training_steps = 0
    
    def select_actions(self, 
                      observations: List[Dict],
                      epsilon: Optional[float] = None,
                      agent_occupancies: Optional[List] = None) -> List[int]:
        """
        Select actions for all agents (decentralized execution).
        
        Each agent uses only its own local observation and optional occupancy.
        
        Args:
            observations: List of observation dicts from environment
            epsilon: Override epsilon (default: use self.epsilon)
            agent_occupancies: Optional list of occupancy maps (one per agent)
        
        Returns:
            actions: List of selected actions
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if agent_occupancies is None:
            agent_occupancies = [None] * len(observations)
        
        # Convert observations to tensors (with optional occupancy)
        local_obs = self._observations_to_tensors(observations, agent_occupancies)
        
        actions = []
        
        for i, qnet in enumerate(self.agent_qnets):
            # Epsilon-greedy
            if random.random() < epsilon:
                # Random action
                action = random.randint(0, config.N_ACTIONS - 1)
            else:
                # Greedy action
                with torch.no_grad():
                    obs = local_obs[i].unsqueeze(0).to(self.device)
                    q_values = qnet(obs).squeeze(0)  # [n_actions]
                    action = q_values.argmax().item()
            
            actions.append(action)
        
        return actions
    
    def _observations_to_tensors(self, 
                                observations: List[Dict],
                                agent_occupancies: Optional[List] = None) -> List[torch.Tensor]:
        """
        Convert observation dicts to tensor format for Q-networks.
        
        Args:
            observations: List of observation dicts from environment
            agent_occupancies: Optional list of occupancy maps (one per agent)
            
        Returns:
            List of [C, H, W] tensors (C=5 or 6 depending on occupancy)
        """
        if agent_occupancies is None:
            agent_occupancies = [None] * len(observations)
        
        # Initialize encoder on first call (lazy initialization)
        if not hasattr(self, '_temp_encoder'):
            from fcn_agent import FCNAgent
            self._temp_encoder = FCNAgent(
                grid_size=self.grid_size,
                input_channels=self.input_channels,
                device=self.device
            )
        
        tensors = []
        for i, obs in enumerate(observations):
            robot_state = obs['robot_state']
            world_state = obs['world_state']
            occupancy = agent_occupancies[i]
            
            # Use the encoder's _encode_state method
            tensor = self._temp_encoder._encode_state(
                robot_state,
                world_state,
                agent_occupancy=occupancy
            )
            
            # Remove batch dimension [1, C, H, W] -> [C, H, W]
            tensor = tensor.squeeze(0)
            tensors.append(tensor)
        
        return tensors
    
    def _extract_global_state(self, state) -> torch.Tensor:
        """
        Extract global state tensor from MultiAgentState.
        
        Args:
            state: MultiAgentState object
            
        Returns:
            Flattened global state tensor
        """
        # Global state: obstacles + coverage + visit_heat
        # Shape: [grid_size, grid_size, 3] -> flatten to [3 * grid_size^2]
        obstacles = torch.FloatTensor(state.obstacles).unsqueeze(0)  # [1, H, W]
        coverage = torch.FloatTensor(state.coverage_map).unsqueeze(0)  # [1, H, W]
        visits = torch.FloatTensor(state.visit_heat_map).unsqueeze(0)  # [1, H, W]
        
        global_tensor = torch.cat([obstacles, coverage, visits], dim=0)  # [3, H, W]
        return global_tensor.flatten()  # [3 * H * W]
    
    def store_transition(self, 
                        observations: List[Dict],
                        actions: List[int],
                        rewards: List[float],
                        next_observations: List[Dict],
                        done: bool,
                        state: any,
                        next_state: any,
                        agent_occupancies: Optional[List] = None,
                        next_agent_occupancies: Optional[List] = None):
        """
        Store transition in replay buffer.
        
        Args:
            observations: List of observation dicts
            actions: List of actions taken
            rewards: List of rewards (per agent)
            next_observations: List of next observation dicts
            done: Episode done flag
            state: Current MultiAgentState
            next_state: Next MultiAgentState
            agent_occupancies: Optional list of current occupancy maps
            next_agent_occupancies: Optional list of next occupancy maps
        """
        # Convert observations to tensors (with optional occupancy)
        local_obs = self._observations_to_tensors(observations, agent_occupancies)
        next_local_obs = self._observations_to_tensors(next_observations, next_agent_occupancies)
        
        # Extract global states
        global_state = self._extract_global_state(state)
        next_global_state = self._extract_global_state(next_state)
        
        # Sum rewards for team reward (QMIX uses team reward)
        team_reward = sum(rewards)
        
        # Store in replay buffer
        self.memory.push(
            local_obs=local_obs,
            actions=actions,
            reward=team_reward,
            next_local_obs=next_local_obs,
            global_state=global_state,
            next_global_state=next_global_state,
            done=done
        )
    
    @property
    def replay_buffer(self):
        """Alias for memory to match train_qmix.py usage."""
        return self.memory
    
    def optimize(self) -> float:
        """
        Optimize QMIX using sampled batch.
        
        Returns:
            loss: TD loss value
        """
        if len(self.memory) < config.MIN_REPLAY_SIZE:
            return 0.0
        
        # Sample batch
        batch = self.memory.sample(config.BATCH_SIZE)
        batch_size = batch['rewards'].shape[0]
        
        # Move to device
        global_state = batch['global_state'].to(self.device)
        actions = batch['actions'].to(self.device)  # [batch, num_agents]
        rewards = batch['rewards'].to(self.device)  # [batch]
        next_global_state = batch['next_global_state'].to(self.device)
        dones = batch['dones'].to(self.device)  # [batch]
        
        # Compute current Q-values (individual)
        current_qs = []
        for i in range(self.num_agents):
            local_obs = batch['local_obs'][i].to(self.device)  # [batch, C, H, W]
            q_i = self.agent_qnets[i](local_obs)  # [batch, n_actions]
            q_i_selected = q_i.gather(1, actions[:, i].unsqueeze(1)).squeeze(1)  # [batch]
            current_qs.append(q_i_selected)
        
        current_qs = torch.stack(current_qs, dim=1)  # [batch, num_agents]
        
        # Mix current Q-values
        current_q_tot = self.mixer(current_qs, global_state)  # [batch]
        
        # Compute next Q-values (from target networks)
        with torch.no_grad():
            next_qs = []
            for i in range(self.num_agents):
                next_local_obs = batch['next_local_obs'][i].to(self.device)
                next_q_i = self.target_qnets[i](next_local_obs)  # [batch, n_actions]
                next_q_i_max = next_q_i.max(1)[0]  # [batch]
                next_qs.append(next_q_i_max)
            
            next_qs = torch.stack(next_qs, dim=1)  # [batch, num_agents]
            
            # Mix next Q-values
            next_q_tot = self.target_mixer(next_qs, next_global_state)  # [batch]
            
            # Compute target
            target_q = rewards + self.gamma * next_q_tot * (1 - dones)
        
        # TD loss
        loss = F.mse_loss(current_q_tot, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.agent_qnets.parameters()) + list(self.mixer.parameters()),
            config.GRAD_CLIP_NORM
        )
        
        self.optimizer.step()
        
        self.training_steps += 1
        
        return loss.item()
    
    def update_target_networks(self, tau: float = None):
        """
        Update target networks (soft or hard update).
        
        Args:
            tau: Soft update coefficient (1.0 = hard update)
        """
        if tau is None:
            tau = 0.005  # Default soft update
        
        # Update individual Q-networks
        for target_qnet, qnet in zip(self.target_qnets, self.agent_qnets):
            for target_param, param in zip(target_qnet.parameters(), qnet.parameters()):
                target_param.data.copy_(
                    tau * param.data + (1 - tau) * target_param.data
                )
        
        # Update mixer
        for target_param, param in zip(self.target_mixer.parameters(), 
                                       self.mixer.parameters()):
            target_param.data.copy_(
                tau * param.data + (1 - tau) * target_param.data
            )
    
    def set_epsilon(self, epsilon: float):
        """Set exploration rate."""
        self.epsilon = epsilon
    
    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay exploration rate."""
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * decay_rate)
    
    def save(self, path: str):
        """Save agent networks."""
        torch.save({
            'agent_qnets': [qnet.state_dict() for qnet in self.agent_qnets],
            'mixer': self.mixer.state_dict(),
            'target_qnets': [qnet.state_dict() for qnet in self.target_qnets],
            'target_mixer': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_steps': self.training_steps
        }, path)
    
    def load(self, path: str):
        """Load agent networks."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i, state_dict in enumerate(checkpoint['agent_qnets']):
            self.agent_qnets[i].load_state_dict(state_dict)
        
        self.mixer.load_state_dict(checkpoint['mixer'])
        
        for i, state_dict in enumerate(checkpoint['target_qnets']):
            self.target_qnets[i].load_state_dict(state_dict)
        
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_steps = checkpoint['training_steps']
