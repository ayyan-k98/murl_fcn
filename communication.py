"""
Communication Module for Multi-Agent Coordination

Implements several communication strategies:
1. No Communication (baseline)
2. Full State Sharing (upper bound)
3. CommNet-style (learned communication)
4. Attention-based Communication (scalable)
5. Targeted Communication (sparse, efficient)

Communication Policy:
- When to communicate: Every N steps, or event-triggered
- What to communicate: Map knowledge, intentions, beliefs
- Who to communicate with: All agents, or nearby only
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from config import config


@dataclass
class Message:
    """
    Message from one agent to another.
    
    Contains:
    - sender_id: Who sent the message
    - receiver_id: Who receives (None = broadcast)
    - content: Encoded message vector
    - metadata: Optional info (position, coverage, etc.)
    """
    sender_id: int
    receiver_id: Optional[int]
    content: torch.Tensor
    metadata: Dict


class CommunicationProtocol:
    """
    Base class for communication protocols.
    
    Defines interface for sending and receiving messages.
    """
    
    def __init__(self, num_agents: int, message_dim: int = 32):
        self.num_agents = num_agents
        self.message_dim = message_dim
    
    def encode_message(self, agent_id: int, hidden_state: torch.Tensor) -> Message:
        """Encode hidden state into message."""
        raise NotImplementedError
    
    def aggregate_messages(self, agent_id: int, messages: List[Message]) -> torch.Tensor:
        """Aggregate received messages."""
        raise NotImplementedError
    
    def should_communicate(self, step: int) -> bool:
        """Decide whether to communicate at this step."""
        raise NotImplementedError


class NoCommunciation(CommunicationProtocol):
    """Baseline: No communication between agents."""
    
    def encode_message(self, agent_id: int, hidden_state: torch.Tensor) -> Message:
        return Message(
            sender_id=agent_id,
            receiver_id=None,
            content=torch.zeros(self.message_dim),
            metadata={}
        )
    
    def aggregate_messages(self, agent_id: int, messages: List[Message]) -> torch.Tensor:
        return torch.zeros(self.message_dim)
    
    def should_communicate(self, step: int) -> bool:
        return False


class FullStateSharing(CommunicationProtocol):
    """
    Upper bound: Share full state information.
    
    Each agent broadcasts its complete local map and position.
    Not realistic (high bandwidth), but provides performance upper bound.
    """
    
    def __init__(self, num_agents: int, grid_size: int = 20):
        super().__init__(num_agents, message_dim=grid_size*grid_size + 2)
        self.grid_size = grid_size
    
    def encode_message(self, agent_id: int, local_map: torch.Tensor, 
                      position: Tuple[int, int]) -> Message:
        """
        Encode full local map + position.
        
        Args:
            local_map: [H, W] coverage map
            position: (x, y) position
        """
        # Flatten map and append position
        content = torch.cat([
            local_map.flatten(),
            torch.tensor([position[0], position[1]], dtype=torch.float32)
        ])
        
        return Message(
            sender_id=agent_id,
            receiver_id=None,  # Broadcast
            content=content,
            metadata={'position': position}
        )
    
    def aggregate_messages(self, agent_id: int, messages: List[Message]) -> Dict:
        """Merge all agents' local maps."""
        merged_maps = {}
        positions = {}
        
        for msg in messages:
            if msg.sender_id != agent_id:
                # Extract map and position
                map_flat = msg.content[:-2]
                pos = msg.content[-2:].numpy().astype(int)
                
                merged_maps[msg.sender_id] = map_flat.view(self.grid_size, self.grid_size)
                positions[msg.sender_id] = tuple(pos)
        
        return {
            'maps': merged_maps,
            'positions': positions
        }
    
    def should_communicate(self, step: int) -> bool:
        return step % 10 == 0  # Communicate every 10 steps


class CommNet(CommunicationProtocol):
    """
    CommNet: Learning Multiagent Communication with Backpropagation (NIPS 2016)
    
    Key idea: Average hidden states across agents.
    """
    
    def __init__(self, num_agents: int, hidden_dim: int = 128):
        super().__init__(num_agents, message_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        
        # Communication encoder/decoder
        self.encoder = nn.Linear(hidden_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, hidden_dim)
    
    def encode_message(self, agent_id: int, hidden_state: torch.Tensor) -> Message:
        """Encode hidden state."""
        content = self.encoder(hidden_state)
        
        return Message(
            sender_id=agent_id,
            receiver_id=None,  # Broadcast
            content=content,
            metadata={}
        )
    
    def aggregate_messages(self, agent_id: int, messages: List[Message]) -> torch.Tensor:
        """Average messages from other agents."""
        other_messages = [msg.content for msg in messages if msg.sender_id != agent_id]
        
        if len(other_messages) == 0:
            return torch.zeros(self.hidden_dim)
        
        # Average
        avg_message = torch.stack(other_messages).mean(dim=0)
        
        # Decode
        aggregated = self.decoder(avg_message)
        
        return aggregated
    
    def should_communicate(self, step: int) -> bool:
        return True  # Communicate every step (low overhead with averaging)


class AttentionComm(CommunicationProtocol):
    """
    Attention-based Communication
    
    Uses attention mechanism to selectively attend to relevant agents.
    More efficient than CommNet for large num_agents.
    """
    
    def __init__(self, num_agents: int, hidden_dim: int = 128):
        super().__init__(num_agents, message_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        
        # Attention mechanism
        self.query_net = nn.Linear(hidden_dim, hidden_dim)
        self.key_net = nn.Linear(hidden_dim, hidden_dim)
        self.value_net = nn.Linear(hidden_dim, hidden_dim)
    
    def encode_message(self, agent_id: int, hidden_state: torch.Tensor) -> Message:
        """Encode hidden state with key-value."""
        # Each message contains key and value
        key = self.key_net(hidden_state)
        value = self.value_net(hidden_state)
        
        content = torch.cat([key, value])
        
        return Message(
            sender_id=agent_id,
            receiver_id=None,
            content=content,
            metadata={}
        )
    
    def aggregate_messages(self, agent_id: int, messages: List[Message],
                          query: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages using attention.
        
        Args:
            agent_id: Receiving agent
            messages: List of messages from all agents
            query: [hidden_dim] query vector from receiving agent
        """
        other_messages = [msg for msg in messages if msg.sender_id != agent_id]
        
        if len(other_messages) == 0:
            return torch.zeros(self.hidden_dim)
        
        # Extract keys and values
        keys = []
        values = []
        for msg in other_messages:
            key = msg.content[:self.hidden_dim]
            value = msg.content[self.hidden_dim:]
            keys.append(key)
            values.append(value)
        
        keys = torch.stack(keys)  # [num_other, hidden_dim]
        values = torch.stack(values)  # [num_other, hidden_dim]
        
        # Compute attention weights
        query_proj = self.query_net(query)  # [hidden_dim]
        attn_scores = torch.matmul(keys, query_proj) / np.sqrt(self.hidden_dim)  # [num_other]
        attn_weights = F.softmax(attn_scores, dim=0)  # [num_other]
        
        # Weighted sum of values
        aggregated = torch.matmul(attn_weights, values)  # [hidden_dim]
        
        return aggregated
    
    def should_communicate(self, step: int) -> bool:
        return True  # Communicate every step


class TargetedComm(CommunicationProtocol):
    """
    Targeted Communication (TarMAC)
    
    Agents learn WHO to communicate with (sparse communication).
    More efficient for large teams.
    """
    
    def __init__(self, num_agents: int, hidden_dim: int = 128, 
                 comm_threshold: float = 0.5):
        super().__init__(num_agents, message_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.comm_threshold = comm_threshold
        
        # Signature network (decides who to communicate with)
        self.signature_net = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim)
        )
        
        # Message encoder
        self.message_net = nn.Linear(hidden_dim, hidden_dim)
    
    def compute_comm_mask(self, agent_id: int, signatures: List[torch.Tensor]) -> torch.Tensor:
        """
        Compute communication mask (who to listen to).
        
        Args:
            agent_id: Receiving agent
            signatures: List of signature vectors from all agents
        
        Returns:
            mask: [num_agents] binary mask (1 = communicate, 0 = ignore)
        """
        own_signature = signatures[agent_id]
        
        # Compute similarity with other agents
        similarities = []
        for i, sig in enumerate(signatures):
            if i == agent_id:
                similarities.append(0.0)  # Don't communicate with self
            else:
                sim = F.cosine_similarity(own_signature, sig, dim=0)
                similarities.append(sim.item())
        
        # Threshold to create binary mask
        mask = torch.tensor([1.0 if s > self.comm_threshold else 0.0 
                            for s in similarities])
        
        return mask
    
    def encode_message(self, agent_id: int, hidden_state: torch.Tensor) -> Message:
        """Encode message with signature."""
        # Compute signature (for deciding who to communicate with)
        signature = self.signature_net(hidden_state)
        
        # Encode message
        message = self.message_net(hidden_state)
        
        # Concatenate
        content = torch.cat([signature, message])
        
        return Message(
            sender_id=agent_id,
            receiver_id=None,
            content=content,
            metadata={'signature': signature}
        )
    
    def aggregate_messages(self, agent_id: int, messages: List[Message],
                          own_signature: torch.Tensor) -> torch.Tensor:
        """Aggregate messages using targeted communication."""
        other_messages = [msg for msg in messages if msg.sender_id != agent_id]
        
        if len(other_messages) == 0:
            return torch.zeros(self.hidden_dim)
        
        # Extract signatures and messages
        signatures = [msg.content[:self.hidden_dim] for msg in other_messages]
        msg_contents = [msg.content[self.hidden_dim:] for msg in other_messages]
        
        # Compute who to listen to
        comm_weights = []
        for sig in signatures:
            sim = F.cosine_similarity(own_signature, sig, dim=0)
            weight = torch.sigmoid((sim - self.comm_threshold) * 10)  # Soft gating
            comm_weights.append(weight)
        
        comm_weights = torch.stack(comm_weights)  # [num_other]
        msg_contents = torch.stack(msg_contents)  # [num_other, hidden_dim]
        
        # Weighted average (only communicate with similar agents)
        if comm_weights.sum() > 0:
            aggregated = torch.matmul(comm_weights, msg_contents) / (comm_weights.sum() + 1e-8)
        else:
            aggregated = torch.zeros(self.hidden_dim)
        
        return aggregated
    
    def should_communicate(self, step: int) -> bool:
        return step % 5 == 0  # Sparse communication


class CommunicationManager:
    """
    Manages communication between agents.
    
    Handles message passing, aggregation, and scheduling.
    """
    
    def __init__(self, num_agents: int, protocol: str = 'attention', 
                 hidden_dim: int = 128):
        self.num_agents = num_agents
        
        # Select protocol
        if protocol == 'none':
            self.protocol = NoCommunciation(num_agents)
        elif protocol == 'full':
            self.protocol = FullStateSharing(num_agents)
        elif protocol == 'commnet':
            self.protocol = CommNet(num_agents, hidden_dim)
        elif protocol == 'attention':
            self.protocol = AttentionComm(num_agents, hidden_dim)
        elif protocol == 'targeted':
            self.protocol = TargetedComm(num_agents, hidden_dim)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")
        
        self.message_buffer = []
        self.communication_count = 0
    
    def communicate(self, agent_id: int, hidden_state: torch.Tensor, 
                   step: int) -> Optional[Message]:
        """
        Send message from agent.
        
        Returns None if agent decides not to communicate.
        """
        if not self.protocol.should_communicate(step):
            return None
        
        message = self.protocol.encode_message(agent_id, hidden_state)
        self.message_buffer.append(message)
        self.communication_count += 1
        
        return message
    
    def receive(self, agent_id: int, query: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Receive and aggregate messages for agent.
        
        Args:
            agent_id: Receiving agent
            query: Query vector (for attention-based protocols)
        """
        if isinstance(self.protocol, AttentionComm) and query is not None:
            return self.protocol.aggregate_messages(agent_id, self.message_buffer, query)
        else:
            return self.protocol.aggregate_messages(agent_id, self.message_buffer)
    
    def reset(self):
        """Clear message buffer."""
        self.message_buffer = []
    
    def get_comm_stats(self) -> Dict:
        """Get communication statistics."""
        return {
            'total_messages': self.communication_count,
            'messages_per_agent': self.communication_count / self.num_agents,
            'protocol': self.protocol.__class__.__name__
        }


def get_communication_protocol(
    protocol_name: str,
    num_agents: int,
    grid_size: int,
    comm_range: float
) -> CommunicationProtocol:
    """
    Factory function to create communication protocol.
    
    Args:
        protocol_name: Protocol name ['none', 'full_state', 'commnet', 'attention', 'targeted']
        num_agents: Number of agents
        grid_size: Grid size
        comm_range: Communication range
        
    Returns:
        CommunicationProtocol instance
    """
    if protocol_name == 'none':
        return NoCommunciation(num_agents=num_agents)
    elif protocol_name == 'full_state':
        return FullStateSharing(num_agents=num_agents, grid_size=grid_size)
    elif protocol_name == 'commnet':
        # Hidden dim = 4 channels (robot_state) + grid_size^2 (world_state flattened)
        hidden_dim = 4 + grid_size * grid_size
        return CommNet(num_agents=num_agents, hidden_dim=hidden_dim)
    elif protocol_name == 'attention':
        hidden_dim = 4 + grid_size * grid_size
        return AttentionComm(num_agents=num_agents, hidden_dim=hidden_dim)
    elif protocol_name == 'targeted':
        return TargetedComm(num_agents=num_agents, grid_size=grid_size, comm_range=comm_range)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}. "
                        f"Choose from: none, full_state, commnet, attention, targeted")
