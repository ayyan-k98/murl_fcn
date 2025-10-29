"""
Communication Module for Multi-Agent Coordination

Implements two communication strategies:
1. No Communication (baseline)
2. Full State Sharing (upper bound)

Communication Policy:
- When to communicate: Every N steps
- What to communicate: Map knowledge and positions
- Who to communicate with: All agents (broadcast)
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


def get_communication_protocol(
    protocol_name: str,
    num_agents: int,
    grid_size: int = 20,
    comm_range: float = 5.0
) -> CommunicationProtocol:
    """
    Factory function to create communication protocol.

    Args:
        protocol_name: Protocol name ['none', 'full_state']
        num_agents: Number of agents
        grid_size: Grid size
        comm_range: Communication range (unused, for API compatibility)

    Returns:
        CommunicationProtocol instance
    """
    if protocol_name == 'none':
        return NoCommunciation(num_agents=num_agents)
    elif protocol_name == 'full_state':
        return FullStateSharing(num_agents=num_agents, grid_size=grid_size)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}. "
                        f"Choose from: none, full_state")


if __name__ == "__main__":
    # Test communication protocols
    print("Testing Communication Protocols...")

    num_agents = 4
    grid_size = 20

    # Test No Communication
    print("\n1. Testing NoCommunciation...")
    no_comm = NoCommunciation(num_agents=num_agents)
    msg = no_comm.encode_message(0, torch.zeros(128))
    agg = no_comm.aggregate_messages(0, [msg])
    print(f"   ✓ NoCommunciation: message_dim={no_comm.message_dim}, should_comm={no_comm.should_communicate(0)}")

    # Test Full State Sharing
    print("\n2. Testing FullStateSharing...")
    full_state = FullStateSharing(num_agents=num_agents, grid_size=grid_size)
    local_map = torch.rand(grid_size, grid_size)
    position = (10, 10)
    msg = full_state.encode_message(0, local_map, position)
    print(f"   ✓ FullStateSharing: message_dim={full_state.message_dim}, should_comm={full_state.should_communicate(10)}")
    print(f"   ✓ Message content size: {msg.content.shape}")
    print(f"   ✓ Position encoded: {msg.metadata['position']}")

    # Test factory function
    print("\n3. Testing factory function...")
    proto_none = get_communication_protocol('none', num_agents=4)
    proto_full = get_communication_protocol('full_state', num_agents=4, grid_size=20)
    print(f"   ✓ Factory creates NoCommunciation: {isinstance(proto_none, NoCommunciation)}")
    print(f"   ✓ Factory creates FullStateSharing: {isinstance(proto_full, FullStateSharing)}")

    print("\n✓ All communication protocol tests passed!")
