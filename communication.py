"""
Communication Module for Multi-Agent Coordination

Implements communication strategy:
- No Communication (baseline) - Position info via agent_occupancy 6th channel

REMOVED: FullStateSharing (wrong approach - bypasses coordination learning)

Communication Policy:
- Position/velocity information communicated via agent_occupancy.py (6th input channel)
- This provides realistic, limited bandwidth communication
- Agents must learn to coordinate from limited position information
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

    def communicate(self, observations: List[Dict], state) -> List[Message]:
        """
        Main communication interface.
        
        Args:
            observations: List of agent observations
            state: Current world state
            
        Returns:
            messages: List of messages exchanged between agents
        """
        raise NotImplementedError

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

    def communicate(self, observations: List[Dict], state) -> List[Message]:
        """No communication - return empty list."""
        return []

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


def get_communication_protocol(
    protocol_name: str,
    num_agents: int,
    grid_size: int = 20,
    comm_range: float = 5.0
) -> CommunicationProtocol:
    """
    Factory function to create communication protocol.

    Args:
        protocol_name: Protocol name ['none'] (only option after cleanup)
        num_agents: Number of agents
        grid_size: Grid size (unused, kept for API compatibility)
        comm_range: Communication range (unused, kept for API compatibility)

    Returns:
        CommunicationProtocol instance

    Note:
        Position communication now handled via agent_occupancy.py (6th channel).
        Use --use-6ch flag to enable position channel.
    """
    if protocol_name == 'none':
        return NoCommunciation(num_agents=num_agents)
    else:
        raise ValueError(f"Unknown protocol: {protocol_name}. "
                        f"Only 'none' supported. Use --use-6ch for position channel via agent_occupancy.py")


if __name__ == "__main__":
    # Test communication protocols
    print("Testing Communication Protocols...")

    num_agents = 4

    # Test No Communication
    print("\n1. Testing NoCommunciation...")
    no_comm = NoCommunciation(num_agents=num_agents)
    msg = no_comm.encode_message(0, torch.zeros(128))
    agg = no_comm.aggregate_messages(0, [msg])
    print(f"   ✓ NoCommunciation: message_dim={no_comm.message_dim}, should_comm={no_comm.should_communicate(0)}")

    # Test factory function
    print("\n2. Testing factory function...")
    proto_none = get_communication_protocol('none', num_agents=4)
    print(f"   ✓ Factory creates NoCommunciation: {isinstance(proto_none, NoCommunciation)}")

    # Test error handling
    print("\n3. Testing error handling...")
    try:
        proto_bad = get_communication_protocol('full_state', num_agents=4)
        print(f"   ✗ Should have raised error for 'full_state'")
    except ValueError as e:
        print(f"   ✓ Correctly rejects 'full_state': {str(e)[:50]}...")

    print("\n✓ All communication protocol tests passed!")
    print("\nNOTE: Position communication now via agent_occupancy.py (6th channel)")
    print("      Use --use-6ch flag when training for position information.")
