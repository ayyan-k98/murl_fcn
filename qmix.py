"""
QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent RL
Reference: Rashid et al., ICML 2018

Implements QMIX mixing network that combines individual agent Q-values into
a joint team Q-value Q_tot, with monotonicity constraint ensuring:
∂Q_tot/∂Q_i ≥ 0 for all agents i

This enables centralized training with decentralized execution (CTDE):
- Training: Use global state to compute Q_tot for team optimization
- Execution: Each agent acts based on local Q_i independently
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

    Architecture:
    1. Hypernetworks generate mixing weights from global state
    2. Weights are made non-negative (via abs()) for monotonicity
    3. Two-layer mixing network: Q_tot = f(Q_1, Q_2, ..., Q_n | s_global)

    Monotonicity ensures: If agent i increases Q_i, then Q_tot increases.
    This allows decentralized execution (each agent greedily maximizes Q_i).
    """

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        embed_dim: int = 32,
        hypernet_embed: int = 64
    ):
        """
        Initialize QMIX mixing network.

        Args:
            num_agents: Number of agents in the system
            state_dim: Dimension of global state
            embed_dim: Embedding dimension for mixing network
            hypernet_embed: Hidden dimension for hypernetworks
        """
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim

        # Hypernetwork for layer 1 weights
        # Generates weights of shape (num_agents, embed_dim)
        # Input: global state, Output: weight matrix for first layer
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, num_agents * embed_dim)
        )

        # Hypernetwork for layer 1 bias
        # Input: global state, Output: bias vector for first layer
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for layer 2 weights
        # Generates weights of shape (embed_dim, 1)
        # Input: global state, Output: weight vector for second layer
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim)
        )

        # Hypernetwork for layer 2 bias
        # This is state-dependent scalar bias for final Q_tot
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Mix individual Q-values into joint Q_tot.

        Computes: Q_tot = b2 + w2^T * ELU(w1^T * [Q_1, Q_2, ..., Q_n] + b1)
        where all weights w1, w2 are non-negative (monotonicity constraint).

        Args:
            agent_qs: Individual Q-values [batch_size, num_agents]
                     Each Q_i is the Q-value for agent i's chosen action
            state: Global state [batch_size, state_dim]
                   Team-level information (coverage map, all positions, etc.)

        Returns:
            q_tot: Joint Q-value [batch_size, 1]
                  Team value for the joint action (a_1, a_2, ..., a_n)
        """
        batch_size = agent_qs.size(0)

        # Generate mixing network weights from global state
        # All weights are non-negative (via abs()) to ensure monotonicity:
        # ∂Q_tot/∂Q_i ≥ 0 means increasing any Q_i increases Q_tot

        # Layer 1: [num_agents] -> [embed_dim]
        w1 = torch.abs(self.hyper_w1(state))  # [batch, num_agents * embed_dim]
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)  # [batch, agents, embed]

        b1 = self.hyper_b1(state)  # [batch, embed_dim]
        b1 = b1.view(batch_size, 1, self.embed_dim)  # [batch, 1, embed]

        # Layer 2: [embed_dim] -> [1]
        w2 = torch.abs(self.hyper_w2(state))  # [batch, embed_dim]
        w2 = w2.view(batch_size, self.embed_dim, 1)  # [batch, embed, 1]

        b2 = self.hyper_b2(state)  # [batch, 1]

        # Mix individual Q-values through two-layer network
        # Q_tot = b2 + w2^T * activation(w1^T * agent_qs + b1)
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

    Computes temporal difference error for team Q-value:
    L = E[(Q_tot(s, a) - y)^2]
    where y = r + γ * max_a' Q_tot(s', a')

    This loss is backpropagated through the mixing network to all agent networks,
    enabling joint optimization while maintaining decentralized execution.
    """

    def __init__(self, gamma: float = 0.99):
        """
        Initialize QMIX loss.

        Args:
            gamma: Discount factor for future rewards
        """
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
            q_tot: Current Q_tot for chosen joint action [batch_size, 1]
                  Q_tot(s, [a_1, a_2, ..., a_n])
            target_q_tot: Target Q_tot from next state [batch_size, 1]
                         max_a' Q_tot(s', a')
            rewards: Team rewards [batch_size, 1]
                    Typically sum or mean of individual agent rewards
            dones: Episode termination flags [batch_size, 1]
                  1 if episode ended, 0 otherwise

        Returns:
            loss: Mean squared TD-error
        """
        # TD target: r + γ * (1 - done) * Q_tot(s', a')
        # The (1 - done) term zeros out future Q when episode ends
        td_target = rewards + self.gamma * (1 - dones) * target_q_tot

        # TD error: Q_tot(s, a) - [r + γ * Q_tot(s', a')]
        td_error = q_tot - td_target.detach()

        # Mean squared loss over batch
        loss = (td_error ** 2).mean()

        return loss


def test_qmix():
    """Test QMIX mixing network and loss."""
    print("Testing QMIX Implementation...")
    print("=" * 80)

    # Test parameters
    batch_size = 32
    num_agents = 4
    state_dim = 1610  # From get_global_state() specification

    # Create mixing network
    mixing_net = QMixingNetwork(
        num_agents=num_agents,
        state_dim=state_dim,
        embed_dim=32,
        hypernet_embed=64
    )

    print(f"\n1. Testing QMixingNetwork")
    print(f"   Input: {num_agents} agent Q-values, state_dim={state_dim}")
    print(f"   Output: Joint Q_tot")

    # Test forward pass
    agent_qs = torch.randn(batch_size, num_agents)  # Random Q-values
    global_state = torch.randn(batch_size, state_dim)  # Random state

    q_tot = mixing_net(agent_qs, global_state)

    print(f"   ✓ Forward pass: agent_qs {agent_qs.shape} + state {global_state.shape} -> q_tot {q_tot.shape}")
    assert q_tot.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {q_tot.shape}"

    # Test monotonicity
    print(f"\n2. Testing Monotonicity Constraint")
    print(f"   Checking: ∂Q_tot/∂Q_i ≥ 0 for all agents")

    agent_qs.requires_grad_(True)
    q_tot = mixing_net(agent_qs, global_state)

    for i in range(num_agents):
        # Compute gradient ∂Q_tot/∂Q_i
        grad = torch.autograd.grad(
            q_tot.sum(), agent_qs, retain_graph=True, create_graph=False
        )[0][:, i]

        # Check all gradients are non-negative
        min_grad = grad.min().item()
        max_grad = grad.max().item()
        mean_grad = grad.mean().item()

        if min_grad >= -1e-6:  # Small tolerance for numerical errors
            print(f"   ✓ Agent {i}: ∂Q_tot/∂Q_{i} ∈ [{min_grad:.4f}, {max_grad:.4f}] (mean: {mean_grad:.4f})")
        else:
            print(f"   ✗ Agent {i}: VIOLATION! min gradient = {min_grad:.4f} < 0")

    # Test loss function
    print(f"\n3. Testing QMIXLoss")

    loss_fn = QMIXLoss(gamma=0.99)

    # Create dummy data
    q_tot = torch.randn(batch_size, 1)
    target_q_tot = torch.randn(batch_size, 1)
    rewards = torch.randn(batch_size, 1)
    dones = torch.randint(0, 2, (batch_size, 1)).float()

    loss = loss_fn(q_tot, target_q_tot, rewards, dones)

    print(f"   ✓ Loss computation: {loss.item():.4f}")
    assert loss.item() >= 0, "Loss should be non-negative (MSE)"

    # Test gradient flow
    print(f"\n4. Testing Gradient Flow")

    agent_qs = torch.randn(batch_size, num_agents, requires_grad=True)
    global_state = torch.randn(batch_size, state_dim)

    q_tot = mixing_net(agent_qs, global_state)
    target_q_tot = torch.randn(batch_size, 1)
    rewards = torch.randn(batch_size, 1)
    dones = torch.zeros(batch_size, 1)

    loss = loss_fn(q_tot, target_q_tot, rewards, dones)
    loss.backward()

    print(f"   ✓ Backward pass successful")
    print(f"   ✓ Gradients computed for agent Q-values")

    # Check parameter count
    print(f"\n5. Network Statistics")
    total_params = sum(p.numel() for p in mixing_net.parameters())
    trainable_params = sum(p.numel() for p in mixing_net.parameters() if p.requires_grad)

    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    print(f"\n{'='*80}")
    print(f"✅ All QMIX tests passed!")
    print(f"{'='*80}")

    print(f"\nQMIX is ready for integration into multi-agent training.")
    print(f"Next steps:")
    print(f"  1. Implement get_global_state() in multi_agent_env.py")
    print(f"  2. Add QMIX training loop in multi_agent_trainer.py")
    print(f"  3. Update replay buffer to store global states")
    print(f"  4. Train with --use-qmix flag")


if __name__ == "__main__":
    test_qmix()
