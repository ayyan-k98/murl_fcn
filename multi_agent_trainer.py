"""
Multi-Agent Trainer

Implements CTDE (Centralized Training, Decentralized Execution) for multi-agent coverage.

Key Features:
- Multiple independent FCN agents
- Shared or separate replay memories
- Coordinated training with team rewards
- Curriculum learning for multi-agent scenarios
- Validation across different team sizes
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from config import config
from fcn_agent import FCNAgent
from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy, MultiAgentState
from replay_memory import StratifiedReplayMemory
from coordination_metrics import CoordinationAnalyzer, CoordinationMetrics, coordination_score


class MultiAgentTrainer:
    """
    CTDE trainer for multi-agent coverage.

    Training Modes:
        - Parameter Sharing: All agents share the same network (default)
        - Independent: Each agent has independent network
        - Hybrid: Shared encoder, independent heads

    Replay Memory:
        - Shared: Single replay buffer for all agents
        - Separate: Individual replay buffers per agent
    """

    def __init__(
        self,
        num_agents: int = 4,
        grid_size: int = 20,
        coordination: CoordinationStrategy = CoordinationStrategy.INDEPENDENT,
        parameter_sharing: bool = True,
        shared_replay: bool = True,
        input_channels: int = 6,  # FIXED: Default to 6 (adds agent occupancy channel)
        learning_rate: float = None,
        gamma: float = None,
        device: str = None
    ):
        """
        Initialize multi-agent trainer.

        Args:
            num_agents: Number of agents
            grid_size: Grid dimension
            coordination: Coordination strategy
            parameter_sharing: If True, all agents share same network
            shared_replay: If True, use single replay buffer
            input_channels: Number of input channels (5 or 6)
            learning_rate: Learning rate (default from config)
            gamma: Discount factor (default from config)
            device: Compute device (default from config)
        """
        self.num_agents = num_agents
        self.grid_size = grid_size
        self.coordination = coordination
        self.parameter_sharing = parameter_sharing
        self.shared_replay = shared_replay
        self.input_channels = input_channels

        # Device
        self.device = device or config.DEVICE

        # Learning parameters
        self.learning_rate = learning_rate or config.LEARNING_RATE
        self.gamma = gamma or config.GAMMA

        # Create agents
        if parameter_sharing:
            # Single shared agent
            self.shared_agent = FCNAgent(
                grid_size=grid_size,
                learning_rate=self.learning_rate,
                gamma=self.gamma,
                device=self.device,
                input_channels=input_channels
            )
            self.agents = [self.shared_agent] * num_agents
            print(f"✓ Using parameter sharing (1 network for {num_agents} agents, {input_channels} channels)")
        else:
            # Independent agents
            self.agents = [
                FCNAgent(
                    grid_size=grid_size,
                    learning_rate=self.learning_rate,
                    gamma=self.gamma,
                    device=self.device,
                    input_channels=input_channels
                )
                for _ in range(num_agents)
            ]
            print(f"✓ Using independent networks ({num_agents} agents, {input_channels} channels)")

        # Replay memory
        if shared_replay:
            # Single shared buffer
            self.shared_memory = StratifiedReplayMemory(
                capacity=config.REPLAY_BUFFER_SIZE
            )
            self.agent_memories = [self.shared_memory] * num_agents
            print(f"✓ Using shared replay memory")
        else:
            # Separate buffers
            self.agent_memories = [
                StratifiedReplayMemory(capacity=config.REPLAY_BUFFER_SIZE)
                for _ in range(num_agents)
            ]
            print(f"✓ Using separate replay memories")

        # Epsilon (exploration)
        self.epsilon = config.EPSILON_START

        # Training metrics
        self.metrics = {
            'episode_rewards': [],
            'team_rewards': [],
            'team_coverages': [],
            'episode_lengths': [],
            'losses': defaultdict(list),
            'collisions': [],
            'agent_collisions': [],
            'coordination_scores': []
        }

    def select_actions(
        self,
        observations: List[Dict],
        epsilon: Optional[float] = None,
        agent_occupancies: Optional[List] = None
    ) -> List[int]:
        """
        Select actions for all agents (decentralized execution).

        Args:
            observations: List of observation dicts (one per agent)
            epsilon: Override epsilon (default: use self.epsilon)
            agent_occupancies: Optional list of occupancy maps (one per agent)

        Returns:
            actions: List of actions [action_0, ..., action_n-1]
        """
        if epsilon is None:
            epsilon = self.epsilon
        
        if agent_occupancies is None:
            agent_occupancies = [None] * self.num_agents

        actions = []

        for i, obs in enumerate(observations):
            agent = self.agents[i]
            robot_state = obs['robot_state']
            world_state = obs['world_state']
            occupancy = agent_occupancies[i]

            action = agent.select_action(
                robot_state, 
                world_state, 
                epsilon=epsilon,
                agent_occupancy=occupancy
            )
            actions.append(action)

        return actions

    def store_transitions(
        self,
        observations: List[Dict],
        actions: List[int],
        rewards: List[float],
        next_observations: List[Dict],
        done: bool,
        info: Dict,
        agent_occupancies: Optional[List] = None,
        next_agent_occupancies: Optional[List] = None
    ):
        """
        Store transitions for all agents.

        Args:
            observations: List of current observations
            actions: List of actions
            rewards: List of rewards
            next_observations: List of next observations
            done: Episode done flag
            info: Info dict from environment
            agent_occupancies: Optional list of current occupancy maps
            next_agent_occupancies: Optional list of next occupancy maps
        """
        if agent_occupancies is None:
            agent_occupancies = [None] * self.num_agents
        if next_agent_occupancies is None:
            next_agent_occupancies = [None] * self.num_agents

        for i in range(self.num_agents):
            # Extract states
            robot_state = observations[i]['robot_state']
            world_state = observations[i]['world_state']
            next_robot_state = next_observations[i]['robot_state']

            # Encode states (with optional occupancy)
            agent = self.agents[i]
            state_tensor = agent._encode_state(
                robot_state, 
                world_state,
                agent_occupancy=agent_occupancies[i]
            )
            next_state_tensor = agent._encode_state(
                next_robot_state, 
                world_state,
                agent_occupancy=next_agent_occupancies[i]
            )

            # Store in replay memory
            transition_info = {
                'coverage_gain': info['individual_coverage_gains'][i],
                'knowledge_gain': info['knowledge_gains'][i],
                'collision': info['collisions'][i]
            }

            self.agent_memories[i].push(
                state_tensor,
                actions[i],
                rewards[i],
                next_state_tensor,
                done,
                transition_info
            )

    def optimize_agents(self) -> Dict[int, Optional[float]]:
        """
        Optimize all agents (centralized training).

        Returns:
            losses: Dict mapping agent_id -> loss (None if not trained)
        """
        losses = {}

        if self.parameter_sharing:
            # Single update for shared network
            if len(self.agent_memories[0]) >= config.MIN_REPLAY_SIZE:
                loss = self.agents[0].optimize()
                losses[0] = loss

                # Track metrics
                if loss is not None:
                    self.metrics['losses'][0].append(loss)
            else:
                losses[0] = None

        else:
            # Independent updates
            for i in range(self.num_agents):
                if len(self.agent_memories[i]) >= config.MIN_REPLAY_SIZE:
                    loss = self.agents[i].optimize()
                    losses[i] = loss

                    # Track metrics
                    if loss is not None:
                        self.metrics['losses'][i].append(loss)
                else:
                    losses[i] = None

        return losses

    def update_target_networks(self):
        """Update target networks for all agents."""
        if self.parameter_sharing:
            self.agents[0].update_target_network()
        else:
            for agent in self.agents:
                agent.update_target_network()

    def decay_epsilon(self, decay_rate: float = 0.995):
        """Decay epsilon for all agents."""
        self.epsilon = max(config.EPSILON_MIN, self.epsilon * decay_rate)

        # Update agent epsilons
        if self.parameter_sharing:
            self.agents[0].set_epsilon(self.epsilon)
        else:
            for agent in self.agents:
                agent.set_epsilon(self.epsilon)

    def set_epsilon(self, epsilon: float):
        """Set epsilon for all agents."""
        self.epsilon = epsilon

        if self.parameter_sharing:
            self.agents[0].set_epsilon(epsilon)
        else:
            for agent in self.agents:
                agent.set_epsilon(epsilon)

    def train_episode(
        self,
        env: MultiAgentCoverageEnv,
        map_type: Optional[str] = None,
        comm_manager=None,
        occupancy_computer=None
    ) -> Dict:
        """
        Train for one episode.

        Args:
            env: Multi-agent environment
            map_type: Map type for this episode
            comm_manager: Communication manager (optional)
            occupancy_computer: Agent occupancy computer (optional)

        Returns:
            episode_info: Dict with episode metrics
        """
        # Reset environment
        state = env.reset(map_type=map_type)
        observations = env.get_observations()

        episode_rewards = [0.0] * self.num_agents
        team_reward = 0.0
        step_count = 0
        episode_collisions = 0
        episode_agent_collisions = 0
        
        # Initialize coordination tracking
        coord_analyzer = CoordinationAnalyzer(self.num_agents, env.grid_size)

        done = False

        while not done:
            # Communication phase (only if enabled)
            messages = None
            if comm_manager is not None:
                # Collect agent states for communication
                agent_states = []
                for i, obs in enumerate(observations):
                    agent_states.append({
                        'agent_id': i,
                        'position': obs['robot_state'].position,
                        'visited': obs['robot_state'].visited_positions,
                        'timestamp': step_count
                    })
                
                # Exchange messages
                messages = comm_manager.communicate(observations, state)
            
            # Compute agent occupancies (only if using 6 channels AND communication enabled)
            agent_occupancies = None
            if occupancy_computer is not None and messages is not None:
                agent_occupancies = [
                    occupancy_computer.compute(i, messages, step_count)
                    for i in range(self.num_agents)
                ]
            
            # Select actions (with optional occupancy)
            actions = self.select_actions(
                observations, 
                epsilon=self.epsilon,
                agent_occupancies=agent_occupancies
            )

            # Execute actions
            next_state, rewards, done, info = env.step(actions)
            next_observations = env.get_observations()

            # Compute next occupancies (only if using 6 channels AND communication enabled)
            next_agent_occupancies = None
            if occupancy_computer is not None and messages is not None:
                # Update messages with new positions
                next_agent_states = []
                for i, obs in enumerate(next_observations):
                    next_agent_states.append({
                        'agent_id': i,
                        'position': obs['robot_state'].position,
                        'visited': obs['robot_state'].visited_positions,
                        'timestamp': step_count + 1
                    })
                
                # Recompute messages for next state
                next_messages = comm_manager.communicate(next_observations, next_state)
                
                next_agent_occupancies = [
                    occupancy_computer.compute(i, next_messages, step_count + 1)
                    for i in range(self.num_agents)
                ]

            # Store transitions (with optional occupancies)
            self.store_transitions(
                observations,
                actions,
                rewards,
                next_observations,
                done,
                info,
                agent_occupancies=agent_occupancies,
                next_agent_occupancies=next_agent_occupancies
            )

            # Optimize agents
            if step_count % config.TRAIN_FREQ == 0:
                losses = self.optimize_agents()

            # Update target networks
            if step_count % config.TARGET_UPDATE_FREQ == 0:
                self.update_target_networks()

            # Accumulate rewards
            for i in range(self.num_agents):
                episode_rewards[i] += rewards[i]

            team_reward += sum(rewards)

            # Track collisions
            episode_collisions += sum(info['collisions'])
            episode_agent_collisions += sum(info['agent_collisions'])
            
            # Update coordination metrics
            # Use each agent's own coverage_history (not shared world_state coverage_map)
            agent_positions = [obs['robot_state'].position for obs in next_observations]
            visited_maps = [np.array(obs['robot_state'].coverage_history > 0, dtype=bool) for obs in next_observations]
            coord_analyzer.update(
                positions=agent_positions,
                visited_maps=visited_maps,
                actions=actions,
                messages=messages
            )

            # Update for next step
            observations = next_observations
            step_count += 1

        # Episode metrics
        final_coverage = info['coverage_pct']
        coord_metrics = coord_analyzer.get_metrics()
        coord_score_val = coordination_score(coord_metrics)

        episode_info = {
            'episode_rewards': episode_rewards,
            'team_reward': team_reward,
            'team_coverage': final_coverage,
            'episode_length': step_count,
            'collisions': episode_collisions,
            'agent_collisions': episode_agent_collisions,
            'epsilon': self.epsilon,
            'coordination_score': coord_score_val,
            'coordination_metrics': coord_metrics
        }

        # Track metrics
        self.metrics['episode_rewards'].append(episode_rewards)
        self.metrics['team_rewards'].append(team_reward)
        self.metrics['team_coverages'].append(final_coverage)
        self.metrics['episode_lengths'].append(step_count)
        self.metrics['collisions'].append(episode_collisions)
        self.metrics['agent_collisions'].append(episode_agent_collisions)
        self.metrics['coordination_scores'].append(coord_score_val)

        return episode_info

    def validate(
        self,
        env: MultiAgentCoverageEnv,
        num_episodes: int = 10,
        map_types: Optional[List[str]] = None,
        comm_manager=None,
        occupancy_computer=None
    ) -> Dict:
        """
        Validate trained agents.

        Args:
            env: Multi-agent environment
            num_episodes: Number of validation episodes
            map_types: List of map types to test (default: all types)
            comm_manager: Communication manager (optional)
            occupancy_computer: Agent occupancy computer (optional)

        Returns:
            validation_results: Dict with validation metrics
        """
        if map_types is None:
            map_types = ["empty", "random", "room", "corridor", "cave"]

        # Store current epsilon
        original_epsilon = self.epsilon

        # Set greedy mode
        self.set_epsilon(0.0)

        results = {
            'coverages': [],
            'team_rewards': [],
            'lengths': [],
            'collisions': [],
            'coordination_scores': [],
            'per_map_type': defaultdict(list)
        }

        for ep in range(num_episodes):
            map_type = map_types[ep % len(map_types)]

            # Reset and run episode
            state = env.reset(map_type=map_type)
            observations = env.get_observations()

            team_reward = 0.0
            step_count = 0
            episode_collisions = 0
            episode_agent_collisions = 0
            
            # Initialize coordination tracking
            coord_analyzer = CoordinationAnalyzer(self.num_agents, env.grid_size)
            
            done = False

            while not done:
                # Communication phase (only if enabled)
                messages = None
                if comm_manager is not None:
                    messages = comm_manager.communicate(observations, state)
                
                # Compute agent occupancies (only if using 6 channels AND communication enabled)
                agent_occupancies = None
                if occupancy_computer is not None and messages is not None:
                    agent_occupancies = [
                        occupancy_computer.compute(i, messages, step_count)
                        for i in range(self.num_agents)
                    ]
                
                # Select actions
                actions = self.select_actions(
                    observations, 
                    epsilon=0.0,
                    agent_occupancies=agent_occupancies
                )
                
                next_state, rewards, done, info = env.step(actions)
                next_observations = env.get_observations()

                team_reward += sum(rewards)
                episode_collisions += sum(info['collisions'])
                episode_agent_collisions += sum(info['agent_collisions'])
                
                # Update coordination metrics
                # Use each agent's own coverage_history (not shared world_state coverage_map)
                agent_positions = [obs['robot_state'].position for obs in next_observations]
                visited_maps = [np.array(obs['robot_state'].coverage_history > 0, dtype=bool) for obs in next_observations]
                coord_analyzer.update(
                    positions=agent_positions,
                    visited_maps=visited_maps,
                    actions=actions,
                    messages=messages
                )
                
                step_count += 1
                observations = next_observations

            final_coverage = info['coverage_pct']
            coord_metrics = coord_analyzer.get_metrics()
            coord_score_val = coordination_score(coord_metrics)

            # Record results
            results['coverages'].append(final_coverage)
            results['team_rewards'].append(team_reward)
            results['lengths'].append(step_count)
            results['collisions'].append(episode_collisions)
            results['coordination_scores'].append(coord_score_val)
            results['per_map_type'][map_type].append(final_coverage)

        # Restore epsilon
        self.set_epsilon(original_epsilon)

        # Compute statistics
        validation_stats = {
            'mean_coverage': np.mean(results['coverages']),
            'std_coverage': np.std(results['coverages']),
            'mean_team_reward': np.mean(results['team_rewards']),
            'mean_length': np.mean(results['lengths']),
            'mean_collisions': np.mean(results['collisions']),
            'mean_coordination_score': np.mean(results['coordination_scores']),
            'std_coordination_score': np.std(results['coordination_scores']),
            'per_map_coverage': {
                map_type: np.mean(coverages)
                for map_type, coverages in results['per_map_type'].items()
            }
        }

        return validation_stats

    def save(self, filepath: str):
        """
        Save trainer state.

        Args:
            filepath: Path to save file
        """
        save_dict = {
            'num_agents': self.num_agents,
            'grid_size': self.grid_size,
            'coordination': self.coordination.value,
            'parameter_sharing': self.parameter_sharing,
            'epsilon': self.epsilon,
            'metrics': self.metrics
        }

        # Save agent networks
        if self.parameter_sharing:
            save_dict['shared_agent'] = self.agents[0].policy_net.state_dict()
        else:
            save_dict['agents'] = [
                agent.policy_net.state_dict() for agent in self.agents
            ]

        torch.save(save_dict, filepath)
        print(f"✓ Saved multi-agent trainer to {filepath}")

    def load(self, filepath: str):
        """
        Load trainer state.

        Args:
            filepath: Path to load file
        """
        checkpoint = torch.load(filepath, map_location=self.device)

        # Load agent networks
        if self.parameter_sharing:
            self.agents[0].policy_net.load_state_dict(checkpoint['shared_agent'])
            self.agents[0].target_net.load_state_dict(checkpoint['shared_agent'])
        else:
            for i, agent_state in enumerate(checkpoint['agents']):
                self.agents[i].policy_net.load_state_dict(agent_state)
                self.agents[i].target_net.load_state_dict(agent_state)

        # Restore epsilon
        self.epsilon = checkpoint.get('epsilon', config.EPSILON_START)
        self.set_epsilon(self.epsilon)

        print(f"✓ Loaded multi-agent trainer from {filepath}")

    def get_training_stats(self, window: int = 100) -> Dict:
        """
        Get recent training statistics.

        Args:
            window: Window size for moving average

        Returns:
            stats: Dict with training statistics
        """
        if len(self.metrics['team_rewards']) == 0:
            return {}

        recent_rewards = self.metrics['team_rewards'][-window:]
        recent_coverages = self.metrics['team_coverages'][-window:]
        recent_lengths = self.metrics['episode_lengths'][-window:]
        recent_collisions = self.metrics['collisions'][-window:]
        recent_coord_scores = self.metrics['coordination_scores'][-window:] if self.metrics['coordination_scores'] else []

        stats = {
            'mean_team_reward': np.mean(recent_rewards),
            'mean_coverage': np.mean(recent_coverages),
            'mean_length': np.mean(recent_lengths),
            'mean_collisions': np.mean(recent_collisions),
            'epsilon': self.epsilon,
            'total_episodes': len(self.metrics['team_rewards'])
        }
        
        # Add coordination score if available
        if recent_coord_scores:
            stats['mean_coordination_score'] = np.mean(recent_coord_scores)

        # Per-agent losses (if available)
        if self.parameter_sharing:
            if len(self.metrics['losses'][0]) > 0:
                stats['mean_loss'] = np.mean(self.metrics['losses'][0][-window:])
        else:
            for i in range(self.num_agents):
                if len(self.metrics['losses'][i]) > 0:
                    stats[f'mean_loss_agent_{i}'] = np.mean(
                        self.metrics['losses'][i][-window:]
                    )

        return stats


if __name__ == "__main__":
    print("Testing MultiAgentTrainer...")

    # Create environment
    env = MultiAgentCoverageEnv(
        num_agents=4,
        grid_size=20,
        coordination=CoordinationStrategy.INDEPENDENT
    )

    # Create trainer (parameter sharing)
    trainer = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        coordination=CoordinationStrategy.INDEPENDENT,
        parameter_sharing=True,
        shared_replay=True
    )

    print(f"\n✓ MultiAgentTrainer initialized")
    print(f"  Parameter sharing: {trainer.parameter_sharing}")
    print(f"  Shared replay: {trainer.shared_replay}")
    print(f"  Epsilon: {trainer.epsilon}")

    # Test episode
    print(f"\n✓ Running test episode...")
    episode_info = trainer.train_episode(env, map_type="empty")

    print(f"  Team reward: {episode_info['team_reward']:.2f}")
    print(f"  Team coverage: {episode_info['team_coverage']*100:.1f}%")
    print(f"  Episode length: {episode_info['episode_length']}")
    print(f"  Collisions: {episode_info['collisions']}")
    print(f"  Agent collisions: {episode_info['agent_collisions']}")

    # Test validation
    print(f"\n✓ Running validation (5 episodes)...")
    val_results = trainer.validate(env, num_episodes=5)

    print(f"  Mean coverage: {val_results['mean_coverage']*100:.1f}%")
    print(f"  Mean team reward: {val_results['mean_team_reward']:.2f}")
    print(f"  Mean collisions: {val_results['mean_collisions']:.1f}")

    print(f"\n✓ MultiAgentTrainer test complete")
