"""
Multi-Agent Visualization Utilities

Tools for visualizing multi-agent coverage performance and behavior.

Key Features:
    - Real-time episode rendering
    - Training metrics plots
    - Team coordination visualization
    - Coverage heatmaps
    - Trajectory analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import colors
from typing import List, Dict, Optional
import os

from multi_agent_env import MultiAgentCoverageEnv, MultiAgentState, CoordinationStrategy


class MultiAgentVisualizer:
    """Visualize multi-agent coverage episodes and training metrics."""

    # Agent colors (distinct for each agent)
    AGENT_COLORS = [
        '#FF4136',  # Red
        '#0074D9',  # Blue
        '#2ECC40',  # Green
        '#FF851B',  # Orange
        '#B10DC9',  # Purple
        '#FFDC00',  # Yellow
        '#39CCCC',  # Teal
        '#F012BE'   # Magenta
    ]

    def __init__(self, grid_size: int = 20):
        """
        Initialize visualizer.

        Args:
            grid_size: Grid dimension
        """
        self.grid_size = grid_size

    def render_episode_state(
        self,
        env: MultiAgentCoverageEnv,
        state: MultiAgentState,
        title: str = "Multi-Agent Coverage",
        save_path: Optional[str] = None
    ):
        """
        Render current episode state.

        Args:
            env: Multi-agent environment
            state: Current state
            title: Plot title
            save_path: Path to save figure (None = display only)
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Get data
        coverage_map = state.world_state.coverage_map
        obstacles = state.world_state.obstacles
        agent_positions = state.get_agent_positions()

        # Plot 1: Coverage Map
        ax = axes[0]
        self._plot_coverage_map(ax, coverage_map, obstacles, agent_positions)
        ax.set_title(f'Coverage Map ({env._get_coverage_percentage()*100:.1f}%)')

        # Plot 2: Agent Visited Regions
        ax = axes[1]
        self._plot_agent_visits(ax, state, obstacles)
        ax.set_title('Agent Visited Regions')

        # Plot 3: Coordination Regions (if applicable)
        ax = axes[2]
        if state.coordination == CoordinationStrategy.VORONOI and env.voronoi_regions:
            self._plot_voronoi_regions(ax, env.voronoi_regions, obstacles, agent_positions)
            ax.set_title('Voronoi Regions')
        elif state.coordination == CoordinationStrategy.MARKET:
            self._plot_task_assignments(ax, state, obstacles)
            ax.set_title('Task Assignments')
        else:
            self._plot_coverage_map(ax, coverage_map, obstacles, agent_positions)
            ax.set_title(f'Coordination: {state.coordination.value}')

        fig.suptitle(f"{title} (Step {state.step_count})", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _plot_coverage_map(
        self,
        ax,
        coverage_map: np.ndarray,
        obstacles: set,
        agent_positions: List[tuple]
    ):
        """Plot coverage map with obstacles and agents."""
        # Create display map
        display = np.zeros((self.grid_size, self.grid_size, 3))

        # Coverage (green gradient)
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) in obstacles:
                    display[y, x] = [0.2, 0.2, 0.2]  # Dark gray obstacles
                else:
                    cov = coverage_map[x, y]
                    display[y, x] = [1 - cov, 1, 1 - cov]  # White to green

        ax.imshow(display, interpolation='nearest')

        # Draw agents
        for i, pos in enumerate(agent_positions):
            color = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            circle = plt.Circle(
                (pos[0], pos[1]), 0.4,
                color=color, ec='black', linewidth=2, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                pos[0], pos[1], str(i),
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=10, zorder=11
            )

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_agent_visits(self, ax, state: MultiAgentState, obstacles: set):
        """Plot agent visited regions (color-coded)."""
        display = np.ones((self.grid_size, self.grid_size, 3))

        # Obstacles
        for x, y in obstacles:
            display[y, x] = [0.2, 0.2, 0.2]

        # Agent visits (overlay with transparency)
        for agent in state.agents:
            color = self.AGENT_COLORS[agent.agent_id % len(self.AGENT_COLORS)]
            color_rgb = colors.hex2color(color)

            for x, y in agent.robot_state.visited_positions:
                # Blend with existing color
                alpha = 0.5
                display[y, x] = (
                    alpha * np.array(color_rgb) +
                    (1 - alpha) * display[y, x]
                )

        ax.imshow(display, interpolation='nearest')

        # Draw current positions
        for agent in state.agents:
            pos = agent.robot_state.position
            color = self.AGENT_COLORS[agent.agent_id % len(self.AGENT_COLORS)]
            circle = plt.Circle(
                (pos[0], pos[1]), 0.4,
                color=color, ec='black', linewidth=2, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                pos[0], pos[1], str(agent.agent_id),
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=10, zorder=11
            )

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_voronoi_regions(
        self,
        ax,
        voronoi_regions: Dict[int, set],
        obstacles: set,
        agent_positions: List[tuple]
    ):
        """Plot Voronoi partitioning."""
        display = np.ones((self.grid_size, self.grid_size, 3))

        # Color regions
        for agent_id, region in voronoi_regions.items():
            color = self.AGENT_COLORS[agent_id % len(self.AGENT_COLORS)]
            color_rgb = colors.hex2color(color)

            for x, y in region:
                if (x, y) not in obstacles:
                    display[y, x] = color_rgb

        # Obstacles
        for x, y in obstacles:
            display[y, x] = [0.2, 0.2, 0.2]

        ax.imshow(display, interpolation='nearest', alpha=0.6)

        # Draw agents
        for i, pos in enumerate(agent_positions):
            color = self.AGENT_COLORS[i % len(self.AGENT_COLORS)]
            circle = plt.Circle(
                (pos[0], pos[1]), 0.4,
                color=color, ec='black', linewidth=2, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                pos[0], pos[1], str(i),
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=10, zorder=11
            )

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def _plot_task_assignments(self, ax, state: MultiAgentState, obstacles: set):
        """Plot market-based task assignments."""
        display = np.ones((self.grid_size, self.grid_size, 3))

        # Obstacles
        for x, y in obstacles:
            display[y, x] = [0.2, 0.2, 0.2]

        # Coverage map
        coverage_map = state.world_state.coverage_map
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if (x, y) not in obstacles:
                    cov = coverage_map[x, y]
                    display[y, x] = [1 - cov*0.5, 1, 1 - cov*0.5]

        ax.imshow(display, interpolation='nearest')

        # Draw task assignments (arrows from agents to tasks)
        for agent in state.agents:
            if agent.task_assignment is not None:
                pos = agent.robot_state.position
                task = agent.task_assignment
                color = self.AGENT_COLORS[agent.agent_id % len(self.AGENT_COLORS)]

                ax.arrow(
                    pos[0], pos[1],
                    task[0] - pos[0], task[1] - pos[1],
                    color=color, width=0.1, head_width=0.4, head_length=0.3,
                    alpha=0.7, zorder=5
                )

                # Mark task
                circle = plt.Circle(
                    (task[0], task[1]), 0.25,
                    color=color, alpha=0.5, zorder=6
                )
                ax.add_patch(circle)

        # Draw agents
        for agent in state.agents:
            pos = agent.robot_state.position
            color = self.AGENT_COLORS[agent.agent_id % len(self.AGENT_COLORS)]
            circle = plt.Circle(
                (pos[0], pos[1]), 0.4,
                color=color, ec='black', linewidth=2, zorder=10
            )
            ax.add_patch(circle)
            ax.text(
                pos[0], pos[1], str(agent.agent_id),
                ha='center', va='center',
                color='white', fontweight='bold', fontsize=10, zorder=11
            )

        ax.set_xlim(-0.5, self.grid_size - 0.5)
        ax.set_ylim(self.grid_size - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.axis('off')

    def plot_training_metrics(
        self,
        metrics: Dict,
        window: int = 100,
        save_path: Optional[str] = None
    ):
        """
        Plot training metrics.

        Args:
            metrics: Metrics dict from trainer
            window: Moving average window
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # Plot 1: Team Reward
        ax = axes[0, 0]
        rewards = metrics['team_rewards']
        if len(rewards) > 0:
            episodes = np.arange(len(rewards))
            ax.plot(episodes, rewards, alpha=0.3, color='blue', label='Raw')

            if len(rewards) >= window:
                smoothed = self._moving_average(rewards, window)
                ax.plot(episodes, smoothed, color='blue', linewidth=2, label=f'MA({window})')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Team Reward')
            ax.set_title('Team Reward over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 2: Coverage
        ax = axes[0, 1]
        coverages = metrics['team_coverages']
        if len(coverages) > 0:
            episodes = np.arange(len(coverages))
            coverages_pct = np.array(coverages) * 100

            ax.plot(episodes, coverages_pct, alpha=0.3, color='green', label='Raw')

            if len(coverages) >= window:
                smoothed = self._moving_average(coverages_pct, window)
                ax.plot(episodes, smoothed, color='green', linewidth=2, label=f'MA({window})')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Coverage (%)')
            ax.set_title('Coverage over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 3: Episode Length
        ax = axes[1, 0]
        lengths = metrics['episode_lengths']
        if len(lengths) > 0:
            episodes = np.arange(len(lengths))
            ax.plot(episodes, lengths, alpha=0.3, color='purple', label='Raw')

            if len(lengths) >= window:
                smoothed = self._moving_average(lengths, window)
                ax.plot(episodes, smoothed, color='purple', linewidth=2, label=f'MA({window})')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Episode Length')
            ax.set_title('Episode Length over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Plot 4: Collisions
        ax = axes[1, 1]
        collisions = metrics['collisions']
        agent_collisions = metrics['agent_collisions']

        if len(collisions) > 0:
            episodes = np.arange(len(collisions))

            ax.plot(episodes, collisions, alpha=0.3, color='red', label='Total Collisions')
            ax.plot(episodes, agent_collisions, alpha=0.3, color='darkred', label='Agent-Agent')

            if len(collisions) >= window:
                smoothed_total = self._moving_average(collisions, window)
                smoothed_agent = self._moving_average(agent_collisions, window)

                ax.plot(episodes, smoothed_total, color='red', linewidth=2, label=f'Total MA({window})')
                ax.plot(episodes, smoothed_agent, color='darkred', linewidth=2, label=f'Agent MA({window})')

            ax.set_xlabel('Episode')
            ax.set_ylabel('Collisions per Episode')
            ax.set_title('Collisions over Training')
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()

    def _moving_average(self, data: List[float], window: int) -> np.ndarray:
        """Compute moving average."""
        data = np.array(data)
        weights = np.ones(window) / window
        return np.convolve(data, weights, mode='same')


def visualize_episode(
    env: MultiAgentCoverageEnv,
    trainer,
    map_type: str = 'empty',
    max_steps: int = 350,
    save_dir: Optional[str] = None
):
    """
    Run and visualize a full episode.

    Args:
        env: Multi-agent environment
        trainer: Trained trainer
        map_type: Map type
        max_steps: Maximum steps
        save_dir: Directory to save frames (None = display only)
    """
    visualizer = MultiAgentVisualizer(grid_size=env.grid_size)

    # Reset
    state = env.reset(map_type=map_type)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    step = 0
    done = False

    while not done and step < max_steps:
        # Render current state
        if save_dir:
            save_path = os.path.join(save_dir, f'step_{step:04d}.png')
        else:
            save_path = None

        visualizer.render_episode_state(
            env, state,
            title=f"Multi-Agent Coverage Episode",
            save_path=save_path
        )

        # Get actions
        observations = env.get_observations()
        actions = trainer.select_actions(observations, epsilon=0.0)

        # Step
        state, rewards, done, info = env.step(actions)
        step += 1

    # Final state
    if save_dir:
        save_path = os.path.join(save_dir, f'step_{step:04d}_FINAL.png')
    else:
        save_path = None

    visualizer.render_episode_state(
        env, state,
        title=f"Multi-Agent Coverage Episode (FINAL)",
        save_path=save_path
    )

    print(f"\n✓ Episode complete:")
    print(f"  Steps: {step}")
    print(f"  Final Coverage: {info['coverage_pct']*100:.1f}%")
    print(f"  Collisions: {sum([c for c in info['collisions']])}")

    if save_dir:
        print(f"  Frames saved to: {save_dir}")


if __name__ == "__main__":
    print("Testing MultiAgentVisualizer...")

    from multi_agent_trainer import MultiAgentTrainer

    # Create environment and trainer
    env = MultiAgentCoverageEnv(
        num_agents=4,
        grid_size=20,
        coordination=CoordinationStrategy.VORONOI
    )

    trainer = MultiAgentTrainer(
        num_agents=4,
        grid_size=20,
        coordination=CoordinationStrategy.VORONOI,
        parameter_sharing=True
    )

    print(f"✓ Created environment and trainer")

    # Run short episode
    state = env.reset()

    for _ in range(20):
        observations = env.get_observations()
        actions = trainer.select_actions(observations, epsilon=0.1)
        state, rewards, done, info = env.step(actions)

        if done:
            break

    # Visualize
    visualizer = MultiAgentVisualizer(grid_size=20)
    visualizer.render_episode_state(env, state, title="Test Visualization")

    # Plot training metrics (dummy data)
    dummy_metrics = {
        'team_rewards': list(np.random.randn(200).cumsum()),
        'team_coverages': list(np.clip(np.random.rand(200).cumsum() / 200, 0, 1)),
        'episode_lengths': list(200 + 50 * np.random.randn(200)),
        'collisions': list(np.random.poisson(5, 200)),
        'agent_collisions': list(np.random.poisson(2, 200))
    }

    visualizer.plot_training_metrics(dummy_metrics, window=20)

    print(f"✓ Visualization test complete")
