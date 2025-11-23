"""
Advanced Visualization System for Coverage Training

Provides static images with coverage heatmaps, trajectories, and GIF generation.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
from matplotlib.collections import LineCollection
import imageio
from typing import List, Tuple, Optional, Dict
from pathlib import Path

from config import config
from data_structures import WorldState, RobotState


class CoverageVisualizer:
    """Comprehensive visualization for coverage episodes."""
    
    def __init__(self, save_dir: str = "visualizations"):
        """
        Initialize visualizer.
        
        Args:
            save_dir: Directory to save visualizations
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_episode_static(self,
                          world_state: WorldState,
                          robot_state: RobotState,
                          trajectory: List[Tuple[int, int]],
                          episode_num: int,
                          coverage_pct: float,
                          episode_reward: float,
                          mode: str = "train",
                          map_type: str = "unknown",
                          save: bool = True,
                          show: bool = False) -> Optional[str]:
        """
        Create comprehensive static visualization of an episode.
        
        Args:
            world_state: Final world state
            robot_state: Final robot state
            trajectory: List of (x, y) positions
            episode_num: Episode number
            coverage_pct: Final coverage percentage
            episode_reward: Total episode reward
            mode: "train" or "validation"
            map_type: Map type descriptor
            save: Whether to save the figure
            show: Whether to display the figure
            
        Returns:
            Path to saved figure if save=True, else None
        """
        grid_size = world_state.grid_size
        
        # Create figure with 4 subplots
        fig = plt.figure(figsize=(20, 10))
        gs = fig.add_gridspec(2, 4, hspace=0.25, wspace=0.3)
        
        # Title with episode info
        env_type = "Probabilistic" if config.USE_PROBABILISTIC_ENV else "Binary"
        title = f"{mode.upper()} Episode {episode_num} - {map_type} - {env_type} Coverage\n"
        title += f"Coverage: {coverage_pct:.1f}% | Reward: {episode_reward:.2f} | Steps: {len(trajectory)}"
        fig.suptitle(title, fontsize=14, weight='bold')
        
        # 1. Coverage Heatmap with Trajectory
        ax1 = fig.add_subplot(gs[0, 0])
        self._plot_coverage_heatmap(ax1, world_state, robot_state, trajectory, "Coverage Map + Trajectory")
        
        # 2. Obstacles and Free Space
        ax2 = fig.add_subplot(gs[0, 1])
        self._plot_environment_layout(ax2, world_state, robot_state, "Environment Layout")
        
        # 3. Visit Heatmap
        ax3 = fig.add_subplot(gs[0, 2])
        self._plot_visit_heatmap(ax3, robot_state, "Visit Frequency")
        
        # 4. Trajectory with timesteps
        ax4 = fig.add_subplot(gs[0, 3])
        self._plot_trajectory_timesteps(ax4, world_state, trajectory, "Trajectory Evolution")
        
        # 5. Coverage Profile (if probabilistic)
        ax5 = fig.add_subplot(gs[1, 0])
        if config.USE_PROBABILISTIC_ENV:
            self._plot_coverage_distribution(ax5, world_state, "Coverage Distribution")
        else:
            self._plot_coverage_histogram(ax5, world_state, "Coverage Histogram")
        
        # 6. Sensor Coverage Visualization
        ax6 = fig.add_subplot(gs[1, 1])
        self._plot_sensor_coverage(ax6, world_state, robot_state, "Sensor Model Visualization")
        
        # 7. Trajectory Statistics
        ax7 = fig.add_subplot(gs[1, 2])
        self._plot_trajectory_stats(ax7, trajectory, world_state, robot_state)
        
        # 8. Coverage Over Time
        ax8 = fig.add_subplot(gs[1, 3])
        self._plot_coverage_over_time(ax8, robot_state)
        
        plt.tight_layout()
        
        save_path = None
        if save:
            filename = f"{mode}_ep{episode_num:04d}_{map_type}_{coverage_pct:.1f}pct.png"
            save_path = self.save_dir / mode / filename
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved visualization: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
        
        return str(save_path) if save_path else None
    
    def _plot_coverage_heatmap(self, ax, world_state, robot_state, trajectory, title):
        """Plot coverage heatmap with trajectory overlay."""
        grid_size = world_state.grid_size
        
        # Create coverage visualization
        coverage_vis = np.zeros((grid_size, grid_size, 4))  # RGBA
        
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in world_state.obstacles:
                    coverage_vis[x, y] = [0, 0, 0, 1]  # Black for obstacles
                else:
                    cov = world_state.coverage_map[x, y]
                    # Green gradient for coverage
                    coverage_vis[x, y] = [0, cov, 0, 0.7 if cov > 0 else 0.1]
        
        ax.imshow(coverage_vis.transpose(1, 0, 2), origin='lower', extent=[-0.5, grid_size-0.5, -0.5, grid_size-0.5])
        
        # Draw trajectory with gradient (blue to cyan)
        if len(trajectory) > 1:
            traj_array = np.array(trajectory)
            colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(trajectory)))
            
            for i in range(len(trajectory) - 1):
                ax.plot([trajectory[i][0], trajectory[i+1][0]], 
                       [trajectory[i][1], trajectory[i+1][1]], 
                       color=colors[i], linewidth=2, alpha=0.7)
            
            # Mark start and end
            ax.plot(trajectory[0][0], trajectory[0][1], 'go', markersize=12, 
                   label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
            ax.plot(trajectory[-1][0], trajectory[-1][1], 'r*', markersize=16, 
                   label='End', markeredgecolor='darkred', markeredgewidth=2)
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _plot_environment_layout(self, ax, world_state, robot_state, title):
        """Plot environment with obstacles and robot."""
        grid_size = world_state.grid_size
        
        # Draw grid
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in world_state.obstacles:
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                   facecolor='black', edgecolor='gray', linewidth=0.5)
                    ax.add_patch(rect)
                else:
                    rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                                   facecolor='white', edgecolor='lightgray', linewidth=0.5)
                    ax.add_patch(rect)
        
        # Draw robot with orientation
        rx, ry = robot_state.position
        orient = robot_state.orientation
        
        circle = Circle((rx, ry), 0.3, facecolor='blue', edgecolor='darkblue', linewidth=2)
        ax.add_patch(circle)
        
        # Orientation arrow
        dx = 0.4 * np.cos(orient)
        dy = 0.4 * np.sin(orient)
        ax.arrow(rx, ry, dx, dy, head_width=0.2, head_length=0.15, fc='yellow', ec='darkorange', linewidth=2)
        
        # Sensor range
        sensor_circle = Circle((rx, ry), config.SENSOR_RANGE,
                             facecolor='none', edgecolor='blue',
                             linestyle='--', linewidth=1.5, alpha=0.5)
        ax.add_patch(sensor_circle)
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _plot_visit_heatmap(self, ax, robot_state, title):
        """Plot visit frequency heatmap."""
        im = ax.imshow(robot_state.visit_heat.T, cmap='hot', origin='lower', 
                      interpolation='nearest')
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.colorbar(im, ax=ax, label='Visits')
        
        # Mark current position
        rx, ry = robot_state.position
        ax.plot(rx, ry, 'c*', markersize=12, markeredgecolor='white', markeredgewidth=1)
    
    def _plot_trajectory_timesteps(self, ax, world_state, trajectory, title):
        """Plot trajectory with time-colored segments."""
        grid_size = world_state.grid_size
        
        # Background
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        
        # Draw obstacles lightly
        for (x, y) in world_state.obstacles:
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1, 
                           facecolor='lightgray', edgecolor='gray', linewidth=0.5, alpha=0.3)
            ax.add_patch(rect)
        
        if len(trajectory) > 1:
            # Create line segments
            points = np.array(trajectory).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
            # Color by time (viridis colormap)
            colors = np.linspace(0, 1, len(trajectory))
            lc = LineCollection(segments, array=colors, cmap='viridis', linewidth=3, alpha=0.8)
            ax.add_collection(lc)
            
            # Colorbar for time
            cbar = plt.colorbar(lc, ax=ax, label='Time Step')
            
            # Mark start and end
            ax.plot(trajectory[0][0], trajectory[0][1], 'go', markersize=10, 
                   markeredgecolor='darkgreen', markeredgewidth=2, label='Start')
            ax.plot(trajectory[-1][0], trajectory[-1][1], 'r*', markersize=14, 
                   markeredgecolor='darkred', markeredgewidth=2, label='End')
        
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.2, linewidth=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    def _plot_coverage_distribution(self, ax, world_state, title):
        """Plot coverage value distribution (for probabilistic coverage)."""
        free_cells = []
        for x in range(world_state.grid_size):
            for y in range(world_state.grid_size):
                if (x, y) not in world_state.obstacles:
                    free_cells.append(world_state.coverage_map[x, y])
        
        ax.hist(free_cells, bins=50, edgecolor='black', alpha=0.7, color='green')
        ax.set_xlabel('Coverage Value')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.axvline(np.mean(free_cells), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(free_cells):.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_coverage_histogram(self, ax, world_state, title):
        """Plot binary coverage histogram."""
        free_cells = []
        for x in range(world_state.grid_size):
            for y in range(world_state.grid_size):
                if (x, y) not in world_state.obstacles:
                    free_cells.append(world_state.coverage_map[x, y])
        
        covered = sum(1 for c in free_cells if c > 0)
        uncovered = len(free_cells) - covered
        
        ax.bar(['Uncovered', 'Covered'], [uncovered, covered], color=['red', 'green'], alpha=0.7, edgecolor='black')
        ax.set_ylabel('Cell Count')
        ax.set_title(title)
        ax.text(0, uncovered + 5, str(uncovered), ha='center', fontsize=10, weight='bold')
        ax.text(1, covered + 5, str(covered), ha='center', fontsize=10, weight='bold')
        ax.grid(True, alpha=0.3, axis='y')
    
    def _plot_sensor_coverage(self, ax, world_state, robot_state, title):
        """Visualize sensor model and coverage decay."""
        if config.USE_PROBABILISTIC_ENV:
            # Plot sigmoid function
            distances = np.linspace(0, config.SENSOR_RANGE, 100)
            r0 = config.PROBABILISTIC_COVERAGE_MIDPOINT
            k = config.PROBABILISTIC_COVERAGE_STEEPNESS
            p_cov = 1.0 / (1.0 + np.exp(k * (distances - r0)))
            
            ax.plot(distances, p_cov, 'b-', linewidth=3, label='Sensor Model')
            ax.axvline(r0, color='red', linestyle='--', linewidth=2, alpha=0.6, label=f'Midpoint (r0={r0})')
            ax.axhline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.5)
            ax.fill_between(distances, 0, p_cov, alpha=0.2, color='blue')
            
            ax.set_xlabel('Distance from Robot (cells)')
            ax.set_ylabel('Coverage Probability')
            ax.set_title(f'{title}\nk={k}, r0={r0}')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, config.SENSOR_RANGE)
            ax.set_ylim(0, 1.05)
        else:
            ax.text(0.5, 0.5, 'Binary Coverage\n(No decay model)', 
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.set_title(title)
            ax.axis('off')
    
    def _plot_trajectory_stats(self, ax, trajectory, world_state, robot_state):
        """Display trajectory statistics."""
        ax.axis('off')
        
        # Calculate stats
        total_distance = 0
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1][0] - trajectory[i][0]
            dy = trajectory[i+1][1] - trajectory[i][1]
            total_distance += np.sqrt(dx**2 + dy**2)
        
        # Unique cells visited
        unique_cells = len(set(trajectory))
        
        # Coverage stats
        total_free = world_state.grid_size ** 2 - len(world_state.obstacles)
        covered_count = np.sum(world_state.coverage_map > 0) - len(world_state.obstacles)
        
        stats_text = f"""
        TRAJECTORY STATISTICS
        ────────────────────────
        Steps: {len(trajectory)}
        Unique cells: {unique_cells}
        Total distance: {total_distance:.2f}
        Avg step distance: {total_distance/(len(trajectory)-1):.3f}
        
        COVERAGE STATISTICS
        ────────────────────────
        Free cells: {total_free}
        Covered cells: {covered_count}
        Coverage: {covered_count/total_free*100:.1f}%
        
        Avg coverage value: {np.mean(world_state.coverage_map[world_state.coverage_map > 0]):.3f}
        Max coverage value: {np.max(world_state.coverage_map):.3f}
        """
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
        ax.set_title('Episode Statistics')
    
    def _plot_coverage_over_time(self, ax, robot_state):
        """Plot coverage progression over time."""
        if hasattr(robot_state, 'coverage_over_time') and len(robot_state.coverage_over_time) > 0:
            steps = list(range(len(robot_state.coverage_over_time)))
            coverages = [c * 100 for c in robot_state.coverage_over_time]
            
            ax.plot(steps, coverages, 'g-', linewidth=2, alpha=0.7)
            ax.fill_between(steps, 0, coverages, alpha=0.2, color='green')
            ax.set_xlabel('Step')
            ax.set_ylabel('Coverage (%)')
            ax.set_title('Coverage Progression')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 100)
        else:
            ax.text(0.5, 0.5, 'Coverage over time\nnot tracked', 
                   ha='center', va='center', fontsize=10, transform=ax.transAxes)
            ax.set_title('Coverage Progression')
            ax.axis('off')
    
    def create_episode_gif(self,
                          frames: List[Dict],
                          episode_num: int,
                          mode: str = "validation",
                          fps: int = 5) -> str:
        """
        Create animated GIF of episode progression.
        
        Args:
            frames: List of dicts with keys: 'world_state', 'robot_state', 'step', 'coverage'
            episode_num: Episode number
            mode: "train" or "validation"
            fps: Frames per second
            
        Returns:
            Path to saved GIF
        """
        temp_frames = []
        
        for i, frame in enumerate(frames):
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            world_state = frame['world_state']
            robot_state = frame['robot_state']
            step = frame['step']
            coverage = frame['coverage']
            
            fig.suptitle(f"Episode {episode_num} - Step {step} - Coverage: {coverage*100:.1f}%", 
                        fontsize=14, weight='bold')
            
            # 1. Coverage Map
            ax = axes[0]
            self._render_frame_coverage(ax, world_state, robot_state, "Coverage Map")
            
            # 2. Visit Heatmap
            ax = axes[1]
            im = ax.imshow(robot_state.visit_heat.T, cmap='hot', origin='lower')
            ax.set_title('Visit Heatmap')
            ax.plot(robot_state.position[0], robot_state.position[1], 'c*', markersize=12)
            plt.colorbar(im, ax=ax, label='Visits')
            
            # 3. Sensor View
            ax = axes[2]
            self._render_frame_sensor(ax, world_state, robot_state, "Sensor View")
            
            plt.tight_layout()
            
            # Save frame to temporary file
            temp_path = self.save_dir / f"temp_frame_{i:04d}.png"
            plt.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close()
            
            temp_frames.append(str(temp_path))
        
        # Create GIF
        gif_filename = f"{mode}_ep{episode_num:04d}_animation.gif"
        gif_path = self.save_dir / mode / gif_filename
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        
        images = [imageio.imread(f) for f in temp_frames]
        imageio.mimsave(str(gif_path), images, fps=fps, loop=0)
        
        # Cleanup temporary files
        for temp_file in temp_frames:
            os.remove(temp_file)
        
        print(f"✓ Created GIF: {gif_path}")
        return str(gif_path)
    
    def _render_frame_coverage(self, ax, world_state, robot_state, title):
        """Render single frame for GIF - coverage view."""
        grid_size = world_state.grid_size
        coverage_vis = np.zeros((grid_size, grid_size, 4))
        
        for x in range(grid_size):
            for y in range(grid_size):
                if (x, y) in world_state.obstacles:
                    coverage_vis[x, y] = [0, 0, 0, 1]
                else:
                    cov = world_state.coverage_map[x, y]
                    coverage_vis[x, y] = [0, cov, 0, 0.7 if cov > 0 else 0.1]
        
        ax.imshow(coverage_vis.transpose(1, 0, 2), origin='lower', 
                 extent=[-0.5, grid_size-0.5, -0.5, grid_size-0.5])
        
        # Robot position
        rx, ry = robot_state.position
        circle = Circle((rx, ry), 0.3, facecolor='blue', edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        
        # Sensor range
        sensor_circle = Circle((rx, ry), config.SENSOR_RANGE,
                             facecolor='none', edgecolor='cyan',
                             linestyle='--', linewidth=2, alpha=0.7)
        ax.add_patch(sensor_circle)
        
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
    
    def _render_frame_sensor(self, ax, world_state, robot_state, title):
        """Render single frame for GIF - sensor view."""
        grid_size = world_state.grid_size
        
        # Gray background for unsensed
        ax.set_xlim(-0.5, grid_size - 0.5)
        ax.set_ylim(-0.5, grid_size - 0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('lightgray')
        
        # Draw only sensed cells
        for (x, y), (coverage, cell_type) in robot_state.local_map.items():
            if cell_type == "obstacle":
                color = 'black'
            elif coverage > 0:
                color = plt.cm.Greens(coverage)
            else:
                color = 'white'
            
            rect = Rectangle((x - 0.5, y - 0.5), 1, 1,
                           facecolor=color, edgecolor='gray', linewidth=0.5)
            ax.add_patch(rect)
        
        # Robot
        rx, ry = robot_state.position
        circle = Circle((rx, ry), 0.3, facecolor='blue', edgecolor='white', linewidth=2)
        ax.add_patch(circle)
        
        ax.set_title(title)
        ax.grid(True, alpha=0.2)


if __name__ == "__main__":
    print("Testing visualization system...")
    
    from environment import CoverageEnvironment
    from fcn_agent import FCNAgent
    
    # Create environment and agent
    env = CoverageEnvironment(grid_size=20, map_type="room")
    agent = FCNAgent(grid_size=20)
    visualizer = CoverageVisualizer(save_dir="test_visualizations")
    
    # Run episode
    state = env.reset()
    trajectory = [state.position]
    frames = []
    total_reward = 0
    
    for step in range(50):
        # Store frame for GIF
        if step % 5 == 0:  # Sample every 5 steps
            frames.append({
                'world_state': env.world_state,
                'robot_state': env.robot_state,
                'step': step,
                'coverage': env.world_state.coverage_map.sum() / (env.world_state.grid_size ** 2 - len(env.world_state.obstacles))
            })
        
        action = agent.select_action(state, env.world_state)
        next_state, reward, done, info = env.step(action)
        trajectory.append(next_state.position)
        total_reward += reward
        state = next_state
        
        if done:
            break
    
    # Final coverage
    coverage_pct = info['coverage_percentage']
    
    # Create static visualization
    print("\nCreating static visualization...")
    visualizer.plot_episode_static(
        env.world_state,
        env.robot_state,
        trajectory,
        episode_num=1,
        coverage_pct=coverage_pct,
        episode_reward=total_reward,
        mode="test",
        map_type="room",
        save=True,
        show=False
    )
    
    # Create GIF
    print("\nCreating GIF animation...")
    visualizer.create_episode_gif(
        frames,
        episode_num=1,
        mode="test",
        fps=3
    )
    
    print("\n✓ Visualization test complete!")
    print(f"✓ Check 'test_visualizations' directory for outputs")
