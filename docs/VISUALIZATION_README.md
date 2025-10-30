# Visualization System Documentation

## Overview

Comprehensive visualization system for coverage training with:
- **Static images** with coverage heatmaps, trajectories, and statistics
- **Animated GIFs** showing episode progression
- **Training plots** with metrics and curriculum phases
- **Validation visualizations** for all map types

## Installation

```bash
pip install -r visualization_requirements.txt
```

## Usage

### Training with Visualization

```bash
# Binary coverage
python train_fcn_with_viz.py --episodes 400 --viz-interval 50

# Probabilistic coverage (k=1.5, r0=2.5)
python train_fcn_with_viz.py --episodes 400 --probabilistic --viz-interval 50
```

### Command-Line Arguments

- `--episodes`: Number of training episodes (default: 400)
- `--grid-size`: Grid size (default: 20)
- `--validate-interval`: Validation frequency (default: 50)
- `--viz-interval`: Visualization frequency (default: 50)
- `--save-dir`: Output directory (default: `training_visualizations`)
- `--probabilistic`: Use probabilistic coverage environment
- `--6ch`: Use 6-channel input (for compatibility)

## Output Structure

```
training_visualizations/
├── train/
│   ├── train_ep0050_room_67.3pct.png
│   ├── train_ep0100_empty_82.1pct.png
│   └── ...
├── validation/
│   ├── validation_ep0050_empty_85.2pct.png
│   ├── validation_ep0050_random_71.3pct.png
│   ├── validation_ep0050_room_68.9pct.png
│   └── ...
├── validation_empty/
│   ├── validation_ep0050_animation.gif
│   └── ...
├── validation_random/
│   └── ...
├── validation_room/
│   └── ...
└── final_plots/
    ├── training_curves.png
    ├── validation_results.png
    └── curriculum_phases.png
```

## Static Visualization Features

Each static image includes 8 subplots:

### Top Row:
1. **Coverage Heatmap + Trajectory**
   - Green gradient shows coverage values (0-100%)
   - Blue→Cyan trajectory shows agent path
   - Start (green circle) and End (red star) markers

2. **Environment Layout**
   - Black cells = obstacles
   - Blue circle = robot position
   - Yellow arrow = orientation
   - Dashed circle = sensor range

3. **Visit Heatmap**
   - Hot colormap shows visit frequency
   - Identifies over-visited areas

4. **Trajectory with Timesteps**
   - Viridis colormap shows time progression
   - Helps analyze exploration strategy

### Bottom Row:
5. **Coverage Distribution** (probabilistic) or **Histogram** (binary)
   - Shows distribution of coverage values
   - Mean coverage highlighted

6. **Sensor Model Visualization**
   - Plots sigmoid: P_cov(r) = 1/(1 + e^(k*(r-r0)))
   - Shows midpoint and coverage decay

7. **Episode Statistics**
   - Steps, distance, unique cells
   - Coverage metrics and averages

8. **Coverage Over Time**
   - Line plot showing coverage progression
   - Reveals learning efficiency

## GIF Animation Features

Animated GIFs show episode progression with 3 views:
- **Coverage Map**: Real-time coverage accumulation
- **Visit Heatmap**: Exploration pattern evolution
- **Sensor View**: POMDP local map updates

Settings:
- Frame rate: 5 FPS (configurable)
- Sampling: Every 5 steps (reduces file size)
- Loop: Infinite

## Probabilistic Coverage Visualization

When `--probabilistic` is enabled:

1. **Sigmoid parameters displayed** in title
2. **Coverage distribution plot** shows probabilistic values
3. **Sensor model plot** visualizes distance-based decay
4. **Coverage heatmap** uses continuous gradients (not binary)

Current parameters (config.py):
- k = 1.5 (steepness)
- r0 = 2.5 (midpoint)

Coverage profile:
- r=0: 95.3% (at robot)
- r=2.5: 50% (midpoint)
- r=5: 11% (sensor edge)

## Final Plots

Generated at end of training:

### 1. Training Curves
- Episode rewards (3-step returns)
- Coverage percentage over time
- Episode length
- DQN loss (log scale)
- Epsilon decay
- Gradient norms

### 2. Validation Results
- Coverage by map type over episodes
- Validation checkpoints marked
- Trend lines for each map type

### 3. Curriculum Phases
- Coverage with phase boundaries
- Target coverage lines per phase
- Phase labels and transitions
- Validation stars overlaid

## Performance Considerations

### Static Images
- Generated every `--viz-interval` episodes
- ~2-3 seconds per image
- ~200-300 KB per PNG

### GIF Creation
- Only for validation episodes
- ~5-10 seconds per GIF
- ~500KB - 2MB per GIF
- Can be disabled by modifying `create_gifs=False` in code

### Disk Space

For 400 episodes with default settings:
- Training images: ~8 × 300KB = 2.4 MB
- Validation images: ~24 × 300KB = 7.2 MB
- GIFs: ~8 × 1MB = 8 MB
- Final plots: ~3 × 200KB = 600 KB
- **Total: ~18 MB**

## Tips

### High-Quality Visualizations
```bash
# More frequent visualizations
python train_fcn_with_viz.py --viz-interval 25 --validate-interval 25

# Focus on specific phase
python train_fcn_with_viz.py --episodes 200 --viz-interval 10
```

### Fast Training (Minimal Visualization)
```bash
# Only validate without GIFs (modify code: create_gifs=False)
python train_fcn_with_viz.py --viz-interval 100 --validate-interval 100
```

### Analyze Specific Episode
```python
from visualization import CoverageVisualizer
from environment import CoverageEnvironment

visualizer = CoverageVisualizer("analysis")
env = CoverageEnvironment(grid_size=20, map_type="room")
# ... run episode ...
visualizer.plot_episode_static(env.world_state, env.robot_state, trajectory, ...)
```

## Troubleshooting

### ImportError: No module named 'imageio'
```bash
pip install imageio imageio-ffmpeg
```

### GIF creation fails
- Check disk space
- Ensure write permissions
- Try reducing frame count (modify step sampling)

### Visualizations not showing
- `show=False` by default (saves without display)
- Set `show=True` for interactive viewing
- Or view saved PNGs/GIFs directly

### Memory issues with large episodes
- GIF creation stores frames in memory
- Reduce sampling rate (currently every 5 steps)
- Or disable GIF generation for long episodes

## Customization

Edit `visualization.py` to customize:
- Color schemes (change colormaps)
- Plot layout (modify subplot grid)
- GIF frame rate (change `fps` parameter)
- Statistics displayed (modify `_plot_trajectory_stats`)

## Examples

### Compare Binary vs Probabilistic

```bash
# Binary coverage
python train_fcn_with_viz.py --episodes 200 --save-dir viz_binary

# Probabilistic coverage
python train_fcn_with_viz.py --episodes 200 --probabilistic --save-dir viz_probabilistic

# Compare final plots side-by-side
```

### Analyze Validation Performance

Check `training_visualizations/validation/` for:
- Coverage patterns on different map types
- Trajectory efficiency (distance vs coverage)
- Sensor utilization (visit heatmap)

### Debug Learning Issues

If coverage is low:
1. Check trajectory plots (random vs systematic?)
2. Check visit heatmap (stuck in loops?)
3. Check coverage over time (plateauing early?)
4. Check sensor visualization (probabilistic too harsh?)

## API Reference

### CoverageVisualizer

```python
visualizer = CoverageVisualizer(save_dir="visualizations")

# Static image
visualizer.plot_episode_static(
    world_state=env.world_state,
    robot_state=env.robot_state,
    trajectory=trajectory,
    episode_num=100,
    coverage_pct=75.3,
    episode_reward=234.5,
    mode="train",  # or "validation"
    map_type="room",
    save=True,
    show=False
)

# Animated GIF
frames = [
    {'world_state': ws, 'robot_state': rs, 'step': i, 'coverage': cov}
    for i, (ws, rs, cov) in enumerate(episode_data)
]
visualizer.create_episode_gif(
    frames=frames,
    episode_num=100,
    mode="validation",
    fps=5
)
```

## Credits

Visualization system designed for:
- Multi-Agent Reinforcement Learning (MARL)
- Coverage Path Planning
- FCN + Spatial Softmax architecture
- Probabilistic sensor models
