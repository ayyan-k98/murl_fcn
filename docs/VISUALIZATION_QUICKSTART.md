# Visualization System - Quick Start

## What You Get

✅ **Static Training Images** (every N episodes)
- Coverage heatmaps with trajectories
- Environment layout with robot position
- Visit frequency analysis
- Sensor model visualization
- Episode statistics
- Coverage progression plots

✅ **Validation Visualizations** (every validation run)
- All map types (empty, random, room)
- Same comprehensive static images
- **Animated GIFs** showing episode progression

✅ **Final Summary Plots**
- Training curves (rewards, coverage, loss)
- Validation results over time
- Curriculum phase analysis

## Quick Start

### 1. Install Dependencies
```powershell
pip install imageio imageio-ffmpeg
```

### 2. Run Training with Visualization

**Binary Coverage:**
```powershell
python train_fcn_with_viz.py --episodes 400 --viz-interval 50
```

**Probabilistic Coverage (k=1.5, r0=2.5):**
```powershell
python train_fcn_with_viz.py --episodes 400 --probabilistic --viz-interval 50
```

### 3. View Results

Check `training_visualizations/` directory:
```
training_visualizations/
├── train/              # Training episode images
├── validation/         # Validation episode images
├── validation_*/       # GIF animations
└── final_plots/        # Summary plots
```

## Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--episodes` | Number of episodes | 400 |
| `--viz-interval` | Visualize every N episodes | 50 |
| `--validate-interval` | Validate every N episodes | 50 |
| `--save-dir` | Output directory | `training_visualizations` |
| `--probabilistic` | Use probabilistic coverage | False |

## Current Probabilistic Settings

In `config.py`:
```python
PROBABILISTIC_COVERAGE_STEEPNESS = 1.5   # k parameter
PROBABILISTIC_COVERAGE_MIDPOINT = 2.5    # r0 parameter
```

Coverage profile:
- **r=0**: 95.3% (at robot)
- **r=2.5**: 50% (midpoint)
- **r=5**: 11% (sensor edge)

## File Naming Convention

**Training:**
`train_ep{episode:04d}_{map_type}_{coverage:.1f}pct.png`

Example: `train_ep0100_room_67.3pct.png`

**Validation:**
`validation_ep{episode:04d}_{map_type}_{coverage:.1f}pct.png`

Example: `validation_ep0100_empty_85.2pct.png`

**GIFs:**
`validation_ep{episode:04d}_animation.gif`

## What's Visualized

### Static Images (8 subplots):

1. **Coverage Map + Trajectory**
   - Green gradient for coverage
   - Blue→Cyan trajectory path
   - Start/end markers

2. **Environment Layout**
   - Robot position and orientation
   - Obstacles
   - Sensor range indicator

3. **Visit Heatmap**
   - Hot colormap (red = most visited)
   - Identifies exploration patterns

4. **Trajectory Timesteps**
   - Time-colored path
   - Shows exploration order

5. **Coverage Distribution**
   - Histogram of coverage values
   - Mean coverage line

6. **Sensor Model**
   - Sigmoid function plot (probabilistic)
   - Shows coverage decay with distance

7. **Episode Statistics**
   - Steps, distance, coverage
   - Trajectory analysis

8. **Coverage Over Time**
   - Line plot showing accumulation
   - Reveals learning efficiency

### GIF Animations (3 panels):

- **Left**: Coverage map accumulation
- **Middle**: Visit heatmap evolution
- **Right**: Robot's POMDP sensor view

Frame rate: 5 FPS, sampled every 5 steps

## Disk Space

For 400 episodes (default settings):
- ~18 MB total
- 8 training images (~2.4 MB)
- 24 validation images (~7.2 MB)
- 8 GIFs (~8 MB)
- 3 final plots (~0.6 MB)

## Performance Impact

- **Static images**: ~2-3 seconds each
- **GIFs**: ~5-10 seconds each
- **Total overhead**: ~5-10% of training time

## Files Created

| File | Purpose |
|------|---------|
| `visualization.py` | Core visualization classes |
| `train_fcn_with_viz.py` | Training script with visualization |
| `visualization_requirements.txt` | Python dependencies |
| `VISUALIZATION_README.md` | Full documentation |
| `VISUALIZATION_QUICKSTART.md` | This file |

## Modified Files

| File | Changes |
|------|---------|
| `data_structures.py` | Added `coverage_over_time` tracking |
| `environment.py` | Track coverage at each step |
| `config.py` | Updated sigmoid parameters (k=1.5, r0=2.5) |

## Tips

### More Frequent Visualizations
```powershell
python train_fcn_with_viz.py --viz-interval 25 --validate-interval 25
```

### Test Visualization System
```powershell
python visualization.py
# Check test_visualizations/ directory
```

### Disable GIFs (Faster)
Edit `train_fcn_with_viz.py`, line ~180:
```python
create_gifs=False  # Change from True
```

## Troubleshooting

**No images created?**
- Check `--save-dir` permissions
- Verify `--viz-interval` divisible by episode count

**GIF errors?**
- Install: `pip install imageio-ffmpeg`
- Check disk space

**Slow training?**
- Increase `--viz-interval` (e.g., 100)
- Disable GIFs (see above)

## Next Steps

1. **Install dependencies** ✓
2. **Run test** (optional): `python visualization.py`
3. **Start training**: `python train_fcn_with_viz.py --probabilistic --episodes 400`
4. **Monitor** `training_visualizations/` directory
5. **Analyze results** after training completes

## Full Documentation

See `VISUALIZATION_README.md` for:
- Detailed API reference
- Customization options
- Advanced usage examples
- Troubleshooting guide
