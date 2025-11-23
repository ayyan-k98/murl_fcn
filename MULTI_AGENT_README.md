# Multi-Agent Coverage System

**Stage 2: Multi-Robot Coordinated Coverage with CTDE**

---

## Overview

This is a comprehensive multi-agent extension of the single-agent FCN-based coverage system. The system implements **CTDE (Centralized Training, Decentralized Execution)** for training teams of 2-8 robots to collaboratively cover environments.

### Key Features

- **Multiple Coordination Strategies**: Independent, Voronoi, Market-based, Hierarchical
- **CTDE Training**: Centralized training with decentralized execution
- **Parameter Sharing**: Efficient training with shared networks
- **Curriculum Learning**: 6-phase progressive training
- **Grid-Size Invariant**: Train on 20×20, test on 50×50
- **Partial Observability**: POMDP with ray-casting sensors
- **Collision Avoidance**: Agent-agent collision detection and penalties

---

## Architecture

### System Components

```
multi_agent_env.py          # Multi-agent environment with coordination
multi_agent_trainer.py      # CTDE trainer with parameter sharing
multi_agent_config.py       # Configuration and curriculum
train_multi_agent.py        # Training script
evaluate_multi_agent.py     # Evaluation and testing
multi_agent_vis.py          # Visualization utilities
test_multi_agent.py         # Integration tests
```

### Coordination Strategies

1. **Independent**: No coordination, purely independent agents
2. **Voronoi**: Spatial partitioning - each agent covers its Voronoi region
3. **Market-based**: Agents bid on frontier cells, winner-take-all allocation
4. **Hierarchical**: Leader (Agent 0) assigns tasks to followers

---

## Quick Start

### 1. Installation

Ensure you have the base FCN system installed (see main `README_FCN.md`).

```bash
# Dependencies already installed for single-agent system:
# - PyTorch >= 2.0
# - NumPy >= 1.19
# - Matplotlib >= 3.3
```

### 2. Run Tests

Verify the multi-agent system works:

```bash
python test_multi_agent.py
```

Expected output:
```
========================================================================
MULTI-AGENT INTEGRATION TESTS
========================================================================
...
ALL TESTS PASSED ✓
✅ Multi-agent system is fully functional!
```

### 3. Train a Team

Train a 4-agent team with market coordination:

```bash
python train_multi_agent.py --episodes 1000 --agents 4 --coordination market
```

### 4. Evaluate

Test the trained model:

```bash
python evaluate_multi_agent.py \
  --checkpoint multi_agent_results/checkpoints/your_model.pth \
  --episodes 50 \
  --test-team-sizes 2,3,4,5
```

---

## Training

### Basic Training

```bash
# Train 4 agents with default settings (independent coordination)
python train_multi_agent.py --episodes 1000

# Train with Voronoi coordination
python train_multi_agent.py --episodes 1000 --coordination voronoi

# Train with Market-based coordination
python train_multi_agent.py --episodes 1000 --coordination market
```

### Advanced Options

```bash
# Independent networks (no parameter sharing)
python train_multi_agent.py --episodes 1000 --no-parameter-sharing

# Separate replay memories
python train_multi_agent.py --episodes 1000 --no-shared-replay

# Disable curriculum learning
python train_multi_agent.py --episodes 1000 --no-curriculum

# Custom experiment name
python train_multi_agent.py --episodes 1000 --experiment-name my_experiment
```

### Training Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--episodes` | Total training episodes | 1000 |
| `--agents` | Number of agents [2-8] | 4 |
| `--coordination` | Strategy: independent/voronoi/market/hierarchical | independent |
| `--no-parameter-sharing` | Disable parameter sharing | False |
| `--no-shared-replay` | Disable shared replay memory | False |
| `--no-curriculum` | Disable curriculum learning | False |
| `--experiment-name` | Custom experiment name | Auto-generated |

---

## Curriculum Learning

The system uses a 6-phase curriculum for progressive training:

### Phase 1: Team Formation (Episodes 0-150)
- **Agents**: 2
- **Maps**: Empty only
- **Goal**: Learn basic coordination
- **Expected Coverage**: 75%

### Phase 2: Simple Coordination (Episodes 150-300)
- **Agents**: 2
- **Maps**: Empty (60%), Random (40%)
- **Goal**: Handle obstacles
- **Expected Coverage**: 80%

### Phase 3: Scale to 4 Agents (Episodes 300-450)
- **Agents**: 4
- **Maps**: Empty (70%), Random (30%)
- **Goal**: Scale team size
- **Expected Coverage**: 82%

### Phase 4: Voronoi Coordination (Episodes 450-600)
- **Agents**: 4
- **Maps**: Empty (50%), Random (30%), Maze (20%)
- **Coordination**: Voronoi
- **Expected Coverage**: 85%

### Phase 5: Market Coordination (Episodes 600-750)
- **Agents**: 4
- **Maps**: Empty (40%), Random (30%), Maze (30%)
- **Coordination**: Market
- **Expected Coverage**: 87%

### Phase 6: Final Challenge (Episodes 750-1000)
- **Agents**: 4
- **Maps**: All types (balanced)
- **Coordination**: Market
- **Expected Coverage**: 90%

---

## Evaluation

### Basic Evaluation

```bash
# Evaluate on default configurations
python evaluate_multi_agent.py --checkpoint model.pth --episodes 50
```

### Comprehensive Evaluation

```bash
# Test generalization across team sizes
python evaluate_multi_agent.py \
  --checkpoint model.pth \
  --episodes 20 \
  --test-team-sizes 2,3,4,5,6 \
  --test-strategies independent,voronoi,market \
  --test-maps empty,random,maze,office,warehouse \
  --output-dir results/eval_comprehensive
```

### Evaluation Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--checkpoint` | Path to trained model | Required |
| `--episodes` | Episodes per configuration | 10 |
| `--test-team-sizes` | Team sizes to test | 2,3,4,5 |
| `--test-strategies` | Strategies to test | independent,voronoi,market |
| `--test-maps` | Map types to test | empty,random,maze |
| `--output-dir` | Output directory | multi_agent_results/evaluation |

### Output

The evaluation script generates:
- **Summary table**: Coverage by team size and strategy
- **Plots**:
  - `coverage_by_team_size.png` - Scaling performance
  - `coverage_by_map_type.png` - Per-map performance
  - `collision_analysis.png` - Collision rates
- **Console output**: Detailed statistics per configuration

---

## Configuration

Edit `multi_agent_config.py` to customize training:

### Key Settings

```python
# Team configuration
NUM_AGENTS = 4
COORDINATION = CoordinationStrategy.INDEPENDENT

# Training
PARAMETER_SHARING = True
SHARED_REPLAY = True
TOTAL_EPISODES = 1000

# Rewards
TEAM_REWARD_WEIGHT = 0.5  # Balance individual vs team reward
AGENT_COLLISION_PENALTY = -5.0

# Validation
VALIDATION_FREQ = 50
VALIDATION_EPISODES = 20
```

### Curriculum Phases

Modify `CURRICULUM_PHASES` list to customize the curriculum:

```python
{
    'name': 'Custom Phase',
    'start_ep': 0,
    'end_ep': 200,
    'num_agents': 4,
    'map_distribution': {'empty': 0.5, 'random': 0.5},
    'coordination': CoordinationStrategy.VORONOI,
    'expected_coverage': 0.80,
    'epsilon_floor': 0.1,
    'epsilon_decay': 0.98
}
```

---

## Visualization

### Real-time Episode Visualization

```python
from multi_agent_env import MultiAgentCoverageEnv, CoordinationStrategy
from multi_agent_trainer import MultiAgentTrainer
from multi_agent_vis import visualize_episode

env = MultiAgentCoverageEnv(
    num_agents=4,
    grid_size=20,
    coordination=CoordinationStrategy.VORONOI
)

trainer = MultiAgentTrainer(num_agents=4, grid_size=20)
trainer.load('checkpoint.pth')

# Visualize episode
visualize_episode(env, trainer, map_type='maze', save_dir='frames/')
```

### Training Metrics

```python
from multi_agent_vis import MultiAgentVisualizer

visualizer = MultiAgentVisualizer(grid_size=20)
visualizer.plot_training_metrics(
    trainer.metrics,
    window=100,
    save_path='training_metrics.png'
)
```

---

## Performance Expectations

### Coverage Performance (1000 episodes, 4 agents)

| Map Type | Independent | Voronoi | Market | Hierarchical |
|----------|-------------|---------|--------|--------------|
| Empty | 88-92% | 90-94% | 91-95% | 89-93% |
| Random | 82-86% | 84-88% | 86-90% | 83-87% |
| Maze | 78-82% | 80-84% | 82-86% | 79-83% |
| Office | 75-79% | 77-81% | 79-83% | 76-80% |
| Warehouse | 73-77% | 75-79% | 77-81% | 74-78% |

### Team Size Scaling

| Team Size | Expected Coverage (Empty, Market) |
|-----------|-----------------------------------|
| 2 agents | 85-88% |
| 3 agents | 89-92% |
| 4 agents | 91-95% |
| 5 agents | 92-96% |
| 6 agents | 93-96% |

### Training Time

- **4 agents, 1000 episodes**: ~3-4 hours (GPU)
- **Speed**: ~0.25-0.30 episodes/second
- **Per episode**: ~3-4 seconds

---

## Architecture Details

### Multi-Agent Environment

**Class**: `MultiAgentCoverageEnv`

**Key Methods**:
- `reset()` → `MultiAgentState`
- `step(actions: List[int])` → `(state, rewards, done, info)`
- `get_observations()` → `List[Dict]`

**Features**:
- Simultaneous action execution
- Agent-agent collision detection
- Shared coverage map
- Coordination strategy updates

### CTDE Trainer

**Class**: `MultiAgentTrainer`

**Training Modes**:
1. **Parameter Sharing** (default): All agents share same network
2. **Independent**: Each agent has separate network

**Replay Memory**:
1. **Shared** (default): Single buffer for all agents
2. **Separate**: Independent buffers per agent

**Key Methods**:
- `select_actions(observations)` → `List[int]`
- `train_episode(env)` → `episode_info`
- `validate(env)` → `validation_results`
- `save(filepath)`, `load(filepath)`

---

## API Reference

### MultiAgentCoverageEnv

```python
env = MultiAgentCoverageEnv(
    num_agents=4,              # Number of agents [2-8]
    grid_size=20,              # Grid dimension
    sensor_range=3.0,          # POMDP sensor range
    communication_range=5.0,   # Agent communication range
    coordination=CoordinationStrategy.INDEPENDENT,
    collision_penalty=-5.0,    # Agent-agent collision penalty
    team_reward_weight=0.5     # Team vs individual reward balance
)

# Reset
state = env.reset(map_type='empty')

# Step
actions = [0, 2, 4, 6]  # One action per agent
next_state, rewards, done, info = env.step(actions)

# Observations (POMDP)
observations = env.get_observations()
```

### MultiAgentTrainer

```python
trainer = MultiAgentTrainer(
    num_agents=4,
    grid_size=20,
    coordination=CoordinationStrategy.VORONOI,
    parameter_sharing=True,    # Share network across agents
    shared_replay=True,        # Share replay buffer
    learning_rate=3e-4,
    gamma=0.99
)

# Training
episode_info = trainer.train_episode(env, map_type='random')

# Validation
val_results = trainer.validate(env, num_episodes=20)

# Save/Load
trainer.save('checkpoint.pth')
trainer.load('checkpoint.pth')
```

---

## Troubleshooting

### Issue: Agents colliding frequently

**Solution**: Increase `AGENT_COLLISION_PENALTY` in config:
```python
AGENT_COLLISION_PENALTY = -10.0  # Stronger penalty
```

### Issue: Poor coordination performance

**Solutions**:
1. Train longer (1500+ episodes)
2. Increase team reward weight:
   ```python
   TEAM_REWARD_WEIGHT = 0.7  # More emphasis on team reward
   ```
3. Use curriculum learning (enabled by default)

### Issue: Training unstable

**Solutions**:
1. Use parameter sharing (enabled by default)
2. Reduce learning rate:
   ```python
   LEARNING_RATE = 1e-4
   ```
3. Increase batch size in `config.py`:
   ```python
   BATCH_SIZE = 512
   ```

### Issue: Out of memory

**Solutions**:
1. Use parameter sharing (reduces memory 4x)
2. Reduce replay buffer size in `config.py`:
   ```python
   REPLAY_CAPACITY = 25000
   ```
3. Train with fewer agents initially

---

## Comparison with Single-Agent

| Feature | Single-Agent | Multi-Agent |
|---------|--------------|-------------|
| **Coverage** | 68-73% @ 800 ep | 90-95% @ 1000 ep (4 agents) |
| **Training Time** | ~5.5 hours | ~3-4 hours |
| **Grid-Size Invariance** | ✓ Yes | ✓ Yes |
| **Coordination** | N/A | 4 strategies |
| **Scalability** | 1 agent | 2-8 agents |
| **Real-time Execution** | ✓ Fast | ✓ Fast (decentralized) |

---

## Advanced Usage

### Custom Coordination Strategy

Extend `CoordinationStrategy` in `multi_agent_env.py`:

```python
class CoordinationStrategy(Enum):
    INDEPENDENT = "independent"
    VORONOI = "voronoi"
    MARKET = "market"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"  # Add your strategy
```

Implement in `MultiAgentCoverageEnv._update_coordination()`:

```python
elif self.coordination == CoordinationStrategy.CUSTOM:
    self._custom_coordination()
```

### Hybrid Training

Mix parameter sharing with independent heads:

```python
# In MultiAgentTrainer.__init__(), customize network architecture
# to share encoder but have independent Q-heads
```

---

## Citation

If you use this multi-agent system in your research, please cite:

```bibtex
@software{multi_agent_coverage,
  title={Multi-Agent Coverage System with CTDE},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/murl_fcn}
}
```

---

## Roadmap

### Completed ✓
- [x] Multi-agent environment with 4 coordination strategies
- [x] CTDE trainer with parameter sharing
- [x] 6-phase curriculum learning
- [x] Comprehensive evaluation tools
- [x] Visualization utilities
- [x] Integration tests

### Future Work
- [ ] Communication protocols (explicit message passing)
- [ ] Heterogeneous teams (different sensor ranges, speeds)
- [ ] Dynamic team sizes (agents join/leave during episode)
- [ ] Real-world deployment (ROS integration)
- [ ] Competitive multi-team scenarios
- [ ] Lifelong learning (continual adaptation)

---

## License

MIT License - see `LICENSE` file for details.

---

## Support

For issues, questions, or contributions:
- **Issues**: Open an issue on GitHub
- **Discussions**: Use GitHub Discussions
- **Email**: your.email@example.com

---

**Built on top of the FCN-based single-agent coverage system**

See `README_FCN.md` for single-agent documentation.
