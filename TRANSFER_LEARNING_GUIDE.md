# Transfer Learning Guide: Single-Agent → Multi-Agent

## Overview

This guide explains how to use pre-trained single-agent checkpoints to initialize multi-agent systems, enabling transfer learning from single-agent coverage to multi-agent coordination.

---

## Complete Training Pipeline with Transfer Learning

### Phase 1: Single-Agent Baseline (5 channels)
Train a single agent with standard 5-channel input:

```bash
python train_fcn.py --episodes 2000
```

**Output**: `checkpoints/fcn_final.pt` (5-channel weights)

**Expected Performance**: 68-73% coverage on validation

---

### Phase 2: Single-Agent with 6th Channel (dummy zeros)
Train a single agent with 6-channel input (6th channel = all zeros):

```bash
python train_fcn.py --episodes 2000 --use-6ch
```

**Output**: `checkpoints/fcn_final.pt` (6-channel weights)

**Expected Performance**: 68-73% coverage (same as 5ch)

**Purpose**: Validate that 6th channel doesn't hurt performance when empty

---

### Phase 3: Multi-Agent Independent (5ch) - Transfer Learning
Use Phase 1 checkpoint to initialize multi-agent system:

```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --resume-from checkpoints/fcn_final.pt
```

**What happens**:
- Each of the 4 agents initialized with Phase 1 weights
- Agents start with good coverage behavior
- Training focuses on coordination, not basic coverage

**Expected Performance**: 75-80% coverage

---

### Phase 4: Multi-Agent Independent (6ch + comm) - Transfer Learning
Use Phase 2 checkpoint to initialize multi-agent with communication:

```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/fcn_final.pt
```

**What happens**:
- Each agent initialized with Phase 2 (6ch) weights
- 6th channel now active (receives agent occupancy from communication)
- Agents can see where others are and coordinate proactively

**Expected Performance**: 82-88% coverage

---

### Phase 5: QMIX (5ch) - Transfer Learning
Use Phase 1 checkpoint for QMIX training:

```bash
python train_qmix.py --episodes 400 --agents 4 \
    --resume-from checkpoints/fcn_final.pt
```

**What happens**:
- Each agent's Q-network initialized with Phase 1 weights
- Mixing network trained from scratch
- Centralized training learns joint value function

**Expected Performance**: 80-85% coverage

---

### Phase 6: QMIX (6ch + comm) - Transfer Learning ⭐ BEST
Use Phase 2 checkpoint for full QMIX with communication:

```bash
python train_qmix.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/fcn_final.pt
```

**What happens**:
- Each agent's Q-network initialized with Phase 2 (6ch) weights
- Communication enables agent occupancy sharing
- QMIX learns centralized coordination with proactive collision avoidance

**Expected Performance**: 88-92% coverage (BEST)

---

## Transfer Learning Benefits

### ✅ Advantages

1. **Faster Convergence**: Agents start with good coverage skills
2. **Better Sample Efficiency**: Less exploration needed for basic behaviors
3. **Higher Final Performance**: More time for coordination learning
4. **Stable Training**: Pre-trained weights provide good initialization

### 📊 Expected Speedup

| Scenario | From Scratch | With Transfer | Speedup |
|----------|-------------|---------------|---------|
| Multi-agent independent | 600 episodes | 400 episodes | 1.5x |
| QMIX | 800 episodes | 400 episodes | 2.0x |
| Convergence time | ~8 hours | ~3 hours | 2.7x |

---

## Technical Details

### How Transfer Loading Works

#### For `train_multi_agent.py`:

**Parameter Sharing Mode** (default):
```python
if parameter_sharing:
    # All agents share one network
    trainer.agents[0].load(resume_from)
    # agents[1], agents[2], agents[3] are references to agents[0]
```

**Independent Networks Mode**:
```python
if not parameter_sharing:
    # Each agent has its own network
    for i, agent in enumerate(trainer.agents):
        agent.load(resume_from)  # Same initial weights, will diverge during training
```

#### For `train_qmix.py`:

```python
checkpoint = torch.load(resume_from)

# Load into each agent's Q-network
for i in range(num_agents):
    qmix_agent.agent_qnets[i].load_state_dict(checkpoint['policy_net_state_dict'])
    qmix_agent.target_qnets[i].load_state_dict(checkpoint['policy_net_state_dict'])

# Mixing network trained from scratch
```

**Important**: Mixing network is NOT initialized from checkpoint (it learns joint coordination)

---

## Channel Compatibility

### ⚠️ CRITICAL: Checkpoint Must Match Training Channels

| Training Mode | Checkpoint Required | Command |
|---------------|---------------------|---------|
| 5-channel multi-agent | 5-channel checkpoint | `--resume-from checkpoints/fcn_final.pt` (Phase 1) |
| 6-channel multi-agent | 6-channel checkpoint | `--use-6ch --resume-from checkpoints/fcn_final.pt` (Phase 2) |

**What happens if mismatched?**
```bash
# ❌ WRONG: Using 5ch checkpoint for 6ch training
python train_multi_agent.py --use-6ch --resume-from checkpoints_5ch/fcn_final.pt
# Error: RuntimeError: size mismatch for encoder.0.weight: [64, 7, 3, 3] vs [64, 8, 3, 3]

# ✅ CORRECT: Matching channels
python train_multi_agent.py --use-6ch --resume-from checkpoints_6ch/fcn_final.pt
```

---

## Checkpoint Management

### Recommended Directory Structure

```
checkpoints/
├── single_agent_5ch/
│   └── fcn_final.pt          # Phase 1 output
├── single_agent_6ch/
│   └── fcn_final.pt          # Phase 2 output
├── multi_agent_5ch/
│   └── multi_agent_FINAL.pth # Phase 3 output
├── multi_agent_6ch_comm/
│   └── multi_agent_FINAL.pth # Phase 4 output
├── qmix_5ch/
│   └── qmix_FINAL.pth        # Phase 5 output
└── qmix_6ch_comm/
    └── qmix_FINAL.pth        # Phase 6 output
```

### Organizing Experiments

```bash
# Phase 1: Train single-agent 5ch
python train_fcn.py --episodes 2000 --checkpoint-dir checkpoints/single_agent_5ch

# Phase 2: Train single-agent 6ch
python train_fcn.py --episodes 2000 --use-6ch --checkpoint-dir checkpoints/single_agent_6ch

# Phase 3: Multi-agent from 5ch checkpoint
python train_multi_agent.py --episodes 400 --agents 4 \
    --resume-from checkpoints/single_agent_5ch/fcn_final.pt \
    --experiment-name multi_agent_5ch_transfer

# Phase 4: Multi-agent 6ch from 6ch checkpoint
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/single_agent_6ch/fcn_final.pt \
    --experiment-name multi_agent_6ch_comm_transfer

# Phase 5: QMIX from 5ch checkpoint
python train_qmix.py --episodes 400 --agents 4 \
    --resume-from checkpoints/single_agent_5ch/fcn_final.pt \
    --experiment-name qmix_5ch_transfer

# Phase 6: QMIX 6ch from 6ch checkpoint
python train_qmix.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/single_agent_6ch/fcn_final.pt \
    --experiment-name qmix_6ch_comm_transfer
```

---

## Ablation Studies

### Experiment 1: Transfer vs From-Scratch

**Baseline**: Train multi-agent from scratch
```bash
python train_multi_agent.py --episodes 800 --agents 4
```

**Transfer**: Use pre-trained checkpoint
```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --resume-from checkpoints/fcn_final.pt
```

**Compare**: Coverage at episode 400 (transfer should be higher)

---

### Experiment 2: 5ch vs 6ch Transfer

**5-channel transfer**:
```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --resume-from checkpoints/single_agent_5ch/fcn_final.pt
```

**6-channel transfer**:
```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/single_agent_6ch/fcn_final.pt
```

**Hypothesis**: 6ch should achieve 5-10% higher coverage due to proactive coordination

---

### Experiment 3: Communication Protocol Impact

Test all protocols with 6ch transfer:

```bash
# No communication
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol none \
    --resume-from checkpoints/single_agent_6ch/fcn_final.pt

# Full state sharing
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/single_agent_6ch/fcn_final.pt

# Attention-based
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol attention \
    --resume-from checkpoints/single_agent_6ch/fcn_final.pt

# CommNet
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol commnet \
    --resume-from checkpoints/single_agent_6ch/fcn_final.pt
```

**Expected Ranking**: full_state ≥ attention ≥ commnet > none

---

## Troubleshooting

### Issue 1: Channel Mismatch Error
```
RuntimeError: size mismatch for encoder.0.weight
```

**Solution**: Ensure checkpoint channels match training mode
```bash
# Check checkpoint channels
python -c "import torch; ckpt=torch.load('checkpoints/fcn_final.pt'); print(ckpt['policy_net_state_dict']['encoder.0.weight'].shape)"
# Output: torch.Size([64, 7, 3, 3])  ← 7 = 5 input + 2 CoordConv (5ch checkpoint)
# Output: torch.Size([64, 8, 3, 3])  ← 8 = 6 input + 2 CoordConv (6ch checkpoint)
```

### Issue 2: No Performance Improvement with Transfer
**Possible causes**:
1. Single-agent checkpoint not fully trained (< 1000 episodes)
2. Learning rate too high (destroys pre-trained weights)
3. Exploration too high (agents unlearn coverage behavior)

**Solution**: Check epsilon decay and reduce learning rate
```bash
# Reduce learning rate for fine-tuning
# Edit config.py: LEARNING_RATE = 1e-4  (instead of 3e-4)
```

### Issue 3: Multi-Agent Performance Worse Than Single-Agent
**Likely cause**: Coordination penalty (collisions, redundant coverage)

**Solution**: Enable collision avoidance and communication
```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/fcn_final.pt
```

---

## Best Practices

### 1. Always Train Single-Agent First
Don't skip Phase 1 and 2! Pre-trained weights are essential.

### 2. Match Checkpoint Channels to Training Mode
5ch → 5ch, 6ch → 6ch. No mixing.

### 3. Use Shorter Multi-Agent Training
With transfer: 400 episodes sufficient
Without transfer: 800+ episodes needed

### 4. Enable Communication for 6ch
If using `--use-6ch`, always specify `--comm-protocol` (at least `full_state`)

### 5. Monitor Early Performance
With good transfer, agents should achieve 60%+ coverage in first 50 episodes

---

## Summary Commands

```bash
# ============================================================================
# COMPLETE PIPELINE WITH TRANSFER LEARNING
# ============================================================================

# PHASE 1: Single-agent baseline (5ch) - 2000 episodes
python train_fcn.py --episodes 2000 --checkpoint-dir checkpoints/single_5ch

# PHASE 2: Single-agent with 6ch (dummy) - 2000 episodes
python train_fcn.py --episodes 2000 --use-6ch --checkpoint-dir checkpoints/single_6ch

# PHASE 3: Multi-agent independent (5ch) - TRANSFER from Phase 1
python train_multi_agent.py --episodes 400 --agents 4 \
    --resume-from checkpoints/single_5ch/fcn_final.pt

# PHASE 4: Multi-agent independent (6ch+comm) - TRANSFER from Phase 2
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/single_6ch/fcn_final.pt

# PHASE 5: QMIX (5ch) - TRANSFER from Phase 1
python train_qmix.py --episodes 400 --agents 4 \
    --resume-from checkpoints/single_5ch/fcn_final.pt

# PHASE 6: QMIX (6ch+comm) - TRANSFER from Phase 2 ⭐ BEST
python train_qmix.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/single_6ch/fcn_final.pt
```

**Total Training Time (with transfer)**: ~20-24 hours
**Total Training Time (without transfer)**: ~40-50 hours

**Speedup**: 2x faster with better final performance! 🚀
