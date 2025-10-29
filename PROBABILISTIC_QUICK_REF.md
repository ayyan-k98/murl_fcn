# Quick Reference: Probabilistic vs Binary Coverage

## Command Examples

### Single-Agent
```bash
# Binary (default)
python train_fcn.py --episodes 400

# Probabilistic  
python train_fcn.py --probabilistic --episodes 400
```

### Multi-Agent
```bash
# Binary (default)
python train_multi_agent.py --agents 4 --episodes 400

# Probabilistic
python train_multi_agent.py --probabilistic --agents 4 --episodes 400
```

## Quick Comparison

| Feature | Binary | Probabilistic |
|---------|--------|---------------|
| **Flag** | None (default) | `--probabilistic` |
| **Coverage/visit** | 100% instant | ~99% first, gradual after |
| **Revisit needed?** | No | Yes (to reach 100%) |
| **Reward scale** | 1.0x | 0.15x |
| **Training speed** | Faster ⚡ | Slower 🐢 |
| **Difficulty** | Easier | Harder |

## When to Use Which?

### Use Binary (Default)
- Standard coverage tasks
- Faster training needed
- Quick baseline experiments
- Most research papers use this

### Use Probabilistic  
- Research on thorough coverage
- Comparing coverage strategies
- You trained single-agent with it
- Want harder challenge

## Important Notes

⚠️ **Don't mix modes** - If you trained single-agent with binary, use binary for multi-agent  
⚠️ **Checkpoints care about mode** - Binary checkpoint → binary training, Prob → prob  
⚠️ **Reward scaling is automatic** - No need to manually adjust  

## Your Previous Training

You mentioned: "we trained single_agent on probabilistic as well"

If you have checkpoints from probabilistic single-agent training:
```bash
# Use them with probabilistic multi-agent
python train_multi_agent.py \
  --probabilistic \
  --resume-from checkpoints/single_5ch/fcn_final.pt \
  --agents 4 \
  --episodes 400
```

## Phase 3-6 Training Commands

### Original Plan (Binary)
```bash
# Phase 3: 5ch baseline
python train_multi_agent.py --episodes 400 --agents 4 --use-curriculum

# Phase 4: 6ch + comm
python train_multi_agent.py --episodes 400 --agents 4 --use-6ch --comm-protocol full_state --use-curriculum

# Phase 6: QMIX
python train_qmix.py --episodes 400 --agents 4 --use-6ch --comm-protocol full_state --use-curriculum
```

### With Probabilistic
```bash
# Phase 3: 5ch baseline (probabilistic)
python train_multi_agent.py --probabilistic --episodes 400 --agents 4 --use-curriculum

# Phase 4: 6ch + comm (probabilistic)
python train_multi_agent.py --probabilistic --episodes 400 --agents 4 --use-6ch --comm-protocol full_state --use-curriculum

# Phase 6: QMIX (probabilistic)
python train_qmix.py --probabilistic --episodes 400 --agents 4 --use-6ch --comm-protocol full_state --use-curriculum
```

---
**Status**: ✅ Fully implemented, ready to use!
