# Quick Reference: Transfer Learning Commands

## Phase 3: Multi-Agent Independent with Transfer

### From 5-channel checkpoint:
```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --resume-from checkpoints/fcn_final.pt
```

### From 6-channel checkpoint (with communication):
```bash
python train_multi_agent.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/fcn_final.pt
```

---

## Phase 5: QMIX with Transfer

### From 5-channel checkpoint:
```bash
python train_qmix.py --episodes 400 --agents 4 \
    --resume-from checkpoints/fcn_final.pt
```

### From 6-channel checkpoint (with communication):
```bash
python train_qmix.py --episodes 400 --agents 4 \
    --use-6ch --comm-protocol full_state \
    --resume-from checkpoints/fcn_final.pt
```

---

## Key Points

1. ✅ **Always match checkpoint channels to training mode**
   - 5ch checkpoint → don't use `--use-6ch`
   - 6ch checkpoint → must use `--use-6ch`

2. ✅ **Communication requires 6 channels**
   - If using `--use-6ch`, specify `--comm-protocol`
   - Without communication, 6th channel is wasted

3. ✅ **Transfer saves ~50% training time**
   - From scratch: 800 episodes
   - With transfer: 400 episodes

4. ✅ **All agents get same initial weights**
   - Parameter sharing: one network for all
   - Independent: each agent initialized with same weights, diverge during training
