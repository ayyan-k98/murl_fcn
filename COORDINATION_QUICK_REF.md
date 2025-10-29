# Coordination Metrics Quick Reference

## Reading Coordination Scores

### Score Interpretation
```
0-40:   Poor     - Random exploration, high overlap, many collisions
40-60:  Fair     - Some coordination emerging, still inefficient
60-75:  Good     - Clear territory division, moderate efficiency
75-85:  Excellent- Well-coordinated, balanced load, low overlap
85-100: Perfect  - Near-optimal coordination, minimal redundancy
```

### Training Output Example
```
Ep 100 | Cov: 75.2% | Coord: 71.8/100 | Rew: 305.0 | Len: 110 | Eps: 0.280
  Overlap: 15.2% | Efficiency: 82.1% | Balance: 0.88 | Collisions: 12
```

**Reading this**:
- **Coord 71.8/100**: Good coordination (in "Good" range)
- **Overlap 15.2%**: 15% of covered cells visited by multiple agents (moderate redundancy)
- **Efficiency 82.1%**: Agents covering at 82% of potential rate
- **Balance 0.88**: Load distributed at 88% of perfect balance (0.0=uneven, 1.0=perfect)
- **Collisions 12**: 12 total collisions this episode

---

## Metric Breakdown

### 1. Overlap Ratio (Lower = Better)
- **0-5%**: Excellent (minimal redundancy)
- **5-15%**: Good (some overlap at frontiers)
- **15-30%**: Fair (agents revisiting areas)
- **>30%**: Poor (agents clustering)

**What it means**: % of covered cells visited by multiple agents.

---

### 2. Exploration Efficiency (Higher = Better)
- **>0.90**: Excellent (near-optimal coverage rate)
- **0.75-0.90**: Good (efficient exploration)
- **0.60-0.75**: Fair (moderate efficiency)
- **<0.60**: Poor (slow coverage)

**Formula**: `efficiency = total_coverage / (num_agents * episode_length)`

**What it means**: How much area covered per agent-step.

---

### 3. Spatial Dispersion (Higher = Better)
- **>0.85**: Excellent (agents well spread)
- **0.70-0.85**: Good (moderate spread)
- **0.50-0.70**: Fair (some clustering)
- **<0.50**: Poor (agents bunched together)

**What it means**: How spatially distributed agents are (based on pairwise distances).

---

### 4. Load Balance (Higher = Better)
- **>0.90**: Excellent (nearly equal workload)
- **0.75-0.90**: Good (balanced load)
- **0.60-0.75**: Fair (some imbalance)
- **<0.60**: Poor (some agents idle, others overworked)

**Formula**: `balance = min_coverage / mean_coverage`

**What it means**: How evenly work is distributed (0=one agent does everything, 1=perfect equality).

---

### 5. Frontier Sharing Score (Higher = Better)
- **>0.80**: Excellent (clear territory division)
- **0.60-0.80**: Good (moderate division)
- **0.40-0.60**: Fair (overlapping territories)
- **<0.40**: Poor (no clear territories)

**What it means**: How well agents divide frontier exploration.

---

### 6. Collisions (Lower = Better)
- **0-5**: Excellent (minimal collisions)
- **5-15**: Good (some collisions)
- **15-30**: Fair (many collisions)
- **>30**: Poor (collision-heavy)

**Types**:
- **Agent-Agent**: Agents colliding with each other
- **Agent-Obstacle**: Agents hitting walls/obstacles
- **Near Misses**: Close calls (same cell, different time)

---

### 7. Communication Effectiveness
Only relevant when using communication protocols.

- **Influenced Decisions >20%**: Effective (messages change actions)
- **Influenced Decisions 10-20%**: Moderate (some impact)
- **Influenced Decisions <10%**: Ineffective (messages ignored)

**What it means**: % of actions influenced by received messages.

---

## Common Patterns

### Pattern 1: High Coverage, Low Coordination
```
Cov: 82.3% | Coord: 58.2/100
  Overlap: 38.5% ← PROBLEM
  Efficiency: 0.76
  Balance: 0.68
```
**Diagnosis**: Agents covering a lot, but with massive redundancy.  
**Fix**: Add 6th channel (agent occupancy), increase communication range.

---

### Pattern 2: Good Coordination, Low Coverage
```
Cov: 62.1% | Coord: 81.5/100
  Overlap: 7.2%
  Efficiency: 0.89
  Balance: 0.94
```
**Diagnosis**: Agents coordinating well but too cautious/slow.  
**Fix**: Increase exploration reward, lower collision penalty, train longer.

---

### Pattern 3: Unbalanced Load
```
Cov: 75.8% | Coord: 65.3/100
  Overlap: 12.1%
  Efficiency: 0.85
  Balance: 0.58 ← PROBLEM
```
**Diagnosis**: Some agents working hard, others idle.  
**Fix**: Reward shaping for balanced coverage, QMIX (joint value).

---

### Pattern 4: Collision-Heavy
```
Cov: 78.2% | Coord: 68.7/100
  Overlap: 15.2%
  Efficiency: 0.88
  Balance: 0.82
  Collisions: 45 ← PROBLEM
```
**Diagnosis**: Too aggressive, not avoiding obstacles/each other.  
**Fix**: Enable collision avoidance, lower epsilon, add collision penalty.

---

## Comparing Experiments

### Example: 5-Channel vs 6-Channel

**5-Channel (Phase 3)**:
```
Validation Results:
  Mean Coverage: 76.2% (±4.1%)
  Mean Coordination Score: 72.8/100 (±6.3)
    Overlap: 18.5%
    Efficiency: 0.84
    Balance: 0.87
    Collisions: 14.2
```

**6-Channel (Phase 4)**:
```
Validation Results:
  Mean Coverage: 81.5% (±3.5%)
  Mean Coordination Score: 84.3/100 (±4.2)
    Overlap: 8.1%  ← -10.4% improvement
    Efficiency: 0.91  ← +8.3% improvement
    Balance: 0.94  ← +8.0% improvement
    Collisions: 5.8  ← -59% reduction
```

**Conclusion**: 6th channel (agent occupancy) significantly improves coordination by reducing overlap and collisions.

---

### Example: Independent vs QMIX

**Independent (Phase 4)**:
```
Mean Coverage: 81.5%
Mean Coordination Score: 84.3/100
  Balance: 0.91
  Overlap: 8.1%
```

**QMIX (Phase 6)**:
```
Mean Coverage: 85.6%
Mean Coordination Score: 89.1/100
  Balance: 0.98  ← +7.7% improvement
  Overlap: 5.2%  ← -35.8% reduction
```

**Conclusion**: QMIX's joint value function improves load balancing and reduces redundancy.

---

## Debugging Low Coordination

### If Coord < 60

1. **Check overlap**: If >25%, agents clustering
   - Add 6th channel
   - Increase communication range
   - Use attention-based communication

2. **Check balance**: If <0.70, uneven workload
   - Use QMIX (joint value)
   - Add reward shaping for balanced coverage
   - Longer training (coordination takes time to emerge)

3. **Check collisions**: If >20, too aggressive
   - Enable collision avoidance
   - Increase collision penalty
   - Lower epsilon (more exploitation)

4. **Check efficiency**: If <0.75, slow exploration
   - Increase exploration bonus
   - Use curriculum learning
   - Check if agents getting stuck

---

## Expected Progression

### Early Training (Ep 0-100)
```
Coord: 45-60
  - Random exploration
  - High overlap (30-50%)
  - Many collisions (20-40)
  - Poor balance (0.60-0.75)
```

### Mid Training (Ep 100-200)
```
Coord: 60-75
  - Patterns emerging
  - Moderate overlap (15-30%)
  - Fewer collisions (10-20)
  - Better balance (0.75-0.85)
```

### Late Training (Ep 200-400)
```
Coord: 75-85
  - Clear coordination
  - Low overlap (5-15%)
  - Minimal collisions (5-10)
  - Good balance (0.85-0.95)
```

### Converged (Ep 400+)
```
Coord: 85-90
  - Mature coordination
  - Minimal overlap (3-8%)
  - Very few collisions (2-5)
  - Excellent balance (0.90-0.98)
```

---

## Validation Checklist

When validating trained agents:

- [ ] Coordination score >75 (good coordination)
- [ ] Overlap ratio <15% (low redundancy)
- [ ] Balance ratio >0.85 (even workload)
- [ ] Collisions <10 per episode (safe navigation)
- [ ] Efficiency >0.85 (fast coverage)
- [ ] Coverage >75% (effective exploration)

**If all checkmarks**: Agent coordination is excellent ✅

---

## Quick Commands

### Train with Coordination Tracking
```bash
# Multi-agent (Phase 3)
python train_multi_agent.py --episodes 400 --agents 4 --use-curriculum --resume-from checkpoints/single_5ch/fcn_final.pt

# Multi-agent + 6ch (Phase 4)
python train_multi_agent.py --episodes 400 --agents 4 --use-6ch --comm-protocol full_state --use-curriculum --resume-from checkpoints/single_6ch/fcn_final.pt

# QMIX + 6ch (Phase 6)
python train_qmix.py --episodes 400 --agents 4 --use-6ch --comm-protocol full_state --use-curriculum --resume-from checkpoints/single_6ch/fcn_final.pt
```

### Watch Coordination During Training
```bash
# Look for these lines every LOG_FREQ episodes:
Ep 100 | Cov: 75.2% | Coord: 71.8/100 | ...

# And detailed breakdown every 5*LOG_FREQ:
  Overlap: 15.2% | Efficiency: 82.1% | Balance: 0.88 | Collisions: 12
```

### Check Final Coordination
```bash
# In validation output:
Validation Results:
  Mean Coordination Score: 84.3/100 (±4.2)
```

---

## Summary

**Coordination Score = Overall Quality (0-100)**

Combines:
- 30% Exploration Efficiency (coverage rate)
- 20% Load Balance (even workload)
- 20% Spatial Dispersion (spread out)
- 15% Frontier Sharing (territory division)
- 10% Collision Avoidance (safety)
- 5% Communication Effectiveness (message usage)

**Good coordination** = High coverage with low redundancy, balanced workload, minimal collisions.

**Poor coordination** = Agents clustering, uneven work, high overlap, many collisions.

---

**Goal**: Achieve Coord >85 with Coverage >80% 🎯
