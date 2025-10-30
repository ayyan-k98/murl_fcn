# Curriculum Redesign: 1000 Episodes

**Total Duration**: 1000 episodes (~6 hours @ 22s/episode)

## Phase Breakdown

| Phase | Episodes | Duration | Map Distribution | Target Coverage |
|-------|----------|----------|------------------|-----------------|
| **1. Foundation** | 0-200 | 200 eps | Empty: 100% | 70% |
| **2. Intro Obstacles** | 200-320 | 120 eps | Empty: 60%, Random: 40% | 70% |
| **3. More Random** | 320-420 | 100 eps | Random: 60%, Empty: 40% | 72% |
| **4. Consolidation 1** | 420-450 | 30 eps | Empty: 50%, Random: 50% | 75% |
| **5. Intro Rooms** | 450-550 | 100 eps | Room: 40%, Empty: 30%, Random: 30% | 72% |
| **6. More Rooms** | 550-630 | 80 eps | Room: 55%, Random: 25%, Empty: 20% | 75% |
| **7. Consolidation 2** | 630-660 | 30 eps | Empty: 35%, Random: 35%, Room: 30% | 78% |
| **8. Intro Corridor** | 660-740 | 80 eps | Room: 45%, Corridor: 25%, Random: 20%, Empty: 10% | 73% |
| **9. Intro Cave** | 740-820 | 80 eps | Room: 35%, Cave: 25%, Corridor: 20%, Random: 20% | 70% |
| **10. Consolidation 3** | 820-850 | 30 eps | Room: 40%, Corridor: 25%, Cave: 20%, Random: 15% | 72% |
| **11. Intro L-Shape** | 850-910 | 60 eps | Room: 30%, Cave: 20%, L-Shape: 20%, Corridor: 15%, Random: 15% | 68% |
| **12. Complex Mix** | 910-960 | 50 eps | Room: 25%, Cave: 20%, L-Shape: 20%, Corridor: 20%, Random: 15% | 70% |
| **13. Final Polish** | 960-1000 | 40 eps | Room: 25%, Empty: 20%, Random: 20%, Cave: 15%, Corridor: 10%, L-Shape: 10% | 72% |

## Key Changes from 2250-Episode Curriculum

### Time Compression
- **Old**: 2250 episodes (~13.75 hours)
- **New**: 1000 episodes (~6 hours)
- **Reduction**: 55% faster

### Phase Distribution
- **Foundation phase**: 500 → 200 episodes (60% reduction)
  - Still sufficient for basic learning with fast epsilon decay (0.98)
  - Agent reaches ε=0.36 by episode 50, ε=0.02 by episode 200
  
- **Early phases** (2-3): 300 → 220 episodes combined
  - Learn obstacle navigation efficiently
  
- **Mid phases** (5-9): 650 → 460 episodes combined
  - Introduce all map types: rooms, corridors, caves
  - Shorter exposure but higher diversity
  
- **Late phases** (11-13): 250 → 150 episodes combined
  - L-shapes, complex mixes, final polish
  - Consolidation phases reduced from 75 → 30 episodes each

### Learning Strategy
- **Same progression**: All 13 phases maintained
- **Same epsilon decay rates**: No changes to exploration strategy
- **Same mastery gates**: Target coverage thresholds unchanged
- **Faster transitions**: Less overlearning, more efficient curriculum

### Expected Outcomes
- **Coverage**: Should reach 70-80% on validation
- **Generalization**: All map types exposed, less overfit risk
- **Training time**: ~6 hours for full curriculum
- **Gradient stability**: With new reward scaling (COVERAGE_REWARD=1.2)

## Training Command

```bash
python train_fcn.py --episodes 1000 --probabilistic
```

## Validation Schedule
- **Interval**: Every 100 episodes
- **Checkpoints**: Episodes 100, 200, 300, ..., 1000
- **Total validations**: 10 validations (vs 23 in old curriculum)

## Expected Timeline
- **Episode 200**: ~70% empty, ~65% random
- **Episode 500**: ~75% empty, ~70% room
- **Episode 800**: ~75% mixed environments
- **Episode 1000**: ~72-80% all map types

## Monitoring
Watch for:
- Gradient norms staying 10-18 (stable)
- Coverage 70-85% (not 95%+ like before)
- Episode rewards ~400-500 (not ~750)
- Smooth phase transitions without performance drops
