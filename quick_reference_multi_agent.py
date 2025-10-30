"""
Quick Reference: Multi-Agent Training with Early Termination

This script shows how to use the new multi-agent curriculum and early termination.
"""

from multi_agent_curriculum import (
    get_multi_agent_curriculum_phase,
    get_multi_agent_epsilon,
    get_multi_agent_map_type,
    print_multi_agent_curriculum_summary
)

def main():
    """Print curriculum summary and examples."""
    
    # Print full curriculum
    print_multi_agent_curriculum_summary()
    
    # Example: Get settings for episode 500
    print("\n" + "="*80)
    print("EXAMPLE: Episode 500 Settings")
    print("="*80)
    
    episode = 500
    phase = get_multi_agent_curriculum_phase(episode)
    epsilon = get_multi_agent_epsilon(episode)
    map_type = get_multi_agent_map_type(episode)
    
    print(f"\nEpisode {episode}:")
    print(f"  Phase: {phase.description}")
    print(f"  Epsilon: {epsilon:.3f}")
    print(f"  Sample Map Type: {map_type}")
    print(f"  Coverage Target: {phase.coverage_target:.1%}")
    print(f"  Overlap Target: ≤{phase.overlap_target:.1%}")
    
    # Example: Integration with training loop
    print("\n" + "="*80)
    print("INTEGRATION EXAMPLE: Training Loop")
    print("="*80)
    
    print("""
# In your training script:

from multi_agent_curriculum import (
    get_multi_agent_epsilon,
    get_multi_agent_map_type,
    get_multi_agent_curriculum_phase
)
from multi_agent_env import MultiAgentCoverageEnv
from config import config

# Initialize environment with early termination enabled
env = MultiAgentCoverageEnv(num_agents=4, grid_size=20)

# Training loop
for episode in range(1000):
    # Get curriculum settings
    epsilon = get_multi_agent_epsilon(episode)
    map_type = get_multi_agent_map_type(episode)
    phase = get_multi_agent_curriculum_phase(episode)
    
    # Reset environment with curriculum map type
    state = env.reset(map_type=map_type)
    
    done = False
    episode_reward = 0
    episode_steps = 0
    
    while not done:
        # Select actions (epsilon-greedy)
        actions = select_actions(state, epsilon)
        
        # Step environment
        next_state, rewards, done, info = env.step(actions)
        
        episode_reward += sum(rewards)
        episode_steps += 1
        
        # Store transitions
        store_transition(state, actions, rewards, next_state, done)
        
        # Train agents
        if should_train():
            train_step()
        
        state = next_state
    
    # Log episode results
    print(f"Episode {episode}: "
          f"Steps={episode_steps}, "
          f"Coverage={info['coverage_pct']:.2%}, "
          f"Reason={info['termination_reason']}, "
          f"Bonus={info['completion_bonus']:.2f}, "
          f"ε={epsilon:.3f}")
    
    # Check phase progress
    if episode_steps < 300 and info['termination_reason'] == 'early_completion':
        print(f"  ✅ Good coordination! Finished in {episode_steps} steps")
    
    # Track metrics
    track_metric('episode_length', episode_steps)
    track_metric('completion_bonus', info['completion_bonus'])
    track_metric('termination_reason', info['termination_reason'])
    track_metric('coverage_target', phase.coverage_target)
    track_metric('overlap_target', phase.overlap_target)
""")
    
    # Early termination settings
    print("\n" + "="*80)
    print("EARLY TERMINATION SETTINGS")
    print("="*80)
    print(f"""
Current Configuration (config.py):
  ENABLE_EARLY_TERMINATION_MULTI: {config.ENABLE_EARLY_TERMINATION_MULTI}
  EARLY_TERM_COVERAGE_TARGET_MULTI: {config.EARLY_TERM_COVERAGE_TARGET_MULTI}
  EARLY_TERM_MIN_STEPS_MULTI: {config.EARLY_TERM_MIN_STEPS_MULTI}
  EARLY_TERM_COMPLETION_BONUS: {config.EARLY_TERM_COMPLETION_BONUS}
  EARLY_TERM_TIME_BONUS_PER_STEP: {config.EARLY_TERM_TIME_BONUS_PER_STEP}

Completion Bonus Formula:
  bonus = FLAT_BONUS + (steps_saved × TIME_BONUS_PER_STEP)
  bonus = {config.EARLY_TERM_COMPLETION_BONUS} + ((350 - steps_used) × {config.EARLY_TERM_TIME_BONUS_PER_STEP})

Examples:
  Finish at step 200: bonus = {config.EARLY_TERM_COMPLETION_BONUS} + (150 × {config.EARLY_TERM_TIME_BONUS_PER_STEP}) = {config.EARLY_TERM_COMPLETION_BONUS + 150 * config.EARLY_TERM_TIME_BONUS_PER_STEP:.2f}
  Finish at step 250: bonus = {config.EARLY_TERM_COMPLETION_BONUS} + (100 × {config.EARLY_TERM_TIME_BONUS_PER_STEP}) = {config.EARLY_TERM_COMPLETION_BONUS + 100 * config.EARLY_TERM_TIME_BONUS_PER_STEP:.2f}
  Finish at step 300: bonus = {config.EARLY_TERM_COMPLETION_BONUS} + (50 × {config.EARLY_TERM_TIME_BONUS_PER_STEP}) = {config.EARLY_TERM_COMPLETION_BONUS + 50 * config.EARLY_TERM_TIME_BONUS_PER_STEP:.2f}
""")

    # Monitoring metrics
    print("\n" + "="*80)
    print("KEY MONITORING METRICS")
    print("="*80)
    print("""
Track these metrics during training:

1. Episode Length Distribution
   - Target: 350 → 250 steps over training
   - Plot: episode_length vs episode_number
   - Good: Decreasing trend, stabilizes around 250

2. Termination Reason Distribution
   - Track: early_completion, high_coverage, max_steps, incomplete
   - Target: early_completion increases, max_steps decreases
   - Good: 60%+ early_completion by episode 1000

3. Completion Bonus Distribution
   - Target: 0 → 15 over training
   - Plot: completion_bonus vs episode_number
   - Good: Increasing trend, stabilizes around 15

4. Coverage at Termination
   - Target: 90-92% for early_completion episodes
   - Good: Consistent 90%+ without quality degradation

5. Coordination Metrics (New)
   - Overlap: Should decrease from 20% → 10%
   - Agent spacing: Agents should spread out
   - Doorway collisions: Should decrease in Phase 4-5

6. Phase Progression
   - Track coverage_target and overlap_target per phase
   - Verify map type distribution matches curriculum
   - Monitor epsilon decay matches expected schedule
""")

    # Expected results
    print("\n" + "="*80)
    print("EXPECTED TRAINING RESULTS")
    print("="*80)
    print("""
Phase 1 (Episodes 0-200): Basic Coordination
  - Episode Length: 320-350 steps
  - Completion Rate: 5-10%
  - Coverage: 50%
  - Overlap: 20%
  - Notes: Agents learning to spread out

Phase 2 (Episodes 200-400): With Obstacles
  - Episode Length: 300-340 steps
  - Completion Rate: 15-25%
  - Coverage: 55%
  - Overlap: 18%
  - Notes: Coordinate around obstacles

Phase 3 (Episodes 400-600): Mixed Obstacles
  - Episode Length: 280-320 steps
  - Completion Rate: 30-45%
  - Coverage: 58%
  - Overlap: 15%
  - Notes: Maintain coordination under pressure

Phase 4 (Episodes 600-800): Doorways
  - Episode Length: 270-310 steps
  - Completion Rate: 40-55%
  - Coverage: 60%
  - Overlap: 12%
  - Notes: Learn doorway negotiation (hard!)

Phase 5 (Episodes 800-1000): Generalization
  - Episode Length: 250-290 steps
  - Completion Rate: 55-70%
  - Coverage: 62%
  - Overlap: 10%
  - Notes: Consistent coordination across all map types

Overall Training Time: 6-8 hours (vs 10-12 without early termination)
""")

    # Troubleshooting
    print("\n" + "="*80)
    print("TROUBLESHOOTING COMMON ISSUES")
    print("="*80)
    print("""
Issue 1: No Early Terminations
Symptom: All episodes reach max_steps (350)
Cause: Coverage target too high or coordination not learning
Fix: Lower EARLY_TERM_COVERAGE_TARGET_MULTI to 0.85

Issue 2: Too Many Early Terminations
Symptom: Episodes terminating at 150-180 steps with low coverage
Cause: Coverage target too low, agents gaming system
Fix: Increase EARLY_TERM_COVERAGE_TARGET_MULTI to 0.92

Issue 3: Completion Bonuses Too Large
Symptom: Bonus > 50% of episode reward consistently
Cause: Bonus parameters too high
Fix: Reduce EARLY_TERM_COMPLETION_BONUS to 5.0

Issue 4: Episodes Too Short in Early Training
Symptom: Early terminations before coordination learned
Cause: Min steps too low
Fix: Increase EARLY_TERM_MIN_STEPS_MULTI to 200

Issue 5: No Coordination Improvement
Symptom: Episode length not decreasing over training
Cause: Epsilon decay too slow or curriculum mismatch
Fix: Check epsilon schedule, verify map types match curriculum
""")

if __name__ == "__main__":
    from config import config
    main()
