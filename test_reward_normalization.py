"""Test reward normalization"""

from multi_agent_env import MultiAgentCoverageEnv
from config import config
import numpy as np
import random

print('=' * 70)
print('REWARD NORMALIZATION VERIFICATION')
print('=' * 70)

# Show config
print('\nConfiguration:')
print(f'  Normalize by N agents: {config.MULTI_AGENT_REWARD_NORMALIZE_BY_N}')
print(f'  Scale factor: {config.MULTI_AGENT_REWARD_SCALE_FACTOR}')
print(f'  Clip min: {config.MULTI_AGENT_REWARD_CLIP_MIN}')
print(f'  Clip max: {config.MULTI_AGENT_REWARD_CLIP_MAX}')

# Create environment
env = MultiAgentCoverageEnv(num_agents=4, grid_size=20)
state = env.reset()

# Run episode
episode_rewards = []
for step in range(50):
    actions = [random.randint(0, 8) for _ in range(4)]
    _, rewards, done, info = env.step(actions)
    episode_rewards.extend(rewards)
    if done:
        break

# Statistics
print(f'\nAfter {step+1} steps ({len(episode_rewards)} reward samples):')
print(f'  Min reward: {min(episode_rewards):.3f}')
print(f'  Max reward: {max(episode_rewards):.3f}')
print(f'  Mean reward: {np.mean(episode_rewards):.3f}')
print(f'  Std reward: {np.std(episode_rewards):.3f}')
print(f'  Total: {sum(episode_rewards):.1f}')

# Expected ranges
print('\nExpected ranges (with normalization):')
print('  Per-step: -0.5 to +3.0')
print('  Episode total: 0 to ~100')
print('\nWithout normalization (for comparison):')
print('  Per-step: -2 to +60')
print('  Episode total: 0 to ~20,000')

# Validation
per_step_ok = -1.0 <= min(episode_rewards) and max(episode_rewards) <= 5.0
total_ok = sum(episode_rewards) < 200

if per_step_ok and total_ok:
    print('\n[OK] Reward normalization is working correctly!')
else:
    print('\n[WARNING] Rewards outside expected range!')
    print(f'  Per-step OK: {per_step_ok}')
    print(f'  Total OK: {total_ok}')

print('=' * 70)
