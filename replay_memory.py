"""
Stratified Replay Memory

Semantic-based experience replay for balanced learning.
"""

import random
from collections import deque
from typing import List, Dict, Tuple, Any

from config import config


class StratifiedReplayMemory:
    """
    Stratified Experience Replay based on reward semantics.
    Ensures balanced learning across different transition types.

    Strata:
        - Coverage (40%): High-value coverage gains
        - Exploration (30%): Knowledge acquisition
        - Failure (20%): Collisions and penalties
        - Neutral (10%): Zero-gain transitions
    """

    def __init__(self,
                 capacity: int = 50000,
                 coverage_frac: float = 0.4,
                 exploration_frac: float = 0.3,
                 failure_frac: float = 0.2,
                 neutral_frac: float = 0.1):

        assert abs(coverage_frac + exploration_frac + failure_frac + neutral_frac - 1.0) < 0.01

        self.capacity = capacity
        self.fractions = {
            "coverage": coverage_frac,
            "exploration": exploration_frac,
            "failure": failure_frac,
            "neutral": neutral_frac
        }

        # Separate buffers for each stratum
        per_stratum_capacity = capacity // 4
        self.coverage_buffer = deque(maxlen=per_stratum_capacity)
        self.exploration_buffer = deque(maxlen=per_stratum_capacity)
        self.failure_buffer = deque(maxlen=per_stratum_capacity)
        self.neutral_buffer = deque(maxlen=per_stratum_capacity)

    def push(self, state, action, reward, next_state, done, info: dict):
        """
        Add transition to appropriate stratum.

        Args:
            info: Dict with keys:
                - 'coverage_gain': int (newly covered cells)
                - 'knowledge_gain': int (newly sensed cells)
                - 'collision': bool
        """
        transition = (state, action, reward, next_state, done, info)

        coverage_gain = info.get('coverage_gain', 0)
        knowledge_gain = info.get('knowledge_gain', 0)
        collision = info.get('collision', False)

        # Classify transition
        if collision or reward < -1.0:
            # Failure (collision, large penalty)
            self.failure_buffer.append(transition)
        elif coverage_gain > 0:
            # Coverage gain (high priority)
            self.coverage_buffer.append(transition)
        elif knowledge_gain > 0:
            # Exploration (new knowledge)
            self.exploration_buffer.append(transition)
        else:
            # Neutral (no gain)
            self.neutral_buffer.append(transition)

    def sample(self, batch_size: int) -> List:
        """
        Sample batch with stratified sampling.
        """
        # Calculate samples from each stratum
        n_coverage = int(batch_size * self.fractions["coverage"])
        n_exploration = int(batch_size * self.fractions["exploration"])
        n_failure = int(batch_size * self.fractions["failure"])
        n_neutral = batch_size - n_coverage - n_exploration - n_failure

        samples = []

        # Sample from each stratum
        samples.extend(self._sample_from_buffer(self.coverage_buffer, n_coverage))
        samples.extend(self._sample_from_buffer(self.exploration_buffer, n_exploration))
        samples.extend(self._sample_from_buffer(self.failure_buffer, n_failure))
        samples.extend(self._sample_from_buffer(self.neutral_buffer, n_neutral))

        # Shuffle to avoid order bias
        random.shuffle(samples)

        return samples

    def _sample_from_buffer(self, buffer: deque, n: int) -> List:
        """Sample n items from buffer (with replacement if needed)."""
        if len(buffer) == 0:
            return []

        if len(buffer) >= n:
            return random.sample(list(buffer), n)
        else:
            # Sample with replacement
            return random.choices(list(buffer), k=n)

    def __len__(self) -> int:
        return (len(self.coverage_buffer) + len(self.exploration_buffer) +
                len(self.failure_buffer) + len(self.neutral_buffer))

    def get_stats(self) -> Dict[str, int]:
        """Get buffer statistics."""
        return {
            "coverage": len(self.coverage_buffer),
            "exploration": len(self.exploration_buffer),
            "failure": len(self.failure_buffer),
            "neutral": len(self.neutral_buffer),
            "total": len(self)
        }


if __name__ == "__main__":
    # Test stratified memory
    memory = StratifiedReplayMemory(capacity=1000)

    # Add some transitions
    for i in range(100):
        state = None  # Dummy
        action = random.randint(0, 8)
        reward = random.uniform(-2, 10)
        next_state = None
        done = False

        # Simulate different types
        if i % 4 == 0:
            info = {'coverage_gain': 1, 'knowledge_gain': 2, 'collision': False}
        elif i % 4 == 1:
            info = {'coverage_gain': 0, 'knowledge_gain': 3, 'collision': False}
        elif i % 4 == 2:
            info = {'coverage_gain': 0, 'knowledge_gain': 0, 'collision': True}
        else:
            info = {'coverage_gain': 0, 'knowledge_gain': 0, 'collision': False}

        memory.push(state, action, reward, next_state, done, info)

    stats = memory.get_stats()
    print(f"✓ StratifiedReplayMemory test:")
    print(f"  Total: {stats['total']}")
    print(f"  Coverage: {stats['coverage']}")
    print(f"  Exploration: {stats['exploration']}")
    print(f"  Failure: {stats['failure']}")
    print(f"  Neutral: {stats['neutral']}")

    # Test sampling
    if len(memory) >= 32:
        batch = memory.sample(32)
        print(f"  Sampled batch size: {len(batch)}")
