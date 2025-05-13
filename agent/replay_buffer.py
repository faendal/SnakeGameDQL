"""
Replay buffer for storing and sampling transitions in DQN training.
"""

import torch
import random
import numpy as np
from collections import deque
from typing import Deque, Tuple

torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.

    Attributes:
        capacity: Maximum number of transitions to store.
        buffer: Deque storing the transitions.
        device: PyTorch device for tensors.
    """

    def __init__(self, capacity: int) -> None:
        """Initialize a ReplayBuffer.

        Args:
            capacity: Maximum number of transitions to store in the buffer.

        Raises:
            ValueError: If capacity is not a positive integer.
        """
        if capacity <= 0:
            raise ValueError(f"ReplayBuffer capacity must be positive, got {capacity}")
        self.capacity: int = capacity
        self.buffer: Deque[Tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.device: torch.device = torch_device

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Save a transition."""
        try:
            self.buffer.append((state, action, reward, next_state, done))
        except Exception as e:
            raise RuntimeError(f"Failed to push to ReplayBuffer: {e}")

    def sample(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a batch of transitions and convert to torch tensors."""
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if batch_size > len(self.buffer):
            raise ValueError(
                f"Cannot sample {batch_size} from buffer with {len(self.buffer)} elements"
            )
        try:
            transitions = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*transitions)

            states_tensor = torch.from_numpy(np.stack(states)).float().to(self.device)
            actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
            rewards_tensor = torch.tensor(
                rewards, dtype=torch.float32, device=self.device
            )
            next_states_tensor = (
                torch.from_numpy(np.stack(next_states)).float().to(self.device)
            )
            dones_tensor = torch.tensor(dones, dtype=torch.bool, device=self.device)

            return (
                states_tensor,
                actions_tensor,
                rewards_tensor,
                next_states_tensor,
                dones_tensor,
            )
        except Exception as e:
            raise RuntimeError(f"Failed to sample from ReplayBuffer: {e}")

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return len(self.buffer)
