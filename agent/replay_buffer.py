from collections import deque, namedtuple
from random import sample
from typing import Deque, Tuple

import torch


Transition = namedtuple(
    "Transition", ("state", "action", "reward", "next_state", "done")
)


class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples for DQN training.

    This buffer holds the most recent `capacity` transitions and supports
    sampling a random batch for learning.

    Attributes:
        capacity (int): Maximum number of transitions to store.
        buffer (Deque[Transition]): Deque holding the stored transitions.
    """

    def __init__(self, capacity: int) -> None:
        """
        Initializes the replay buffer.

        Args:
            capacity: Maximum number of transitions to store.
        """
        self.capacity: int = capacity
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: float,
        next_state: torch.Tensor,
        done: bool,
    ) -> None:
        """
        Saves a transition.

        Args:
            state: Current state tensor.
            action: Action tensor taken at this state.
            reward: Reward received after taking the action.
            next_state: Next state tensor after the action.
            done: True if the episode terminated after this transition.
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(
        self, batch_size: int, device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples a batch of transitions, converting them into stacked tensors.

        Args:
            batch_size: Number of transitions to sample.
            device: Device on which to return the tensors (e.g., cuda or cpu).

        Returns:
            A tuple of five tensors:
                states      (batch_size, *state_shape)
                actions     (batch_size, 1)
                rewards     (batch_size, 1)
                next_states (batch_size, *state_shape)
                dones       (batch_size, 1)
        """
        transitions = sample(self.buffer, batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.stack(batch.state).to(device)
        actions = torch.cat(batch.action, dim=0).to(device)
        rewards = torch.tensor(
            batch.reward, dtype=torch.float32, device=device
        ).unsqueeze(1)
        next_states = torch.stack(batch.next_state).to(device)
        dones = torch.tensor(batch.done, dtype=torch.uint8, device=device).unsqueeze(1)

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        """
        Returns the current size of internal memory.

        Returns:
            Number of transitions currently stored.
        """
        return len(self.buffer)
