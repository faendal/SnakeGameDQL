import torch
import random
import numpy as np
from collections import deque, namedtuple


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and sampling experiences.
    """

    def __init__(self, action_size: int, buffer_size: int, batch_size: int):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_size: int = action_size
        self.memory: deque = deque(maxlen=buffer_size)
        self.batch_size: int = batch_size
        self.experiences: namedtuple = namedtuple(
            "Experience",
            field_names=["state", "action", "reward", "next_state", "done"],
        )

    def add(self, state, action, reward, next_state, done):
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = [e.state for e in experiences if e is not None]
        actions = (
            torch.from_numpy(
                np.vstack([e.action for e in experiences if e is not None])
            )
            .long()
            .to(self.device)
        )
        rewards = (
            torch.from_numpy(
                np.vstack([e.reward for e in experiences if e is not None])
            )
            .float()
            .to(self.device)
        )
        next_states = [e.next_state for e in experiences if e is not None]
        dones = (
            torch.from_numpy(
                np.vstack([e.done for e in experiences if e is not None]).astype(
                    np.uint8
                )
            )
            .float()
            .to(self.device)
        )
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
