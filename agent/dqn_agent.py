"""
DQN Agent implementation for training and acting in the Snake environment.
"""

import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional
from agent.network import QNetwork
from agent.replay_buffer import ReplayBuffer


def torch_device() -> torch.device:
    """Get the torch device, preferring CUDA if available."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """Deep Q-Learning Agent for Snake game."""

    def __init__(
        self,
        state_shape: Tuple[int, int],
        num_actions: int,
        replay_buffer_size: int = 10000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        target_update_freq: int = 1000,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay: int = 10000,
        device: Optional[torch.device] = None,
    ) -> None:
        """Initialize the DQNAgent."""
        try:
            self.device = device or torch_device()
            self.num_actions = num_actions
            self.batch_size = batch_size
            self.gamma = gamma
            self.target_update_freq = target_update_freq

            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay = eps_decay
            self.steps_done = 0

            self.policy_net = QNetwork(state_shape, num_actions).to(self.device)
            self.target_net = QNetwork(state_shape, num_actions).to(self.device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
            self.target_net.eval()

            self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
            self.loss_fn = nn.MSELoss()

            self.memory = ReplayBuffer(replay_buffer_size)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize DQNAgent: {e}")

    def select_action(self, state: np.ndarray) -> int:
        """Select an action using ε-greedy policy."""
        try:
            eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * max(
                0.0, (self.eps_decay - self.steps_done) / self.eps_decay
            )
            self.steps_done += 1

            if random.random() < eps_threshold:
                return random.randrange(self.num_actions)

            state_tensor = torch.tensor(
                state, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
            return int(q_values.argmax(dim=1).item())
        except Exception as e:
            raise RuntimeError(f"Error selecting action: {e}")

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store a transition in replay buffer."""
        try:
            self.memory.push(state, action, reward, next_state, done)
        except Exception as e:
            raise RuntimeError(f"Error pushing transition: {e}")

    def optimize(self) -> Optional[float]:
        """Perform one optimization step on the policy network."""
        if len(self.memory) < self.batch_size:
            return None

        try:
            states, actions, rewards, next_states, dones = self.memory.sample(
                self.batch_size
            )

            q_values = (
                self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            )
            with torch.no_grad():
                next_q = self.target_net(next_states).max(1)[0]

            expected_q = rewards + (self.gamma * next_q * (~dones))

            loss = self.loss_fn(q_values, expected_q)
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
            self.optimizer.step()

            if self.steps_done % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return float(loss.item())
        except Exception as e:
            raise RuntimeError(f"Error during optimization: {e}")

    def save(self, path: str) -> None:
        """Save the policy network checkpoint."""
        try:
            torch.save(self.policy_net.state_dict(), path)
        except Exception as e:
            raise RuntimeError(f"Failed to save model to {path}: {e}")

    def load(self, path: str) -> None:
        """Load a policy network checkpoint."""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.policy_net.load_state_dict(state_dict)
            self.target_net.load_state_dict(state_dict)
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {path}: {e}")
