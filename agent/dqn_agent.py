import math
import random
import logging
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from agent.replay_buffer import ReplayBuffer
from model.q_network import QNetwork


class DQNAgent:
    """
    Deep Q-Network agent for training on SnakeGame.

    Implements an ε-greedy policy, experience replay, and a target network.

    Attributes:
        device: Torch device (cpu or cuda).
        policy_net: Online Q-network.
        target_net: Target Q-network, updated periodically.
        optimizer: Optimizer for the policy network.
        memory: Replay buffer for experience tuples.
        batch_size: Number of samples per training batch.
        gamma: Discount factor.
        eps_start: Initial ε for ε-greedy.
        eps_end: Final ε.
        eps_decay: Decay rate for ε.
        steps_done: Counter of action selections (for ε decay).
        target_update: Number of episodes between target network updates.
        action_size: Number of discrete actions.
        logger: Logger for debug/info.
    """

    def __init__(
        self,
        state_shape: Tuple[int, ...],
        action_size: int,
        device: torch.device,
        memory_size: int = 10_000,
        batch_size: int = 64,
        gamma: float = 0.99,
        lr: float = 1e-3,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_decay: int = 500,
        target_update: int = 10,
    ) -> None:
        """
        Initializes the DQN agent.

        Args:
            state_shape: Shape of each state (e.g., (1, 20, 20)).
            action_size: Number of valid discrete actions.
            device: Torch device to run computations on.
            memory_size: Capacity of the replay buffer.
            batch_size: Mini-batch size for training.
            gamma: Discount factor for future rewards.
            lr: Learning rate for the optimizer.
            eps_start: Starting value of ε in ε-greedy.
            eps_end: Final value of ε.
            eps_decay: Rate at which ε decays.
            target_update: Episodes between target network sync.
        """
        self.device = device
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.steps_done = 0
        self.target_update = target_update

        # Networks
        self.policy_net = QNetwork(state_shape, action_size).to(device)
        self.target_net = QNetwork(state_shape, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(memory_size)

        # Logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def select_action(self, state: torch.Tensor) -> torch.Tensor:
        """
        Selects an action using an ε-greedy policy.

        Args:
            state: Current state tensor, shape matching state_shape.

        Returns:
            A tensor of shape (1, 1) with the chosen action index.
        """
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1.0 * self.steps_done / self.eps_decay
        )
        self.steps_done += 1

        if random.random() > eps_threshold:
            # Exploit: best action
            with torch.no_grad():
                state = state.unsqueeze(0).to(self.device)  # add batch dim
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].view(1, 1)
        else:
            # Explore: random action
            action = torch.tensor(
                [[random.randrange(self.action_size)]],
                device=self.device,
                dtype=torch.long,
            )
        return action

    def optimize_model(self) -> None:
        """
        Samples a batch from memory and performs a single optimization step.
        """
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(
            self.batch_size, self.device
        )

        # Compute Q(s_t, a) using policy network
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1}) using target network
        next_state_values = self.target_net(next_states).max(1)[0].detach().unsqueeze(1)

        # Compute expected Q values
        expected_state_action_values = rewards + (
            self.gamma * next_state_values * (1 - dones.float())
        )

        # Compute loss (MSE)
        loss = nn.functional.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to improve stability
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.logger.debug(f"Optimize_model loss: {loss.item():.4f}")

    def update_target_network(self) -> None:
        """
        Copies the policy network weights to the target network.
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.logger.info("Target network updated.")
