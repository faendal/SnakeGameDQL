import torch
import random
import numpy as np
import torch.nn.functional as F
from Models.QNetwork import QNetwork
from Models.ReplayBuffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        state_shape,
        action_size,
        device,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        tau=1e-3,
        update_every=4,
    ):
        self.state_shape = state_shape
        self.action_size = action_size
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        # Q-Networks
        self.qnetwork_local = QNetwork(action_size).to(device)
        self.qnetwork_target = QNetwork(action_size).to(device)
        self.optimizer = torch.optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)

        # Learn every `update_every` time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.0):
        state_tensor = (
            torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        )  # (1, 1, 20, 20)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Convert states and next_states to tensors
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(self.device)

        # Get max predicted Q values (for next states) from target model
        with torch.no_grad():
            Q_targets_next = (
                self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
            )

        # Compute Q targets for current states
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
