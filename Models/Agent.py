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
        batch_size=256,
        gamma=0.99,
        lr=1e-3,
        tau=5e-3,
        update_every=2,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        self.qnetwork_local = QNetwork(action_size).to(device)
        self.qnetwork_target = QNetwork(action_size).to(device)
        self.optimizer = torch.optim.Adam(
            self.qnetwork_local.parameters(), lr=lr, weight_decay=1e-5
        )

        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.0):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.qnetwork_local.fc2.out_features))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        with torch.no_grad():
            best_actions = self.qnetwork_local(next_states).argmax(1).unsqueeze(1)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions)

        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.smooth_l1_loss(Q_expected, Q_targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(
            target_model.parameters(), local_model.parameters()
        ):
            target_param.data.copy_(
                self.tau * local_param.data + (1.0 - self.tau) * target_param.data
            )
