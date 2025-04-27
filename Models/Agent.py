import torch
import random
import torch.nn.functional as F
from Models.QNetwork import QNetwork
from torch.optim.lr_scheduler import StepLR
from Models.ReplayBuffer import ReplayBuffer


class Agent:
    def __init__(
        self,
        state_shape,
        action_size,
        device,
        buffer_size=100000,
        batch_size=128,
        gamma=0.99,
        lr=1e-3,
        tau=5e-3,
        update_every=4,
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.update_every = update_every

        # Q-networks
        self.qnetwork_local = QNetwork(action_size).to(device)
        self.qnetwork_target = QNetwork(action_size).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

        # Optimizer + scheduler
        self.optimizer = torch.optim.Adam(
            self.qnetwork_local.parameters(), lr=lr, weight_decay=1e-5
        )
        self.lr_scheduler = StepLR(self.optimizer, step_size=1000, gamma=0.5)

        # Replay buffer
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, device)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # store and learn
        self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    def act(self, state, eps=0.0):
        # state shape: (3, H, W)
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            qvals = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if random.random() > eps:
            return int(qvals.argmax(dim=1).item())
        else:
            return random.randrange(self.qnetwork_local.fc_adv.out_features)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        states = torch.from_numpy(states).float().to(self.device)
        next_states = torch.from_numpy(next_states).float().to(self.device)

        with torch.no_grad():
            best_next = self.qnetwork_local(next_states).argmax(1, keepdim=True)
            q_next = self.qnetwork_target(next_states).gather(1, best_next)

        q_targets = rewards + (self.gamma * q_next * (1 - dones))
        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.smooth_l1_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=0.5)
        self.optimizer.step()

    def soft_update(self):
        for tp, lp in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            tp.data.copy_(self.tau * lp.data + (1.0 - self.tau) * tp.data)

    def step_scheduler(self):
        self.lr_scheduler.step()
