import torch
import numpy as np
import seaborn as sns
from collections import deque
from Models.Agent import Agent
import matplotlib.pyplot as plt
from Models.Environment import Environment

sns.set_theme(style="darkgrid")
sns.set_palette("Set1")


class GameTrainer:
    def __init__(
        self,
        grid_size=20,
        n_episodes=3000,
        max_t=300,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.997,
        device=None,
    ):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.env = Environment(grid_size=grid_size)
        self.agent = Agent(
            state_shape=(3, grid_size, grid_size),
            action_size=4,
            device=self.device,
        )
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.scores = []
        self.scores_window = deque(maxlen=100)

    def train(self):
        eps = self.eps_start
        for i in range(1, self.n_episodes + 1):
            state = self.env.reset()
            total_reward = 0.0

            for _ in range(self.max_t):
                action = self.agent.act(state, eps)
                nxt, reward, done = self.env.step(action)
                self.agent.step(state, action, reward, nxt, done)
                state = nxt
                total_reward += reward
                if done:
                    break

            self.scores_window.append(total_reward)
            self.scores.append(total_reward)

            # linear epsilon decay
            eps = max(
                self.eps_end,
                self.eps_start
                - (i / self.n_episodes) * (self.eps_start - self.eps_end),
            )
            # step the LR scheduler
            self.agent.step_scheduler()

            if i % 50 == 0:
                lr = self.agent.optimizer.param_groups[0]["lr"]
                print(
                    f"Episode {i}\tAvgScore:{np.mean(self.scores_window):.2f}\tLR:{lr:.2e}"
                )

        self.plot_scores()

    def plot_scores(self):
        plt.plot(self.scores)
        plt.xlabel("Episode")
        plt.ylabel("Score")
        plt.title("Training Progress")
        plt.grid(True)
        plt.show()

    def save_model(self, path="snake_dqn.pth"):
        torch.save(self.agent.qnetwork_local.state_dict(), path)

    def load_model(self, path="snake_dqn.pth"):
        self.agent.qnetwork_local.load_state_dict(
            torch.load(path, map_location=self.device)
        )
        self.agent.qnetwork_local.eval()
