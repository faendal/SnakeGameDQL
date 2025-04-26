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
        n_episodes=10000,
        max_t=300,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=0.995,
        update_every=4,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        tau=1e-3,
    ):
        self.grid_size = grid_size
        self.n_episodes = n_episodes
        self.max_t = max_t
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = Environment(grid_size=grid_size)
        state_shape = (1, grid_size, grid_size)
        action_size = 4  # Up, Right, Down, Left

        self.agent = Agent(
            state_shape=state_shape,
            action_size=action_size,
            device=self.device,
            buffer_size=buffer_size,
            batch_size=batch_size,
            gamma=gamma,
            lr=lr,
            tau=tau,
            update_every=update_every,
        )

        self.scores = []
        self.scores_window = deque(maxlen=100)

    def train(self):
        eps = self.eps_start

        for i_episode in range(1, self.n_episodes + 1):
            state = self.env.reset()
            total_reward = 0

            for t in range(self.max_t):
                action = self.agent.act(state, eps)
                next_state, reward, done = self.env.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break

            self.scores_window.append(total_reward)
            self.scores.append(total_reward)
            eps = max(self.eps_end, self.eps_decay * eps)

            if i_episode % 50 == 0:
                avg_score = np.mean(self.scores_window)
                print(f"Episode {i_episode}\tAverage Score: {avg_score:.2f}")

        print("Training completed.")
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
