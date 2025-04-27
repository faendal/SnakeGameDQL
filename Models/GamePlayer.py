import time
import torch
import numpy as np
import seaborn as sns
from Models.Agent import Agent
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Models.Environment import Environment
from IPython.display import display, clear_output

sns.set_theme(style="darkgrid")
sns.set_palette("Set1")


class GamePlayer:
    def __init__(self, model_path="snake_dqn.pth", grid_size=20, fps=10):
        self.env = Environment(grid_size=grid_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = fps

        # Load trained agent
        self.agent = Agent(
            state_shape=(3, grid_size, grid_size), action_size=4, device=self.device
        )
        self.agent.qnetwork_local.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.agent.qnetwork_local.eval()

        # Color map: 0=empty, 1=head, 2=body, 3=food
        self.cmap = mcolors.ListedColormap(["white", "blue", "green", "red"])
        self.bounds = [0, 0.5, 1.5, 2.5, 3.5]
        self.norm = mcolors.BoundaryNorm(self.bounds, self.cmap.N)

    def _decode_state(self, state):
        # state: (3, H, W); head/body/food one-hot
        head = state[0]
        body = state[1]
        food = state[2]
        grid = np.zeros_like(head, dtype=np.float32)
        grid[body.astype(bool)] = 2
        grid[head.astype(bool)] = 1
        grid[food.astype(bool)] = 3
        return grid

    def play(self, episodes=1):
        for ep in range(1, episodes + 1):
            state = self.env.reset()
            done = False
            grid = self._decode_state(state)

            fig, ax = plt.subplots()
            img = ax.imshow(grid, cmap=self.cmap, norm=self.norm)
            ax.set_title(f"Episode {ep}")
            plt.axis("off")

            total_reward = 0.0
            while not done:
                action = self.agent.act(state, eps=0.0)
                next_state, reward, done = self.env.step(action)
                state = next_state
                total_reward += reward

                grid = self._decode_state(state)
                img.set_data(grid)
                clear_output(wait=True)
                display(fig)
                time.sleep(1.0 / self.fps)

            print(f"Episode {ep} finished with score: {self.env.score}")
            plt.close(fig)
