import time
import torch
import seaborn as sns
from Models.Agent import Agent
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from Models.Environment import Environment
from IPython.display import display, clear_output

sns.set_theme(style="darkgrid")
sns.set_palette("Set1")


class GamePlayer:
    def __init__(self, model_path="snake_dqn.pth", grid_size=20, fps=50):
        self.env = Environment(grid_size=grid_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fps = fps
        self.agent = Agent(
            state_shape=(1, grid_size, grid_size), action_size=4, device=self.device
        )
        self.agent.qnetwork_local.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.agent.qnetwork_local.eval()
        self.cmap = mcolors.ListedColormap(["white", "blue", "green", "red"])
        self.bounds = [0, 0.5, 1.5, 2.5, 3.5]
        self.norm = mcolors.BoundaryNorm(self.bounds, self.cmap.N)

    def play(self, episodes=1):
        for ep in range(episodes):
            state = self.env.reset()
            done = False
            fig, ax = plt.subplots()
            img = ax.imshow(state[0], cmap=self.cmap, norm=self.norm)
            ax.set_title("Snake Game")
            plt.axis("off")

            while not done:
                action = self.agent.act(state, eps=0.0)
                next_state, reward, done = self.env.step(action)
                state = next_state
                img.set_data(state[0])
                clear_output(wait=True)
                display(fig)
                time.sleep(1.0 / self.fps)

            print(f"Episode {ep + 1} finished with score: {self.env.score}")
            plt.close()
