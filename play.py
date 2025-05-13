import time
import argparse
import torch
import numpy as np

from environment.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.config import DEVICE, GRID_SIZE, CHECKPOINT_DIR


def play(model_path: str, episodes: int = 5, delay: float = 0.2) -> None:
    """
    Juega usando el agente DQN cargado desde un checkpoint.

    Args:
        model_path: Ruta al fichero .pth del modelo entrenado.
        episodes: Número de episodios a ejecutar.
        delay: Retraso (en segundos) entre pasos para visualización.
    """
    # Configurar entorno y agente
    env = SnakeGame(grid_size=GRID_SIZE)
    agent = DQNAgent(state_shape=(1, *GRID_SIZE), action_size=3, device=DEVICE)
    agent.policy_net.load_state_dict(torch.load(model_path, map_location=DEVICE))
    agent.policy_net.eval()

    for ep in range(1, episodes + 1):
        state_np = env.reset()
        state = torch.from_numpy(state_np).float().to(DEVICE)
        total_reward = 0.0

        print(f"\n--- Episodio {ep}/{episodes} ---")
        env.render()

        done = False
        while not done:
            # Selección determinística (greedy)
            with torch.no_grad():
                q_vals = agent.policy_net(state.unsqueeze(0))
                action_idx = q_vals.max(1)[1].item()

            action_onehot = np.eye(agent.action_size, dtype=int)[action_idx]
            next_state_np, reward, done = env.step(action_onehot)
            state = torch.from_numpy(next_state_np).float().to(DEVICE)
            total_reward += reward

            env.render()
            time.sleep(delay)

        print(f"Episodio {ep} finalizado con score: {int(total_reward)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Juega al Snake con un agente DQN entrenado"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=f"{CHECKPOINT_DIR}/best_model.pth",
        help="Ruta al checkpoint del modelo",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Número de episodios a jugar"
    )
    parser.add_argument(
        "--delay", type=float, default=0.2, help="Retraso (segundos) entre pasos"
    )
    args = parser.parse_args()
    play(args.model, args.episodes, args.delay)
