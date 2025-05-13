import os
import logging
from typing import List

import numpy as np
import torch

from environment.snake_game import SnakeGame
from agent.dqn_agent import DQNAgent
from utils.config import (
    DEVICE,
    GRID_SIZE,
    MEMORY_SIZE,
    BATCH_SIZE,
    GAMMA,
    LR,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    TARGET_UPDATE,
    MAX_EPISODES,
    MAX_STEPS_PER_EPISODE,
    CHECKPOINT_DIR,
)


def train() -> None:
    """
    Entrena el agente DQN en el entorno SnakeGame.

    Crea el entorno y el agente, ejecuta el bucle de entrenamiento
    por episodios, optimiza el modelo, actualiza la red objetivo y
    guarda checkpoints del mejor y del modelo final.
    """
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    env = SnakeGame(grid_size=GRID_SIZE)
    agent = DQNAgent(
        state_shape=(1, *GRID_SIZE),
        action_size=3,
        device=DEVICE,
        memory_size=MEMORY_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        lr=LR,
        eps_start=EPS_START,
        eps_end=EPS_END,
        eps_decay=EPS_DECAY,
        target_update=TARGET_UPDATE,
    )

    best_score = 0
    scores: List[int] = []

    for episode in range(1, MAX_EPISODES + 1):
        state_np = env.reset()
        state = torch.from_numpy(state_np).float().to(DEVICE)

        total_reward = 0.0

        for _ in range(MAX_STEPS_PER_EPISODE):
            # Selección de acción ε-greedy
            action_tensor = agent.select_action(state)
            action_idx = action_tensor.item()

            # Convertir índice a acción one-hot para el entorno
            action_onehot = np.eye(agent.action_size, dtype=int)[action_idx]
            next_state_np, reward, done = env.step(action_onehot)
            next_state = torch.from_numpy(next_state_np).float().to(DEVICE)

            # Almacenar transición y optimizar
            agent.memory.push(state, action_tensor, reward, next_state, done)
            agent.optimize_model()

            state = next_state
            total_reward += reward

            if done:
                break

        # Actualizar la red objetivo periódicamente
        if episode % TARGET_UPDATE == 0:
            agent.update_target_network()

        scores.append(int(total_reward))
        logger.info(f"Episode {episode}/{MAX_EPISODES} - Score: {int(total_reward)}")

        # Guardar checkpoint del mejor modelo
        if total_reward > best_score:
            best_score = total_reward
            path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
            try:
                torch.save(agent.policy_net.state_dict(), path)
                logger.info(f"New best model (score={best_score}) saved to {path}")
            except Exception as e:
                logger.error(f"Failed to save best model: {e}")

    # Guardar el modelo final
    final_path = os.path.join(CHECKPOINT_DIR, "final_model.pth")
    torch.save(agent.policy_net.state_dict(), final_path)
    logger.info(f"Final model saved to {final_path}")


if __name__ == "__main__":
    train()
