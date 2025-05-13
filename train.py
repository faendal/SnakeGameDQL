"""
Main training script for two-stage curriculum DQN on Snake environment.
"""

import os
import torch
import random
import numpy as np
from agent.dqn_agent import DQNAgent
from environment.snake_env import SnakeEnv
from utils.config import DQNConfig, StageConfig


def train_stage(config: DQNConfig, stage_cfg: StageConfig) -> str:
    """Train the agent for one curriculum stage and save checkpoint.

    Args:
        config: Global DQN configuration.
        stage_cfg: Configuration for this stage.

    Returns:
        Path to the saved checkpoint.
    """
    print(f"Starting Stage {stage_cfg.stage}: {stage_cfg.num_episodes} episodes")

    # Enable cuDNN autotuner for potential speedup on fixed input sizes
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    env = SnakeEnv(
        width=config.width,
        height=config.height,
        stage=stage_cfg.stage,
        max_steps=stage_cfg.env_kwargs.get("max_steps"),
    )

    # Unpack any stage-specific agent_kwargs, falling back to globals
    agent_params = dict(
        state_shape=(config.height, config.width),
        num_actions=config.num_actions,
        replay_buffer_size=config.replay_buffer_size,
        batch_size=config.batch_size,
        gamma=config.gamma,
        lr=config.lr,
        target_update_freq=config.target_update_freq,
        eps_start=config.eps_start,
        eps_end=config.eps_end,
        eps_decay=config.eps_decay,
    )
    agent_params.update(stage_cfg.agent_kwargs or {})

    agent = DQNAgent(**agent_params)

    # Optionally load existing weights
    if stage_cfg.checkpoint_path:
        print(f"Loading checkpoint from {stage_cfg.checkpoint_path}")
        agent.load(stage_cfg.checkpoint_path)

    for episode in range(1, stage_cfg.num_episodes + 1):
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.push_transition(state, action, reward, next_state, done)
            _ = agent.optimize()
            state = next_state
            total_reward += reward

        if episode % 100 == 0:
            print(
                f"Stage {stage_cfg.stage} Episode {episode}/{stage_cfg.num_episodes} "
                f"- Total Reward: {total_reward:.2f}"
            )

    # Save checkpoint
    os.makedirs(os.path.dirname(stage_cfg.output_path), exist_ok=True)
    agent.save(stage_cfg.output_path)
    print(f"Stage {stage_cfg.stage} checkpoint saved to {stage_cfg.output_path}")
    return stage_cfg.output_path


def main():
    """Run the two-stage curriculum training."""
    try:
        # Reproducibility seeds
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        config = DQNConfig()

        # Stage 1: Eat only
        stage1 = StageConfig(
            stage=1,
            env_kwargs={"max_steps": config.max_steps_stage1},
            agent_kwargs={},  # using global config
            num_episodes=config.episodes_stage1,
            checkpoint_path=None,
            output_path=os.path.join(config.checkpoint_dir, "stage1.pth"),
        )
        ckpt1 = train_stage(config, stage1)

        # Stage 2: Eat + Survive
        stage2 = StageConfig(
            stage=2,
            env_kwargs={"max_steps": config.max_steps_stage2},
            agent_kwargs={},
            num_episodes=config.episodes_stage2,
            checkpoint_path=ckpt1,
            output_path=os.path.join(config.checkpoint_dir, "stage2.pth"),
        )
        train_stage(config, stage2)

        print("Training complete.")
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()
