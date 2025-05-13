"""Streamlit app to load and play a trained DQN agent for Snake."""

import time
import numpy as np
from PIL import Image
import streamlit as st
from utils.config import DQNConfig
from agent.dqn_agent import DQNAgent
from environment.snake_env import SnakeEnv

# Color mapping for grid values
COLOR_MAP = {
    0: (255, 255, 255),  # empty: white
    1: (0, 0, 255),  # head: blue
    2: (0, 255, 0),  # body: green
    3: (255, 0, 0),  # food: red
}


def grid_to_image(obs: np.ndarray, scale: int = 10) -> Image.Image:
    """Convert grid observation to a PIL Image."""
    h, w = obs.shape
    img_array = np.zeros((h, w, 3), dtype=np.uint8)
    for val, color in COLOR_MAP.items():
        img_array[obs == val] = color
    img = Image.fromarray(img_array)
    return img.resize((w * scale, h * scale), resample=Image.NEAREST)


def main():
    st.sidebar.title("Snake DQN Player")
    checkpoint_default = f"{DQNConfig().checkpoint_dir}/stage2.pth"
    checkpoint_path = st.sidebar.text_input("Checkpoint path", checkpoint_default)
    num_episodes = st.sidebar.number_input(
        "Episodes", min_value=1, max_value=20, value=5
    )
    speed = st.sidebar.slider("Delay (s)", min_value=0.0, max_value=1.0, value=0.1)

    if st.sidebar.button("Play"):
        config = DQNConfig()
        env = SnakeEnv(
            width=config.width,
            height=config.height,
            stage=2,
            max_steps=config.max_steps_stage2,
        )
        agent = DQNAgent(
            state_shape=(config.height, config.width),
            num_actions=config.num_actions,
            replay_buffer_size=config.replay_buffer_size,
            batch_size=config.batch_size,
            gamma=config.gamma,
            lr=config.lr,
            target_update_freq=config.target_update_freq,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=1,
        )
        try:
            agent.load(checkpoint_path)
        except Exception as e:
            st.error(f"Error loading model: {e}")
            return

        placeholder = st.empty()
        for episode in range(1, num_episodes + 1):
            state = env.reset()
            done = False
            total_reward = 0.0

            st.write(f"Episode {episode}/{num_episodes}")
            while not done:
                action = agent.select_action(state)
                state, reward, done, _ = env.step(action)
                total_reward += reward
                img = grid_to_image(state)
                placeholder.image(img)
                time.sleep(speed)

            st.write(f"Total Reward: {total_reward:.2f}")


if __name__ == "__main__":
    main()
