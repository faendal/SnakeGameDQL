"""
Environment for the Snake game with curriculum-based stages and image rendering.
"""

import os
import random
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any


class SnakeEnv:
    """Snake game environment for DQN training with stage-aware reward shaping and rendering."""

    def __init__(
        self, width: int, height: int, stage: int = 1, max_steps: int = 1000
    ) -> None:
        """Initialize the SnakeEnv."""
        self.width = width
        self.height = height
        self.stage = stage
        self.max_steps = max_steps

        self.snake: list[Tuple[int, int]] = []
        self.direction: Tuple[int, int] = (0, 1)
        self.food_pos: Tuple[int, int]
        self.step_count: int = 0

        self.reset()

    def reset(self) -> np.ndarray:
        """Reset environment for a new episode."""
        try:
            self.step_count = 0
            mid = (self.height // 2, self.width // 2)
            self.snake = [mid]
            self.direction = (0, 1)
            self._place_food()
            return self._get_observation()
        except Exception as e:
            raise RuntimeError(f"Error during reset: {e}")

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """Take action, return (observation, reward, done, info)."""
        try:
            self._apply_action(action)
            self.step_count += 1

            head = self.snake[0]
            new_head = (head[0] + self.direction[0], head[1] + self.direction[1])

            # Check collision
            if self._check_collision(new_head):
                reward = -1.0 if self.stage >= 2 else 0.0
                return self._get_observation(), reward, True, {}

            # Move snake
            self.snake.insert(0, new_head)
            if new_head == self.food_pos:
                reward = 1.0
                self._place_food()
            else:
                self.snake.pop()
                reward = 0.0

            # Stage 2 step penalty
            if self.stage >= 2:
                reward -= 0.01

            # Terminate as soon as max_steps is reached
            done = self.step_count >= self.max_steps
            return self._get_observation(), reward, done, {}
        except Exception as e:
            raise RuntimeError(f"Error during step: {e}")

    def render(self) -> None:
        """Render current state to console."""
        try:
            os.system("cls" if os.name == "nt" else "clear")
            print("+" + "-" * self.width + "+")
            for r in range(self.height):
                row = ""
                for c in range(self.width):
                    pos = (r, c)
                    if pos == self.snake[0]:
                        row += "@"
                    elif pos in self.snake[1:]:
                        row += "O"
                    elif pos == self.food_pos:
                        row += "*"
                    else:
                        row += " "
                print("|" + row + "|")
            print("+" + "-" * self.width + "+")
        except Exception as e:
            raise RuntimeError(f"Error during render: {e}")

    def render_image(self, scale: int = 10) -> Image.Image:
        """Render current state to a PIL Image, scaled by `scale`."""
        try:
            obs = self._get_observation()
            h, w = obs.shape
            color_map = {
                0: (255, 255, 255),
                1: (0, 0, 255),
                2: (0, 255, 0),
                3: (255, 0, 0),
            }
            img = Image.new("RGB", (w, h))
            pixels = img.load()
            for y in range(h):
                for x in range(w):
                    pixels[x, y] = color_map[int(obs[y, x])]
            return img.resize((w * scale, h * scale), resample=Image.NEAREST)
        except Exception as e:
            raise RuntimeError(f"Error during render_image: {e}")

    def _apply_action(self, action: int) -> None:
        """Convert action index to direction."""
        directions = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}
        if action not in directions:
            raise ValueError(f"Invalid action: {action}")
        self.direction = directions[action]

    def _check_collision(self, pos: Tuple[int, int]) -> bool:
        """Check collision with walls or self."""
        r, c = pos
        if r < 0 or r >= self.height or c < 0 or c >= self.width:
            return True
        return pos in self.snake

    def _place_food(self) -> None:
        """Place food randomly in empty cell."""
        empty = [
            (r, c)
            for r in range(self.height)
            for c in range(self.width)
            if (r, c) not in self.snake
        ]
        self.food_pos = random.choice(empty)

    def _get_observation(self) -> np.ndarray:
        """Get current grid observation with head=1, body=2, food=3."""
        grid = np.zeros((self.height, self.width), dtype=int)
        # head
        head_r, head_c = self.snake[0]
        grid[head_r, head_c] = 1
        # body
        for r, c in self.snake[1:]:
            grid[r, c] = 2
        # food
        fr, fc = self.food_pos
        grid[fr, fc] = 3
        return grid
