import random
import numpy as np
from time import sleep
from Models.Snake import Snake
from Models.Agent import Agent


class Game:
    def __init__(
        self,
        agent: Agent,
        grid_size: tuple[int, int] = (20, 20),
        max_steps: int = 500,
        render_delay: float = 0.1,
    ):
        self.grid_size: tuple[int, int] = grid_size
        self.snake: Snake = Snake(grid_size)
        self.agent: Agent = agent
        self.max_steps: int = max_steps
        self.render_delay: int = render_delay
        self.reset()

    def reset(self):
        self.snake.reset()
        self.food = self.place_food()
        self.steps = 0
        self.score = 0
        return self.get_state()

    def place_food(self):
        empty_cells = set(
            (i, j) for i in range(self.grid_size[0]) for j in range(self.grid_size[1])
        )
        snake_cells = set(tuple(p) for p in self.snake.body)
        possible_positions = list(empty_cells - snake_cells)
        if not possible_positions:
            return None
        return np.array(random.choice(possible_positions))

    def get_state(self):
        grid = np.zeros(self.grid_size, dtype=np.float32)
        for segment in self.snake.body[1:]:
            grid[tuple(segment)] = 2
        grid[tuple(self.snake.body[0])] = 1
        if self.food is not None:
            grid[tuple(self.food)] = 3
        return grid

    def get_reward(self, done, ate_food, prev_head_pos):
        if done:
            return -10.0
        elif ate_food:
            return 5.0
        else:
            # Distancia Manhattan antes y después del paso
            old_dist = np.abs(prev_head_pos[0] - self.food[0]) + np.abs(
                prev_head_pos[1] - self.food[1]
            )
            new_head = self.snake.body[0]
            new_dist = np.abs(new_head[0] - self.food[0]) + np.abs(
                new_head[1] - self.food[1]
            )

            # Si se acercó a la comida: recompensa positiva pequeña, si se alejó: negativa
            delta = old_dist - new_dist
            return delta * 0.1  # Escalamos el cambio para que no domine el aprendizaje

    def is_collision(self, position):
        x, y = position
        if x < 0 or y < 0 or x >= self.grid_size[0] or y >= self.grid_size[1]:
            return True
        if any(np.array_equal(position, part) for part in self.snake.body[1:]):
            return True
        return False

    def step(self, action):
        prev_head = self.snake.body[0].copy()

        next_head = (
            self.snake.body[0]
            + self.snake.movements[self.snake.check_action(self.snake.direction, action)]
        )
        done = self.is_collision(next_head)

        if done:
            reward = self.get_reward(done, False, prev_head)
            return self.get_state(), reward, done

        self.snake.step(action)
        ate_food = np.array_equal(self.snake.body[0], self.food)
        if ate_food:
            self.snake.grow()
            self.score += 1
            self.food = self.place_food()

        reward = self.get_reward(done, ate_food, prev_head)
        next_state = self.get_state()
        return next_state, reward, done

    def train(
        self,
        num_episodes=1000,
        epsilon_start=1.0,
        epsilon_end=0.01,
    ):
        eps = epsilon_start
        scores = []
        for episode in range(1, num_episodes + 1):
            state = self.reset()
            total_reward = 0
            for _ in range(self.max_steps):
                action = self.agent.act(state, eps)
                next_state, reward, done = self.step(action)
                self.agent.step(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            eps = epsilon_end + (epsilon_start - epsilon_end) * np.exp(
                -episode / num_episodes
            )

            scores.append(total_reward)
            print(
                f"Episode {episode}/{num_episodes} - Score: {self.score} - Total Reward: {total_reward:.2f} - Epsilon: {eps:.4f}"
            )
        return scores

    def render(self, grid):
        for row in grid:
            print(
                "".join(
                    [
                        (
                            "⬛"
                            if cell == 0
                            else "🟥" if cell == 1 else "🟨" if cell == 2 else "🟩"
                        )
                        for cell in row
                    ]
                )
            )
        print()

    def play(self, render=True):
        state = self.reset()
        total_reward = 0
        for _ in range(self.max_steps):
            if render:
                self.render(self.get_state())
                sleep(self.render_delay)

            action = self.agent.act(state, eps=0.0)  # Always exploit
            next_state, reward, done = self.step(action)
            total_reward += reward
            state = next_state
            if done:
                break
        print(f"Game Over! Score: {self.score}, Total Reward: {total_reward}")
