import random
import numpy as np


class Environment:
    def __init__(self, grid_size=20):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.grid = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = random.choice([0, 1, 2, 3])  # Up, Right, Down, Left
        self.spawn_food()
        self.update_grid()
        self.score = 0
        self.done = False
        return self.grid.copy()

    def spawn_food(self):
        empty = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.snake
        ]
        self.food_pos = random.choice(empty)

    def update_grid(self):
        self.grid.fill(0)
        for idx, (x, y) in enumerate(self.snake):
            self.grid[0, x, y] = 1 if idx == 0 else 2
        fx, fy = self.food_pos
        self.grid[0, fx, fy] = 3

    def get_distance_to_food(self, head):
        fx, fy = self.food_pos
        hx, hy = head
        return abs(fx - hx) + abs(fy - hy)

    def step(self, action):
        if self.done:
            return self.grid.copy(), 0, True

        # Disallow 180° reversal
        if abs(action - self.direction) != 2:
            self.direction = action

        # Movement vector
        dx, dy = [(-1, 0), (0, 1), (1, 0), (0, -1)][self.direction]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        # Pre-calculate distance for reward
        old_dist = self.get_distance_to_food(self.snake[0])
        new_dist = self.get_distance_to_food(new_head)

        # Check collisions
        if (new_head in self.snake) or not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        ):
            self.done = True
            reward = -10
            reward -= 0.1
            return self.grid.copy(), reward, self.done

        self.snake.insert(0, new_head)

        if new_head == self.food_pos:
            reward = 10
            self.score += 1
            self.spawn_food()
        else:
            self.snake.pop()
            reward = 2 if new_dist < old_dist else -2
        
        reward -= 0.1

        self.update_grid()
        return self.grid.copy(), reward, self.done
