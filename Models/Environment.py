import random
import numpy as np


class Environment:
    def __init__(self, grid_size=20, max_steps_without_food=50):
        self.grid_size = grid_size
        self.max_steps_without_food = max_steps_without_food
        self.reset()

    def reset(self):
        self.grid = np.zeros((1, self.grid_size, self.grid_size), dtype=np.float32)
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = random.choice([0, 1, 2, 3])  # Up, Right, Down, Left
        self.spawn_food()
        self.update_grid()
        self.reward = 0
        self.score = 0
        self.done = False
        self.steps_since_last_food = 0  # Track steps without eating
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
            self.reward = -30  # Big penalty for dying
            return self.grid.copy(), self.reward, self.done

        self.snake.insert(0, new_head)

        if new_head == self.food_pos:
            self.reward = 20  # Big reward for eating food
            self.score += 1
            self.spawn_food()
            self.steps_since_last_food = 0  # Reset steps counter
        else:
            self.snake.pop()
            self.steps_since_last_food += 1
            # Bonus/punishment based on distance to food
            if new_dist < old_dist:
                self.reward += 0.2  # Bonus for approaching
            else:
                self.reward -= 0.2  # Penalty for moving away

        # Extra reward for longer snake
        self.reward += (len(self.snake) - 1) * 0.05

        # Force death if too long without eating
        if self.steps_since_last_food > self.max_steps_without_food:
            self.done = True
            self.reward = -10  # Penalty for being too slow

        self.update_grid()
        return self.grid.copy(), self.reward, self.done
