import random
import numpy as np


class Environment:
    def __init__(self, grid_size=20, base_steps_without_food=50):
        self.grid_size = grid_size
        self.base_steps_without_food = base_steps_without_food
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

    def get_wall_distance_penalty(self, head):
        hx, hy = head
        min_dist = min(hx, hy, self.grid_size - 1 - hx, self.grid_size - 1 - hy)
        if min_dist <= 1:
            return -0.3  # Big penalty if right next to wall
        elif min_dist <= 2:
            return -0.1
        else:
            return 0.0

    def is_almost_colliding(self, new_head):
        return new_head in self.snake[1:]  # Check if about to collide into itself

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

        old_dist = self.get_distance_to_food(self.snake[0])
        new_dist = self.get_distance_to_food(new_head)

        # Check collisions
        if (new_head in self.snake) or not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        ):
            self.done = True
            self.reward = -30
            return self.grid.copy(), self.reward, self.done

        self.snake.insert(0, new_head)

        # Start small penalty for just existing
        self.reward = -0.1

        if new_head == self.food_pos:
            self.reward += 30  # Big reward for eating food
            self.score += 1
            self.spawn_food()
            self.steps_since_last_food = 0
        else:
            self.snake.pop()
            self.steps_since_last_food += 1

            # Reward based on distance improvement
            base_scale = 0.5
            scale_growth = 0.02
            urgency_scale = 0.1

            distance_change = old_dist - new_dist
            adaptive_scale = base_scale + (len(self.snake) - 1) * scale_growth
            base_reward = distance_change * adaptive_scale

            urgency_bonus = 0
            if new_dist <= 3:
                urgency_bonus = (3 - new_dist) * urgency_scale

            self.reward += base_reward + urgency_bonus

            # Clip total reward to prevent instability
            self.reward = np.clip(self.reward, -10, 30)

        # Wall proximity penalty
        self.reward += self.get_wall_distance_penalty(new_head)

        # Near-self penalty
        if self.is_almost_colliding(new_head):
            self.reward -= 0.2

        # Reward for longer snake
        self.reward += (len(self.snake) - 1) * 0.05

        # Dynamic patience based on snake length
        allowed_steps = self.base_steps_without_food + len(self.snake) * 2
        if self.steps_since_last_food > allowed_steps:
            self.done = True
            self.reward = -10

        self.update_grid()
        return self.grid.copy(), self.reward, self.done
