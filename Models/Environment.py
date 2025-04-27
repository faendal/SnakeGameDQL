import random
import numpy as np
from collections import deque


class Environment:
    MOVE_VECTORS = [(-1, 0), (0, 1), (1, 0), (0, -1)]

    def __init__(self, grid_size=20, base_steps_without_food=50, loop_k=8):
        self.grid_size = grid_size
        self.base_steps_without_food = base_steps_without_food
        self.loop_history = deque(maxlen=loop_k)
        self.reset()

    def reset(self):
        self.grid = np.zeros((3, self.grid_size, self.grid_size), dtype=np.float32)
        center = self.grid_size // 2
        self.snake = [(center, center)]
        self.direction = random.randrange(4)
        self.spawn_food()
        self.visit_counts = np.zeros((self.grid_size, self.grid_size), dtype=np.int32)
        self.steps_since_last_food = 0
        self.loop_history.clear()
        return self._get_state()

    def spawn_food(self):
        empty = [
            (i, j)
            for i in range(self.grid_size)
            for j in range(self.grid_size)
            if (i, j) not in self.snake
        ]
        self.food_pos = random.choice(empty)

    def _get_state(self):
        # one-hot channels
        s = np.zeros_like(self.grid)
        for idx, (x, y) in enumerate(self.snake):
            if idx == 0:
                s[0, x, y] = 1.0  # head channel
            else:
                s[1, x, y] = 1.0  # body channel
        fx, fy = self.food_pos
        s[2, fx, fy] = 1.0  # food channel
        return s.copy()

    def get_distance(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def step(self, action):
        # direction guard
        if abs(action - self.direction) != 2:
            self.direction = action
        dx, dy = self.MOVE_VECTORS[self.direction]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        # collision or wall
        if (new_head in self.snake) or not (
            0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size
        ):
            self.done = True
            return self._get_state(), -30.0, True
        self.snake.insert(0, new_head)
        ate = new_head == self.food_pos
        if ate:
            reward = 30.0
            self.spawn_food()
            self.steps_since_last_food = 0
        else:
            self.snake.pop()
            self.steps_since_last_food += 1
            # distance shaping
            old_d = self.get_distance(head, self.food_pos)
            new_d = self.get_distance(new_head, self.food_pos)
            reward = (old_d - new_d) * (0.5 + 0.02 * len(self.snake))
            if new_d <= 3:
                reward += (3 - new_d) * 0.1
        # wall penalty
        md = min(
            new_head[0],
            new_head[1],
            self.grid_size - 1 - new_head[0],
            self.grid_size - 1 - new_head[1],
        )
        if md <= 1:
            reward -= 0.3
        elif md <= 2:
            reward -= 0.1
        # loop penalty
        if new_head in self.loop_history:
            reward -= 0.5
        self.loop_history.append(new_head)
        # visit penalty
        self.visit_counts[new_head] += 1
        if self.visit_counts[new_head] > 1:
            reward -= 0.1 * (self.visit_counts[new_head] - 1)
        # time penalty scales
        reward -= 0.01 + 0.001 * len(self.snake)
        # dynamic death
        max_steps = self.base_steps_without_food + 3 * len(self.snake)
        done = self.steps_since_last_food > max_steps
        if done:
            reward = -10.0
        reward = float(np.clip(reward, -30, 30))
        return self._get_state(), reward, done
