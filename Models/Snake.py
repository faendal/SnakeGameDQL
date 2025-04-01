import numpy as np

class Snake:
    def __init__(self, grid_size=(20, 20)):
        self.directions: list[str] = ["up", "down", "left", "right", "none"]
        self.movements: dict[str, np.ndarray] = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
            "none": np.array([0, 0]),
        }
        self.grid_size = grid_size
        self.direction = np.random.choice(self.directions, p=[0.25, 0.25, 0.25, 0.25, 0])
        self.reset()

    def reset(self):
        head = np.random.randint(low=1, high=self.grid_size[0] - 1, size=2)
        tail = head - self.movements[self.direction]
        self.body = [head, tail]

    def move(self, action):
        new_direction = self.movements.get(action, self.direction)
        if not np.array_equal(new_direction, -self.direction):
            self.direction = new_direction
        new_head = self.body[0] + self.direction
        return new_head

    def grow(self):
        self.body.append(self.body[-1])

    def update_body(self, new_head):
        self.body.insert(0, new_head)
        self.body.pop()

    def check_collision(self, new_head):
        if (
            new_head[0] < 0
            or new_head[0] >= self.grid_size[0]
            or new_head[1] < 0
            or new_head[1] >= self.grid_size[1]
            or any(np.array_equal(new_head, part) for part in self.body)
        ):
            return True
        return False
