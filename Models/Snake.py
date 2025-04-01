import numpy as np


class Snake:
    """
    Class representing the snake in the game.
    The snake is represented as a list of coordinates on a grid.
    The snake can grow and change direction.
    """

    def __init__(self, grid_size=(20, 20)):
        """
        Initialize the Snake class.
        :param grid_size: Size of the grid (height, width).
        """

        self.grid_size = grid_size
        self.body: list[np.ndarray] = []
        self.directions: list[str] = ["up", "down", "left", "right"]
        self.movements: dict[str, np.ndarray] = {
            "up": np.array([0, 1]),
            "down": np.array([0, -1]),
            "left": np.array([-1, 0]),
            "right": np.array([1, 0]),
        }
        self.reset()

    def reset(self):
        """
        Reset the snake to its initial state.
        The snake is placed at a random position on the grid,
        and its initial direction is randomly chosen.
        """

        self.direction = np.random.choice(self.directions, p=[0.25, 0.25, 0.25, 0.25])
        head = np.random.randint(low=1, high=self.grid_size[0] - 1, size=2)
        tail = head - self.movements[self.direction]
        self.body = [head, tail]

    def grow(self):
        """
        Grow the snake by adding a new segment to its body.
        The new segment is placed as the tail
        """

        self.body.append(self.body[-1] - self.movements[self.direction])

    def move(self, action: str):
        """
        Move the snake in the specified direction.
        :param action: Direction to move the snake ["up", "down", "left", "right"].
        """

        if action not in self.directions:
            raise ValueError(
                f"Invalid action: {action}. Valid actions are: {self.directions}."
            )

        movement = self.movements[action]
        head = self.body[0] + movement

        self.body.insert(0, head)
        self.body.pop()
