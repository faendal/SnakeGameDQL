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

        self.grid_size: tuple[int, int] = grid_size
        self.body: list[np.ndarray] = []
        self.directions: list[str] = [0, 1, 2, 3]
        self.movements: dict[str, np.ndarray] = {
            0: np.array([0, 1]),
            1: np.array([0, -1]),
            2: np.array([-1, 0]),
            3: np.array([1, 0]),
        }
        self.reset()

    def reset(self):
        """
        Reset the snake to its initial state.
        The snake is placed at a random position on the grid,
        and its initial direction is randomly chosen.
        """

        self.direction = np.random.choice(self.directions)
        head = np.random.randint(low=1, high=self.grid_size[0] - 1, size=2)
        tail = head - self.movements[self.direction_mapper[self.direction]]
        self.body = [head, tail]

    def grow(self):
        """
        Grow the snake by adding a new segment to its body.
        The new segment is placed as the tail
        """

        self.body.append(self.body[-1] - self.movements[self.direction])

    def check_action(self, direction: int, action: int) -> int:

        if direction == 0 or direction == 1:
            if action == 0 or action == 1:
                return direction
            else:
                return action
        elif direction == 2 or direction == 3:
            if action == 2 or action == 3:
                return direction
            else:
                return action

    def step(self, action: int):
        """
        Move the snake in the specified direction.
        
        :param action: Direction to move the snake [0(up), 1(down), 2(left), 3(right)].
        """

        self.direction = self.check_action(self.direction, action)
        head = self.body[0] + self.movements[self.direction]
        self.body.insert(0, head)
        self.body.pop()
