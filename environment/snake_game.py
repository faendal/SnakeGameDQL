import random
from collections import deque
from typing import Deque, Tuple

import numpy as np


class SnakeGame:
    """
    Snake Game environment for Deep Q-Learning.

    Representa el juego sobre una rejilla de celdas, sin dependencia de pygame.
    Cada celda puede ser:
        0: vacío
        1: cabeza de la serpiente
        2: cuerpo de la serpiente
        3: comida

    Attributes:
        grid_size: Tupla (filas, columnas) del tamaño de la rejilla.
        max_steps_without_food: Límite de pasos sin comer antes de terminar.
        snake: Deque de pares (fila, col) con la posición de cada segmento.
        direction_idx: Índice en self._directions de la dirección actual.
        food_pos: Par (fila, col) de la comida.
        score: Puntuación acumulada (comidas capturadas).
        frame_iteration: Pasos desde el último alimento (para evitar bucles).
        done: Indica si el episodio ha terminado.
    """

    def __init__(
        self, grid_size: Tuple[int, int] = (20, 20), max_steps_without_food: int = 100
    ) -> None:
        """
        Inicializa el entorno.

        Args:
            grid_size: Tamaño de la rejilla en celdas (filas, columnas).
            max_steps_without_food: Máximo de pasos sin comer antes de terminar.
        """
        self.grid_size = grid_size
        self.max_steps_without_food = max_steps_without_food

        # Direcciones: Up, Right, Down, Left
        self._directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.reset()

    def reset(self) -> np.ndarray:
        """
        Reinicia el entorno al estado inicial.

        Returns:
            state: Array numpy de forma (1, filas, columnas) con el estado inicial.
        """
        rows, cols = self.grid_size
        # Empieza en el centro desplazada a la derecha
        head = (rows // 2, cols // 2)
        self.direction_idx = 1  # Right
        self.snake: Deque[Tuple[int, int]] = deque(
            [
                head,
                (head[0], head[1] - 1),
                (head[0], head[1] - 2),
            ]
        )
        self.head = head
        self.score = 0
        self.frame_iteration = 0
        self.done = False

        self._place_food()
        return self.get_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Ejecuta un paso de simulación dado un action one-hot [straight, right, left].

        Args:
            action: Array de numpy con forma (3,) y un único 1 en la acción.

        Returns:
            state: Nuevo estado tras la acción, shape (1, filas, columnas).
            reward: Recompensa obtenida en este paso.
            done: True si el episodio ha terminado.
        """
        self.frame_iteration += 1

        # Mueve la cabeza y actualiza la deque
        self._move(action)
        self.snake.appendleft(self.head)

        reward = 0.0
        rows, cols = self.grid_size

        # Colisión o demasiados pasos sin comer
        if (
            self.is_collision()
            or self.frame_iteration > self.max_steps_without_food * len(self.snake)
        ):
            self.done = True
            return self.get_state(), -10.0, True

        # Si come comida
        if self.head == self.food_pos:
            self.score += 1
            reward = 10.0
            self._place_food()
            self.frame_iteration = 0
        else:
            # Desplaza cola
            self.snake.pop()

        return self.get_state(), reward, False

    def is_collision(self, point: Tuple[int, int] = None) -> bool:
        """
        Comprueba si un punto colisiona contra pared o contra la serpiente.

        Args:
            point: Par (fila, col) a testear. Si es None, usa la cabeza actual.

        Returns:
            True si hay colisión.
        """
        if point is None:
            point = self.head

        rows, cols = self.grid_size
        r, c = point

        # Colisión con bordes
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return True

        # Colisión con su propio cuerpo
        if point in list(self.snake)[1:]:
            return True

        return False

    def _place_food(self) -> None:
        """
        Coloca comida en posición aleatoria que no esté ocupada por la serpiente.
        """
        rows, cols = self.grid_size
        while True:
            pos = (random.randint(0, rows - 1), random.randint(0, cols - 1))
            if pos not in self.snake:
                self.food_pos = pos
                break

    def _move(self, action: np.ndarray) -> None:
        """
        Actualiza la posición de la cabeza según la acción relativa.

        Args:
            action: One-hot array [straight, right, left].

        Raises:
            ValueError: Si action no es válido.
        """
        if action.shape != (3,):
            raise ValueError(f"Action must be shape (3,), got {action.shape}")

        idx = self.direction_idx
        # Straight
        if np.array_equal(action, np.array([1, 0, 0])):
            new_idx = idx
        # Right turn
        elif np.array_equal(action, np.array([0, 1, 0])):
            new_idx = (idx + 1) % 4
        # Left turn
        elif np.array_equal(action, np.array([0, 0, 1])):
            new_idx = (idx - 1) % 4
        else:
            raise ValueError(f"Invalid action vector: {action}")

        self.direction_idx = new_idx
        dr, dc = self._directions[new_idx]
        r, c = self.head
        self.head = (r + dr, c + dc)

    def get_state(self) -> np.ndarray:
        """
        Devuelve el estado actual como un array de una capa.

        Returns:
            Array numpy de tipo uint8 con shape (1, filas, columnas).
        """
        rows, cols = self.grid_size
        state = np.zeros((1, rows, cols), dtype=np.uint8)

        # Marca la serpiente: cabeza=1, cuerpo=2
        for i, (r, c) in enumerate(self.snake):
            state[0, r, c] = 1 if i == 0 else 2

        # Marca la comida=3
        fr, fc = self.food_pos
        state[0, fr, fc] = 3

        return state

    def render(self) -> None:
        """
        Imprime en consola una representación ASCII del grid:
        '.' vacío, 'H' cabeza, 'B' cuerpo, 'F' comida.
        """
        mapping = {0: ".", 1: "H", 2: "B", 3: "F"}
        grid = self.get_state()[0]
        for row in grid:
            print(" ".join(mapping[int(v)] for v in row))
        print(f"Score: {self.score}\n")
