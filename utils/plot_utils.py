import os
from typing import List, Optional

import matplotlib.pyplot as plt


def plot_scores(scores: List[int], save_path: Optional[str] = None) -> None:
    """
    Grafica la evolución de la puntuación por episodio.

    Args:
        scores: Lista de puntuaciones obtenidas en cada episodio.
        save_path: Ruta opcional donde guardar la imagen (PNG).
        Si es None, no guarda archivo.
    """
    plt.figure()
    plt.plot(scores)
    plt.title("Score por episodio")
    plt.xlabel("Episodio")
    plt.ylabel("Score")
    plt.grid(True)
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
    plt.show()
