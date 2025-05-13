import torch
from typing import Tuple

# Device configuration
#: Torch device to perform computations on (CUDA if available, else CPU).
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment
#: Size of the SnakeGame grid (rows, columns).
GRID_SIZE: Tuple[int, int] = (20, 20)

# Replay buffer
#: Maximum number of transitions to store.
MEMORY_SIZE: int = 10_000

# Training hyperparameters
#: Mini-batch size for sampling from replay buffer.
BATCH_SIZE: int = 64
#: Discount factor for future rewards.
GAMMA: float = 0.99
#: Learning rate for the optimizer.
LR: float = 1e-3
#: Initial ε for ε-greedy policy.
EPS_START: float = 1.0
#: Final ε for ε-greedy policy.
EPS_END: float = 0.01
#: Decay rate for ε (higher means slower decay).
EPS_DECAY: int = 500
#: Number of episodes between target network updates.
TARGET_UPDATE: int = 10
#: Total number of training episodes.
MAX_EPISODES: int = 500
#: Maximum steps per episode to avoid infinite loops.
MAX_STEPS_PER_EPISODE: int = 1_000

# Checkpointing
#: Directory where model checkpoints will be saved.
CHECKPOINT_DIR: str = "checkpoints"
