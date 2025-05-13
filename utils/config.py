"""
Configuration module for hyperparameters and curriculum stage settings.
"""

import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class StageConfig:
    """Configuration for a single curriculum training stage."""

    stage: int
    env_kwargs: Dict[str, Any]
    agent_kwargs: Dict[str, Any]
    num_episodes: int
    checkpoint_path: Optional[str]
    output_path: str


@dataclass
class DQNConfig:
    """Global configuration for DQN agent and environment."""

    # Environment settings
    state_shape: Tuple[int, int] = (20, 20)
    num_actions: int = 4
    width: int = 20
    height: int = 20
    max_steps_stage1: int = 500
    max_steps_stage2: int = 1000

    # Agent hyperparameters
    replay_buffer_size: int = 10000
    batch_size: int = 64
    gamma: float = 0.99
    lr: float = 1e-3
    target_update_freq: int = 1000
    eps_start: float = 1.0
    eps_end: float = 0.1
    eps_decay: int = 10000

    # Training schedule
    episodes_stage1: int = 5000
    episodes_stage2: int = 10000

    # Checkpoint directory
    checkpoint_dir: str = "checkpoints"

    def __post_init__(self) -> None:
        """Ensure checkpoint directory exists."""
        try:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
        except Exception as e:
            raise RuntimeError(
                f"Could not create checkpoint dir {self.checkpoint_dir}: {e}"
            )
