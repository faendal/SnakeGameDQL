"""
Q-Network definition for the DQN agent using a simple CNN architecture.
"""

import torch
import torch.nn as nn
from typing import Tuple


class QNetwork(nn.Module):
    """CNN-based Q-network for the Snake DQN agent.

    This network takes a single-channel grid observation and outputs a Q-value for each action.
    """

    def __init__(self, input_shape: Tuple[int, int], num_actions: int) -> None:
        """Initialize the QNetwork architecture.

        Args:
            input_shape: Tuple (height, width) of the input grid.
            num_actions: Number of discrete actions in the environment.
        """
        super().__init__()
        height, width = input_shape

        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
            ),
            nn.ReLU(),
        )

        # Compute flattened feature size
        conv_output_size = 64 * height * width

        # Fully connected (MLP) layers
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=conv_output_size, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: compute Q-values for each action.

        Args:
            x: Tensor of shape (batch_size, height, width) or (batch_size, 1, height, width).

        Returns:
            Tensor of shape (batch_size, num_actions).
        """
        try:
            # Ensure channel dimension
            if x.dim() == 3:
                x = x.unsqueeze(1)
            elif x.dim() != 4:
                raise ValueError(
                    f"Expected input tensor with 3 or 4 dims, got {x.dim()} dims."
                )

            # Normalize grid values (max value is 3 for food)
            x = x.float() / 3.0

            # Convolutional feature extraction
            features = self.conv_layers(x)

            # Compute Q-values
            q_values = self.fc_layers(features)
            return q_values
        except Exception as e:
            raise RuntimeError(f"Error in QNetwork forward pass: {e}")
