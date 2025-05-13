import torch
import torch.nn as nn
from typing import Tuple


class QNetwork(nn.Module):
    """
    Convolutional Q-Network for approximating state-action values in SnakeGame.

    This network takes as input a single-channel 20×20 grid (or any grid size)
    and outputs a vector of action_size Q-values.

    Architecture:
        - Conv2d: in_channels → 32, kernel 3×3, padding 1
        - ReLU
        - Conv2d: 32 → 64, kernel 3×3, padding 1
        - ReLU
        - Conv2d: 64 → 64, kernel 3×3, padding 1
        - ReLU
        - Flatten
        - Linear: (64 * H * W) → 256
        - ReLU
        - Linear: 256 → action_size

    Attributes:
        conv1: First convolutional layer.
        conv2: Second convolutional layer.
        conv3: Third convolutional layer.
        fc1: First fully connected layer.
        fc2: Output layer producing Q-values.
    """

    def __init__(self, state_shape: Tuple[int, int, int], action_size: int) -> None:
        """
        Initializes the Q-network.

        Args:
            state_shape: Shape of the input state (channels, height, width).
            action_size: Number of discrete actions (output dimension).
        """
        super().__init__()
        channels, height, width = state_shape

        # Convolutional feature extractor
        self.conv1 = nn.Conv2d(
            in_channels=channels, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # Compute size of flattened features
        conv_output_size = 64 * height * width

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 256)
        self.fc2 = nn.Linear(256, action_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x: Input state tensor of shape (batch_size, channels, height, width).

        Returns:
            Tensor of shape (batch_size, action_size) with Q-values for each action.
        """
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # flatten
        x = torch.relu(self.fc1(x))
        q_values = self.fc2(x)
        return q_values
