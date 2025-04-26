import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()

        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, stride=1, padding=1
        )  # (B, 32, 20, 20)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(
            32, 64, kernel_size=3, stride=1, padding=1
        )  # (B, 64, 20, 20)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(
            64, 128, kernel_size=3, stride=1, padding=1
        )  # (B, 128, 20, 20)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(
            128, 256, kernel_size=3, stride=1, padding=1
        )  # (B, 256, 20, 20)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)  # (B, 256, 10, 10)

        self.fc1 = nn.Linear(256 * 10 * 10, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, action_size)

        self.dropout = nn.Dropout(p=0.3)

    def forward(self, state):
        x = F.relu(self.bn1(self.conv1(state)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)  # Flatten

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.silu(self.fc2(x)))
        return self.fc3(x)
