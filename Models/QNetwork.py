import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.act_fn = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        squeeze = x.view(batch_size, channels, -1).mean(dim=2)
        excitation = self.act_fn(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation))
        excitation = excitation.view(batch_size, channels, 1, 1)
        return x * excitation


class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()

        self.act_fn = nn.LeakyReLU(negative_slope=0.01)

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # 1x1 conv for matching shortcut dimensions
        self.residual_conv = nn.Conv2d(128, 256, kernel_size=1, stride=1)

        self.se = SEBlock(256)

        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))

        self.fc1 = nn.Linear(256 * 5 * 5, 512)
        self.fc2 = nn.Linear(512, action_size)

        self.dropout = nn.Dropout(p=0.2)

    def forward(self, state):
        x = self.act_fn(self.bn1(self.conv1(state)))
        x = self.act_fn(self.bn2(self.conv2(x)))

        residual = x  # Save for shortcut

        x = self.act_fn(self.bn3(self.conv3(x)))

        residual = self.residual_conv(residual)  # Adjust residual to 256 channels

        x += residual  # Add adjusted shortcut

        x = self.se(x)  # Apply Squeeze-Excite

        x = self.adaptive_pool(x)  # Pool to (5x5)

        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(self.act_fn(self.fc1(x)))
        return self.fc2(x)
