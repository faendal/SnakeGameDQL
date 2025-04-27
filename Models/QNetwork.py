import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1
        )
        self.gn1 = nn.GroupNorm(num_groups=out_channels // 8, num_channels=out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        self.gn2 = nn.GroupNorm(num_groups=out_channels // 8, num_channels=out_channels)
        self.act = nn.LeakyReLU(0.01)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.GroupNorm(num_groups=out_channels // 8, num_channels=out_channels),
            )

    def forward(self, x):
        residual = x
        out = self.act(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample:
            residual = self.downsample(residual)
        out += residual
        return self.act(out)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.act_fn = nn.LeakyReLU(0.01)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        squeeze = x.view(b, c, -1).mean(dim=2)
        excitation = self.act_fn(self.fc1(squeeze))
        excitation = self.sigmoid(self.fc2(excitation))
        excitation = excitation.view(b, c, 1, 1)
        return x * excitation


class QNetwork(nn.Module):
    def __init__(self, action_size):
        super(QNetwork, self).__init__()
        # Input: 3-channel one-hot grid
        self.res1 = ResBlock(3, 64, stride=1)
        self.res2 = ResBlock(64, 128, stride=2)  # downsample 20x20 -> 10x10
        self.res3 = ResBlock(128, 256, stride=2)  # downsample 10x10 -> 5x5
        self.se = SEBlock(256)

        # Shared fully-connected
        self.fc_shared = nn.Linear(256 * 5 * 5, 512)
        self.dropout = nn.Dropout(p=0.1)
        self.act = nn.LeakyReLU(0.01)

        # Dueling heads
        self.fc_value = nn.Linear(512, 1)
        self.fc_adv = nn.Linear(512, action_size)

    def forward(self, state):
        # state: (B,3,20,20)
        x = self.res1(state)
        x = self.res2(x)
        x = self.res3(x)
        x = self.se(x)
        x = x.view(x.size(0), -1)
        x = self.act(self.fc_shared(x))
        x = self.dropout(x)
        val = self.fc_value(x)  # (B,1)
        adv = self.fc_adv(x)  # (B,action_size)
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q
