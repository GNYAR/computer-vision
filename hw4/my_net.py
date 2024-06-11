import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(
            4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False
        )

    def forward(self, x):
        new_features = self.conv1(F.relu(self.bn1(x)))
        new_features = self.conv2(F.relu(self.bn2(new_features)))
        return torch.cat([x, new_features], 1)


class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super().__init__()
        layer = lambda x: DenseLayer(in_channels + x * growth_rate, growth_rate)
        self.layers = nn.ModuleList([layer(i) for i in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(F.relu(self.bn(x)))
        x = self.pool(x)
        return x


class MyNet(nn.Module):
    def __init__(
        self, growth_rate=12, num_blocks=3, num_layers_per_block=4, num_classes=10
    ):
        super().__init__()
        num_channels = 2 * growth_rate

        self.conv1 = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)

        xs = []
        for i in range(num_blocks):
            xs.append(DenseBlock(num_layers_per_block, num_channels, growth_rate))
            num_channels += num_layers_per_block * growth_rate
            if i != num_blocks - 1:
                xs.append(TransitionLayer(num_channels, num_channels // 2))
                num_channels = num_channels // 2
        self.blocks = nn.ModuleList(xs)

        self.bn = nn.BatchNorm2d(num_channels)
        self.fc = nn.Linear(num_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        for block in self.blocks:
            x = block(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
