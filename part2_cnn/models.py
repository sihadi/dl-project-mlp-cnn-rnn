import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class SimpleMLP(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LeNetLikeCNN(nn.Module):
    def __init__(self, num_classes: int = 10, base_channels: int = 6, use_1x1: bool = False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, base_channels, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=5, padding=2)
        self.conv1x1 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=1) if use_1x1 else nn.Identity()
        self.fc1 = nn.Linear(base_channels * 2 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv1x1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class FlexibleCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_filters: int = 16,
        padding: int = 1,
        stride: int = 1,
        pool_type: str = 'max',
        use_1x1: bool = False,
    ):
        super().__init__()
        pool_type = pool_type.lower()
        if pool_type not in {'max', 'avg'}:
            raise ValueError("pool_type must be 'max' or 'avg'")
        self.pool = nn.MaxPool2d(2, 2) if pool_type == 'max' else nn.AvgPool2d(2, 2)
        self.conv1 = nn.Conv2d(1, num_filters, kernel_size=3, stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=3, stride=1, padding=padding)
        self.conv1x1 = nn.Conv2d(num_filters * 2, num_filters * 2, kernel_size=1) if use_1x1 else nn.Identity()
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv1x1(x))
        return self.head(x)
