import torch
import torch.nn as nn

# Define the Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, down_sample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.down_sample is not None:
            identity = self.down_sample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, input_dim, hidden_dim, output_dim):
        super(ResNet, self).__init__()
        self.in_channels = hidden_dim

        # since the convolution happens on sequence dimension, we want 3 step as a patch, thus kernel size is 3
        # since all steps are equally important, we want the stride to be 1
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)

        # Stacking residual blocks
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=1)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1)

        self.avgpool = nn.AdaptiveAvgPool1d(3)

        # Final fully connected layer
        # input features should be the output channel of the last layer
        self.fc = nn.Linear(256, output_dim)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        down_sample = None
        if stride != 1 or self.in_channels != out_channels:
            down_sample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, down_sample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        # 3 layers of resnet does not change sequence, pooling from 5 -> 3
        x = self.avgpool(x)
        x = x.transpose(1, 2)
        x = self.fc(x)

        return x
