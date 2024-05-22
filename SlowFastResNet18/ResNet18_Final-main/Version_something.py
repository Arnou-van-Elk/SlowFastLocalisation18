import torch
import torch.nn as nn
from torch import Tensor
from typing import Type, List


NUM_CLASSES = 114
ALPHA = 8
BETA_INV = 8 
FREQUENCY_STRIDES = [[1], [2], [2], [2]]

class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, 
                 temp_kernel_size: int = 3, downsample: nn.Module = None):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // self.expansion
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=(temp_kernel_size, temp_kernel_size), 
                               padding=(temp_kernel_size//2, temp_kernel_size//2), stride=(1, stride), bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)
        return out

class ResNetSlowFast(nn.Module):
    def __init__(self, img_channels: int, block: Type[Bottleneck], num_classes: int = NUM_CLASSES):
        super(ResNetSlowFast, self).__init__()

        self.slow_channel = int(64 * (8 / ALPHA))
        self.fast_channel = self.slow_channel // BETA_INV
        
        self.slow_path = nn.Sequential(
            self.make_layer(img_channels, self.slow_channel, [2, 2, 2, 2], block),
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.fast_path = nn.Sequential(
            self.make_layer(img_channels, self.fast_channel, [2, 2, 2, 2], block),
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Linear(self.slow_channel + self.fast_channel, num_classes)
        
    def make_layer(self, in_channels, out_channels, layers, block):
        layers_list = []
        for i, num_blocks in enumerate(layers):
            stride = FREQUENCY_STRIDES[i][0]
            if i > 0:  
                downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )
            else:
                downsample = None
            block_layer = block(in_channels, out_channels, stride, downsample=downsample)
            layers_list.append(block_layer)
            in_channels = out_channels  
        return nn.Sequential(*layers_list)
    
    def forward(self, x: Tensor) -> Tensor:
        slow = self.slow_path(x)
        fast = self.fast_path(x)
        
        combined = torch.cat([slow.view(slow.size(0), -1), fast.view(fast.size(0), -1)], dim=1)
        out = self.fc(combined)
        return out

if __name__ == "__main__":
    model = ResNetSlowFast(img_channels=2, block=Bottleneck)
    print(model)
    tensor = torch.randn(1, 2, 128, 200)  
    output = model(tensor)
    print("Output shape:", output.shape)
    print("Output:", output)