import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ChannelAdjust(nn.Module):
    """ there is some problem in the amount of channels after a fusion """
    def __init__(self, input_channels, output_channels):
        super(ChannelAdjust, self).__init__()
        self.conv1x1 = nn.Conv2d(input_channels, output_channels, kernel_size=1)

    def forward(self, x):
        return self.conv1x1(x)

class FuseFastToSlow(nn.Module):
    def __init__(self, dim_in_slow, dim_in_fast, fusion_conv_channel_ratio, fusion_kernel, alpha, eps=1e-5, bn_mmt=0.1, inplace_relu=True, norm_module=nn.BatchNorm2d):
        super(FuseFastToSlow, self).__init__()
        out_channels_fusion = int(dim_in_slow * fusion_conv_channel_ratio)
        self.conv_f2s = nn.Conv2d(
            in_channels=dim_in_fast,
            out_channels=out_channels_fusion,
            kernel_size=[fusion_kernel, 1],
            stride=[alpha, 1],
            padding=[fusion_kernel // 2, 0],
            bias=False
        )
        self.bn = norm_module(num_features=out_channels_fusion, eps=eps, momentum=bn_mmt)
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x_s, x_f):
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        fuse = F.interpolate(fuse, size=(x_s.size(2), x_s.size(3)), mode='nearest')  # ensure matching dimensions
        x_s_fused = torch.cat([x_s, fuse], dim=1)
        return x_s_fused, x_f  

class SlowFastStreams_ResNet34(nn.Module):
    def __init__(self, block, layers, num_classes=114, zero_init_residual=False, alpha=4, beta=8):
        super().__init__()
        self.in_channels_slow = 64
        self.in_channels_fast = self.in_channels_slow // beta
        
        self.slow_conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=4, padding=3, bias=False)
        self.fast_conv1 = nn.Conv2d(2, 8, kernel_size=7, stride=1, padding=3, bias=False)
        self.slow_bn1 = nn.BatchNorm2d(64)
        self.fast_bn1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Layer 1
        self.fusion_maxpool = FuseFastToSlow(64, 8, 1/beta, 7, alpha)
        self.adjust_maxpool = ChannelAdjust(136, 64)
        
        # First 
        self.layer2 = self._make_layer(block, 64, layers[0], 1)
        self.fusion2 = FuseFastToSlow(64, 8, 1/beta, 7, alpha)
        self.adjust2 = ChannelAdjust(64 + int(64 / beta), 64)  

        # Second 
        self.layer3 = self._make_layer(block, 128, layers[1], 1)
        self.fusion3 = FuseFastToSlow(128, 128//beta, 1/beta, 7, alpha)
        self.adjust3 = ChannelAdjust(128 + int(128 // beta), 128)

        # Third 
        self.layer4 = self._make_layer(block, 256, layers[2], 1)
        self.fusion4 = FuseFastToSlow(256, 256//beta, 1/beta, 7, alpha)
        self.adjust4 = ChannelAdjust(256 + int(256 // beta), 256)

        # Look into head_helper and its functions
        self.layer5 = self._make_layer(block, 512, layers[3], 1)
        self.fusion5 = FuseFastToSlow(512, 512//beta, 1/beta, 7, alpha)
        self.adjust5 = ChannelAdjust(512 + int(512 // beta), 512)


        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion * 2, num_classes) 

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels_slow != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels_slow, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels_slow, out_channels, stride, downsample))
        self.in_channels_slow = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels_slow, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x_slow, x_fast):
        # Initial Convolution and Activation on Both Streams
        x_slow = self.slow_conv1(x_slow)
        x_slow = self.slow_bn1(x_slow)
        x_slow = self.relu(x_slow)

        x_fast = self.fast_conv1(x_fast)
        x_fast = self.fast_bn1(x_fast)
        x_fast = self.relu(x_fast)

        # Initial pooling
        x_slow = self.maxpool(x_slow)
        x_fast = self.maxpool(x_fast)

        # Layering
        x_slow, x_fast = self.fusion2(x_slow, x_fast)  # Fusion affects both, mainly alters x_slow
        x_slow = self.adjust2(x_slow)  # Adjust x_slow channels to match expected dimensions for next layer
        x_slow = self.layer2(x_slow)  # Here we only forward x_slow, but typically, you'd also want to manage x_fast similarly if independent paths are necessary

        x_slow, x_fast = self.fusion3(x_slow, x_fast)
        x_slow = self.adjust3(x_slow)  # Assume adjust2 adjusts x_slow to proper channels after fusion
        x_slow = self.layer3(x_slow)

        x_slow, x_fast = self.fusion4(x_slow, x_fast)
        x_slow = self.adjust4(x_slow)  # Assume adjust2 adjusts x_slow to proper channels after fusion
        x_slow = self.layer4(x_slow)
    
        x_slow, x_fast = self.fusion5(x_slow, x_fast)
        x_slow = self.adjust5(x_slow)  # Assume adjust2 adjusts x_slow to proper channels after fusion
        x_slow = self.layer5(x_slow)

        # Global Average Pooling
        x_slow = self.avgpool(x_slow)
        x_fast = self.avgpool(x_fast)
    
        x_final = torch.cat([x_slow, x_fast], dim=1)  
        x_final = self.fc(x_final)  

        return x_final

# Testing Initialization
tensor = torch.rand([1,2,200,128])
tensor2 = torch.rand([1,2,200,128])
model = SlowFastStreams_ResNet34(BasicBlock, [3, 4, 6, 3], num_classes=114, zero_init_residual=True)
print(model)
output = model(tensor, tensor2)
total_params = sum(p.numel() for p in model.parameters())
print(f"{total_params:,} total parameters.")
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print(f"{total_trainable_params:,} training parameters.")