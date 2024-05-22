import torch
import torch.nn as nn
from torchvision.models import resnet18

class FuseFastToSlow(nn.Module):

    """

    Fuses the information from the Fast pathway to the Slow pathway.

    """
    def __init__(
        self,
        dim_in,
        fusion_conv_channel_ratio,
        fusion_kernel,
        alpha,
        eps=1e-5,
        bn_mmt=0.1,
        inplace_relu=True,
        norm_module=nn.BatchNorm2d,
    ):

        super(FuseFastToSlow, self).__init__()

        self.conv_f2s = nn.Conv2d(
            dim_in,
            dim_in * fusion_conv_channel_ratio,
            kernel_size=[fusion_kernel, 1],
            stride=[alpha, 1],
            padding=[fusion_kernel // 2, 0],
            bias=False,
        )

        self.bn = norm_module(
            num_features=dim_in * fusion_conv_channel_ratio,
            eps=eps,
            momentum=bn_mmt,
        )
        self.relu = nn.ReLU(inplace=inplace_relu)



    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]



class SlowFastResNet18(nn.Module):

    """

    SlowFast model with ResNet18 backbone.

    """
    def __init__(self, num_classes=1000):
        super(SlowFastResNet18, self).__init__()
        self.norm_module = nn.BatchNorm2d
        self.alpha = 4
        self.beta_inv = 8
        self._construct_network(num_classes)



    def _construct_network(self, num_classes):
        resnet = resnet18()

        # First convolutional layer should accept 2 input channels
        self.layer1_s = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False),
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1
        )

        self.layer1_f = nn.Sequential(
            nn.Conv2d(2, 64 // self.alpha, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64 // self.alpha),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64 // self.alpha, 64 // self.alpha, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64 // self.alpha),
            nn.ReLU()
        )

        self.layer2_s = resnet.layer2
        self.layer2_f = nn.Sequential(
            nn.Conv2d(64 // self.alpha, 128 // self.alpha, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128 // self.alpha),
            nn.ReLU()
        )

        self.layer3_s = resnet.layer3
        self.layer3_f = nn.Sequential(
            nn.Conv2d(128 // self.alpha, 256 // self.alpha, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256 // self.alpha),
            nn.ReLU()
        )
        self.layer4_s = resnet.layer4
        self.layer4_f = nn.Sequential(
            nn.Conv2d(256 // self.alpha, 512 // self.alpha, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512 // self.alpha),
            nn.ReLU()
        )
        # Fusion layers
        self.s1_fuse = FuseFastToSlow(64 // self.alpha, fusion_conv_channel_ratio=2, fusion_kernel=5, alpha=self.alpha)
        self.s2_fuse = FuseFastToSlow(128 // self.alpha, fusion_conv_channel_ratio=2, fusion_kernel=5, alpha=self.alpha)
        self.s3_fuse = FuseFastToSlow(256 // self.alpha, fusion_conv_channel_ratio=2, fusion_kernel=5, alpha=self.alpha)
        self.s4_fuse = FuseFastToSlow(512 // self.alpha, fusion_conv_channel_ratio=2, fusion_kernel=5, alpha=self.alpha)



        # Final classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024, num_classes),  
        )

    def forward(self, x):
        x_s = self.layer1_s(x[0])
        x_f = self.layer1_f(x[1])
        print(f'After layer1_s: {x_s.shape}')
        print(f'After layer1_f: {x_f.shape}')
        x = [x_s, x_f]
        x = self.s1_fuse(x)
        x_s = self.layer2_s(x[0])
        x_f = self.layer2_f(x[1])
        print(f'After layer2_s: {x_s.shape}')
        print(f'After layer2_f: {x_f.shape}')
        x = [x_s, x_f]
        x = self.s2_fuse(x)
        x_s = self.layer3_s(x[0])
        x_f = self.layer3_f(x[1])
        print(f'After layer3_s: {x_s.shape}')
        print(f'After layer3_f: {x_f.shape}')
        x = [x_s, x_f]
        x = self.s3_fuse(x)
        x_s = self.layer4_s(x[0])
        x_f = self.layer4_f(x[1])
        print(f'After layer4_s: {x_s.shape}')
        print(f'After layer4_f: {x_f.shape}')
        x = [x_s, x_f]
        x = self.s4_fuse(x)
        x_s = self.head(x[0])
        return x_s



if __name__ == '__main__':
    slow_input = torch.rand(1, 2, 128, 200)   #[batch_size, channels, frequency, framerate]
    fast_input = torch.rand(1, 2, 128, 200)

    model = SlowFastResNet18(num_classes=114)
    print(model)



    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    output = model([slow_input, fast_input])
    print("Output shape:", output.shape)