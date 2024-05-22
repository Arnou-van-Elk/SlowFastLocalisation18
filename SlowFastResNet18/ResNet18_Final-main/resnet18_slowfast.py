import torch.nn as nn
import torch

from torch import Tensor
from typing import Type # this is for type checking which is relevant when
                        # building complex architectures
    

_MODEL_STAGE_DEPTH = {18: (2, 2, 2,2)}
BETA_INV = 8
FUSION_CONV_CHANNEL_RATIO = 2
ALPHA = 8
FUSION_KERNEL_SZ = 5
INPUT_CHANNEL_NUM = [1,1]
FREQUENCY_STRIDES = [[1],[2],[2],[2]]
FREQUENCY_STRIDES = [[1],[1],[1],[1]]
NUM_BLOCK_TEMP_KERNEL = [[2],[2],[2],[2]]
TRANS_FUNC = "bottleneck_transform"
NUM_FRAMES = 200
NUM_FREQUENCIES = 128
HEAD_ACT ="softmax"
DROPOUT_RATE = 0.5

# Check if seperate kern basises of slow and fast are needed.
_TEMPORAL_KERNEL_BASIS = {
    "slowfast": [
        [[1], [5]],  # conv1 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res2 temporal kernel for slow and fast pathway.
        [[1], [3]],  # res3 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res4 temporal kernel for slow and fast pathway.
        [[3], [3]],  # res5 temporal kernel for slow and fast pathway.
    ]
}

# Check if seperate pools for slow and fast are needed
_POOL1 = {
    "slowfast": [[1, 1], [1, 1]]
}

# define a Python class for the Basic blocks
class BasicBlock(nn.Module):
    def __init__(
        self, 
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        # Multiplicative factor for the subsequent conv2d layer's output channels.
        # It is 1 for ResNet18 and ResNet34.
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=3, 
            stride=stride, # Compare stride sizes to those of OG slowfast model
            padding=1,
            bias=False
            # Add frequency dilation -> work throuhg ResNetHelper.ResStage
            # Transformation function 
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, 
            out_channels*self.expansion, 
            kernel_size=3, 
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity # add the original tensor to the output, this is the skip connection
        out = self.relu(out)
        return  out

# code for the ResNet module

class ResNet(nn.Module):
    def __init__(
        self, 
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int  = 114#1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            # The following `layers` list defines the number of `BasicBlock` 
            # to use to build the network and how many basic blocks to stack
            # together.
            layers = [2, 2, 2, 2]
            self.expansion = 1
        
        self.in_channels = 64
        # All ResNets (18 to 152) contain a Conv2d => BN => ReLU for the first
        # three layers. Here, kernel size is 7.
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7, 
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)
    def _make_layer(
        self, 
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            """
            This should pass from `layer2` to `layer4` or 
            when building ResNets50 and above. Section 3.3 of the paper
            Deep Residual Learning for Image Recognition
            (https://arxiv.org/pdf/1512.03385v1.pdf).
            """
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels, 
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False 
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        # The spatial dimension of the final layer's feature 
        # map should be (7, 7) for all ResNets.
        #print('Dimensions of the last convolutional feature map: ', x.shape)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class FuseFastToSlow(nn.Module):
    """
    Fuses the information from the Fast pathway to the Slow pathway. Given the
    tensors from Slow pathway and Fast pathway, fuse information from Fast to
    Slow, then return the fused tensors from Slow and Fast pathway in order.
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
        """
        Args:
            dim_in (int): the channel dimension of the input.
            fusion_conv_channel_ratio (int): channel ratio for the convolution
                used to fuse from Fast pathway to Slow pathway.
            fusion_kernel (int): kernel size of the convolution used to fuse
                from Fast pathway to Slow pathway.
            alpha (int): the frame rate ratio between the Fast and Slow pathway.
            eps (float): epsilon for batch norm.
            bn_mmt (float): momentum for batch norm. Noted that BN momentum in
                PyTorch = 1 - BN momentum in Caffe2.
            inplace_relu (bool): if True, calculate the relu on the original
                input without allocating new memory.
            norm_module (nn.Module): nn.Module for the normalization layer. The
                 default is nn.BatchNorm2d.
        """
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
        self.relu = nn.ReLU(inplace_relu)

    def forward(self, x):
        x_s = x[0]
        x_f = x[1]
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        x_s_fuse = torch.cat([x_s, fuse], 1)
        return [x_s_fuse, x_f]


class SlowFast(nn.Module):
    """
    SlowFast model builder for SlowFast network.

    Evangelos Kazakos, Arsha Nagrani, Andrew Zisserman, Dima Damen.
    "Auditory Slow-Fast Networks for Audio Recognition"

    Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He.
    "SlowFast networks for video recognition."
    https://arxiv.org/pdf/1812.03982.pdf
    """

    def __init__(self, cfg):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = get_norm(cfg)
        self.num_pathways = 2
        self._construct_network(cfg)
        # init_helper.init_weights(
        #     self, cfg.MODEL.FC_INIT_STD, cfg.RESNET.ZERO_INIT_FINAL_BN
        # )

    def _construct_network(self, cfg):
        """
        Builds a SlowFast model. The first pathway is the Slow pathway and the
            second pathway is the Fast pathway.

        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        # assert cfg.MODEL.ARCH in _POOL1.keys()
        pool_size = _POOL1["slowfast"]
        # assert len({len(pool_size), self.num_pathways}) == 1
        # assert cfg.RESNET.DEPTH in _MODEL_STAGE_DEPTH.keys()

        (d2, d3, d4, d5) = _MODEL_STAGE_DEPTH[18]

        num_groups = 1
        width_per_group = 64
        dim_inner = num_groups * width_per_group
        out_dim_ratio = (
            BETA_INV // FUSION_CONV_CHANNEL_RATIO
        )

        temp_kernel = _TEMPORAL_KERNEL_BASIS["slowfast"]

        self.s1 = slowfast_utils.AudioModelStem(
            dim_in=INPUT_CHANNEL_NUM,
            dim_out=[width_per_group, width_per_group // BETA_INV],
            kernel=[temp_kernel[0][0] + [7], temp_kernel[0][1] + [7]],
            stride=[[2, 2]] * 2,
            padding=[
                [temp_kernel[0][0][0] // 2, 3],
                [temp_kernel[0][1][0] // 2, 3],
            ],
            norm_module=self.norm_module,
        )
        self.s1_fuse = FuseFastToSlow(
            width_per_group // BETA_INV,
            FUSION_CONV_CHANNEL_RATIO,
            FUSION_KERNEL_SZ,
            ALPHA,
            norm_module=self.norm_module,
        )
        self.s2 = resnet_helper.ResStage(
            dim_in=[
                width_per_group + width_per_group // out_dim_ratio,
                width_per_group // BETA_INV,
            ],
            dim_out=[
                width_per_group * 4,
                width_per_group * 4 // BETA_INV,
            ],
            dim_inner=[dim_inner, dim_inner // BETA_INV],
            temp_kernel_sizes=temp_kernel[1],
            stride=FREQUENCY_STRIDES[0],
            num_blocks=[d2] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[0],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[0],
            norm_module=self.norm_module,
        )
        self.s2_fuse = FuseFastToSlow(
            width_per_group * 4 // BETA_INV,
            FUSION_CONV_CHANNEL_RATIO,
            FUSION_KERNEL_SZ,
            ALPHA,
            norm_module=self.norm_module,
        )

        for pathway in range(self.num_pathways):
            pool = nn.MaxPool2d(
                kernel_size=pool_size[pathway],
                stride=pool_size[pathway],
                padding=[0, 0],
            )
            self.add_module("pathway{}_pool".format(pathway), pool)

        self.s3 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 4 + width_per_group * 4 // out_dim_ratio,
                width_per_group * 4 // BETA_INV,
            ],
            dim_out=[
                width_per_group * 8,
                width_per_group * 8 // BETA_INV,
            ],
            dim_inner=[dim_inner * 2, dim_inner * 2 // BETA_INV],
            temp_kernel_sizes=temp_kernel[2],
            stride=FREQUENCY_STRIDES[1],
            num_blocks=[d3] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[1],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[1],
            norm_module=self.norm_module,
        )
        self.s3_fuse = FuseFastToSlow(
            width_per_group * 8 // BETA_INV,
            FUSION_CONV_CHANNEL_RATIO,
            FUSION_KERNEL_SZ,
            ALPHA,
            norm_module=self.norm_module,
        )

        self.s4 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 8 + width_per_group * 8 // out_dim_ratio,
                width_per_group * 8 // BETA_INV,
            ],
            dim_out=[
                width_per_group * 16,
                width_per_group * 16 // BETA_INV,
            ],
            dim_inner=[dim_inner * 4, dim_inner * 4 // BETA_INV],
            temp_kernel_sizes=temp_kernel[3],
            stride=FREQUENCY_STRIDES[2],
            num_blocks=[d4] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[2],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[2],
            norm_module=self.norm_module,
        )
        self.s4_fuse = FuseFastToSlow(
            width_per_group * 16 // BETA_INV,
            FUSION_CONV_CHANNEL_RATIO,
            FUSION_KERNEL_SZ,
            ALPHA,
            norm_module=self.norm_module,
        )

        self.s5 = resnet_helper.ResStage(
            dim_in=[
                width_per_group * 16 + width_per_group * 16 // out_dim_ratio,
                width_per_group * 16 // BETA_INV,
            ],
            dim_out=[
                width_per_group * 32,
                width_per_group * 32 // BETA_INV,
            ],
            dim_inner=[dim_inner * 8, dim_inner * 8 // BETA_INV],
            temp_kernel_sizes=temp_kernel[4],
            stride=FREQUENCY_STRIDES[3],
            num_blocks=[d5] * 2,
            num_groups=[num_groups] * 2,
            num_block_temp_kernel=NUM_BLOCK_TEMP_KERNEL[3],
            trans_func_name=TRANS_FUNC,
            dilation=FREQUENCY_DILATIONS[3],
            norm_module=self.norm_module,
        )

        self.head = head_helper.ResNetBasicHead(
            dim_in=[
                width_per_group * 32,
                width_per_group * 32 // BETA_INV,
            ],
            num_classes=NUM_CLASSES,
            pool_size=[
                [
                    NUM_FRAMES
                    // ALPHA // 4
                    // pool_size[0][0],
                    NUM_FREQUENCIES // 32 // pool_size[0][1],
                ],
                [
                    NUM_FRAMES // 4 // pool_size[1][0],
                    NUM_FREQUENCIES // 32 // pool_size[1][1],
                ],
            ],
            dropout_rate=DROPOUT_RATE,
            act_func=HEAD_ACT,
        )

    def forward(self, x, bboxes=None):
        x = self.s1(x)
        x = self.s1_fuse(x)
        x = self.s2(x)
        x = self.s2_fuse(x)
        for pathway in range(self.num_pathways):
            pool = getattr(self, "pathway{}_pool".format(pathway))
            x[pathway] = pool(x[pathway])
        x = self.s3(x)
        x = self.s3_fuse(x)
        x = self.s4(x)
        x = self.s4_fuse(x)
        x = self.s5(x)
        x = self.head(x)
        return x

    
        
if __name__ == '__main__':
    tensor = torch.rand([1, 2, 224, 224])
    model = ResNet(img_channels=2, num_layers=18, block=BasicBlock, num_classes=114).float()
    print(model)
    
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    output = model(tensor)
