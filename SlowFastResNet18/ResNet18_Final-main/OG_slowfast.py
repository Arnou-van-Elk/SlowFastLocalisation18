import torch.nn as nn
import torch

from torch import Tensor
from typing import Type # this is for type checking which is relevant when
                        # building complex architectures
import slowfast_utils

NUM_CLASSES = [114,]
_MODEL_STAGE_DEPTH = {18: (2, 2, 2,2), 34: (3, 4, 6, 3)}
BETA_INV = 8
FUSION_CONV_CHANNEL_RATIO = 2
ALPHA = 8
FUSION_KERNEL_SZ = 5
INPUT_CHANNEL_NUM = [1,1]
FREQUENCY_STRIDES = [[1],[2],[2],[2]]
FREQUENCY_DILATIONS = [[1],[1],[1],[1]]
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
        print('x_s:', x_s.shape)
        fuse = self.conv_f2s(x_f)
        fuse = self.bn(fuse)
        fuse = self.relu(fuse)
        print("fuse:", fuse.shape)
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

    def __init__(self):
        """
        The `__init__` method of any subclass should also contain these
            arguments.
        Args:
            cfg (CfgNode): model building configs, details are in the
                comments of the config file.
        """
        super(SlowFast, self).__init__()
        self.norm_module = nn.BatchNorm2d
        self.num_pathways = 2
        self._construct_network()
        # init_helper.init_weights(
        #     self, 0.01, False
        # )

    def _construct_network(self):
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
        self.s2 = slowfast_utils.ResStage(
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

        self.s3 = slowfast_utils.ResStage(
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

        self.s4 = slowfast_utils.ResStage(
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

        self.s5 = slowfast_utils.ResStage(
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

        self.head = slowfast_utils.ResNetBasicHead(
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
    #tensor = torch.rand([1, 2, 128, 200])
    #model = ResNet(img_channels=2, num_layers=34, block=BasicBlock, num_classes=114).float()
    slow_input = torch.rand(1, 1, 50, 128)   # Example dimension, e.g., [batch_size, channels, frames, H, W]
    fast_input = torch.rand(1, 1, 200, 64)   # Higher number of frames but possibly smaller spatial dims

    model = SlowFast()
    print(model)
    
    #Total parameters and trainable parameters.
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"{total_params:,} total parameters.")
    # total_trainable_params = sum(
    #     p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"{total_trainable_params:,} training parameters.")
    

     # Check total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")

    output = model([slow_input, fast_input])