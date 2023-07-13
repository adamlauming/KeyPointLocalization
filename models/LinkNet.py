import numpy as np
import torch
import torch.nn as nn
import torchsummary
from torch.nn import functional as F
from torch.nn import init
from torchvision import models

from models.layers.init_weights import init_weights
from models.layers.unet_layers import *

#===================================================================================
# LinkNet
#===================================================================================
# 解码部分其中一个1x1卷积换成3x3卷积
class ResUnet4(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channel_reduction=2):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]
        filters = [int(c / channel_reduction) for c in filters]
        resnet = models.resnet18(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock2(512, filters[2])
        self.decoder3 = DecoderBlock2(filters[2], filters[1])
        self.decoder2 = DecoderBlock2(filters[1], filters[0])
        self.decoder1 = DecoderBlock2(filters[0], filters[0])

        self.finalconv = nn.Conv2d(in_channels=filters[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        outputs = dict()
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finalconv(d1)

        outputs.update({"out": out})
        return outputs


# 中间增加若干的gap 卷积
class ResUnet3(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channel_reduction=2):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]
        filters = [int(c / channel_reduction) for c in filters]
        resnet = models.resnet18(pretrained=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.gap0 = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.gap1 = nn.Conv2d(filters[0], filters[0], 3, padding=1)
        self.gap2 = nn.Conv2d(filters[1], filters[1], 3, padding=1)

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finalconv = nn.Conv2d(in_channels=filters[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        outputs = dict()
        # Encoder
        x = self.upsample(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        e1 = self.encoder1(self.firstmaxpool(x))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + self.gap2(e2) + e2
        d2 = self.decoder2(d3) + self.gap1(e1) + e1
        d1 = self.decoder1(d2) + self.gap0(x) + x

        out = self.finalconv(d1)

        outputs.update({"out": out})
        return outputs


# 增加第一层的跨连接
class ResUnet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channel_reduction=2):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]
        filters = [int(c / channel_reduction) for c in filters]
        resnet = models.resnet18(pretrained=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finalconv = nn.Conv2d(in_channels=filters[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        outputs = dict()
        # Encoder
        x = self.upsample(x)
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)

        e1 = self.firstmaxpool(x)
        e1 = self.encoder1(e1)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2) + x

        out = self.finalconv(d1)

        outputs.update({"out": out})
        return outputs


class ResUnet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, channel_reduction=2):
        super().__init__()

        filters = [64, 128, 256, 512, 1024]
        filters = [int(c / channel_reduction) for c in filters]
        resnet = models.resnet18(pretrained=True)

        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool

        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.decoder4 = DecoderBlock(512, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finalconv = nn.Conv2d(in_channels=filters[0], out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        outputs = dict()
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        # x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finalconv(d1)
        # out = F.relu(out)

        outputs.update({"out": out})
        return outputs




#===================================================================================
# Basic Unet
#===================================================================================
# class Unet(nn.Module):
#     def __init__(self, n_filters=16, in_channels=1, out_channels=1):
#         super().__init__()

#         filters = n_filters * np.array([1, 2, 4, 8, 16])

#         self.conv_down1 = double_conv(in_channels, filters[0])
#         self.conv_down2 = double_conv(filters[0], filters[1])
#         self.conv_down3 = double_conv(filters[1], filters[2])
#         self.conv_down4 = double_conv(filters[2], filters[3])
#         self.conv_down5 = double_conv(filters[3], filters[4])

#         self.maxpool = nn.MaxPool2d(2)
#         self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

#         self.conv_up4 = double_conv(filters[3] + filters[4], filters[3])
#         self.conv_up3 = double_conv(filters[2] + filters[3], filters[2])
#         self.conv_up2 = double_conv(filters[1] + filters[2], filters[1])
#         self.conv_up1 = double_conv(filters[0] + filters[1], filters[0])

#         self.conv_last = nn.Conv2d(in_channels=filters[0], out_channels=out_channels, kernel_size=1)

#         # initialise weights
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init_weights(m, init_type='kaiming')
#             elif isinstance(m, nn.BatchNorm2d):
#                 init_weights(m, init_type='kaiming')

#     def forward(self, x):
#         conv1_2 = self.conv_down1(x)
#         conv2_1 = self.maxpool(conv1_2)

#         conv2_2 = self.conv_down2(conv2_1)
#         conv3_1 = self.maxpool(conv2_2)

#         conv3_2 = self.conv_down3(conv3_1)
#         conv4_1 = self.maxpool(conv3_2)

#         conv4_2 = self.conv_down4(conv4_1)
#         conv5_1 = self.maxpool(conv4_2)

#         conv5_2 = self.conv_down5(conv5_1)

#         conv4_3 = self.upsample(conv5_2)
#         conv4_3 = torch.cat([conv4_3, conv4_2], dim=1)
#         conv4_4 = self.conv_up4(conv4_3)

#         conv3_3 = self.upsample(conv4_4)
#         conv3_3 = torch.cat([conv3_3, conv3_2], dim=1)
#         conv3_4 = self.conv_up3(conv3_3)

#         conv2_3 = self.upsample(conv3_4)
#         conv2_3 = torch.cat([conv2_3, conv2_2], dim=1)
#         conv2_4 = self.conv_up2(conv2_3)

#         conv1_3 = self.upsample(conv2_4)
#         conv1_3 = torch.cat([conv1_3, conv1_2], dim=1)
#         conv1_4 = self.conv_up1(conv1_3)

#         out = self.conv_last(conv1_4)
#         out = torch.sigmoid(out)
#         # out = F.relu(out)

#         return out
