import torch
import torch.nn as nn
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int,
            activation=nn.ReLU(),
            groups=1,
            batch_norm=True,
            bias=False
    ):
        super().__init__()

        # If k = 1 -> p = 0, k = 3 -> p = 1, k = 5, p = 2.
        padding = kernel_size // 2
        self.c = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.act = activation

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.bn(self.c(x)))


class SEBlock(nn.Module):
    # Squeeze and Excitation Block.
    def __init__(
            self,
            in_channels: int
    ):
        super().__init__()

        in_ch = in_channels
        res = in_ch // 4
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_ch, res, bias=False)
        self.fc2 = nn.Linear(res, in_ch, bias=False)
        self.relu = nn.ReLU()
        self.h_sigmoid = nn.Hardsigmoid()

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [N, C, H, W].
        f = self.global_pool(x)
        f = torch.flatten(f, 1)
        f = self.relu(self.fc1(f))
        f = self.h_sigmoid(self.fc2(f))
        f = f[:, :, None, None]
        # f shape: [N, C, 1, 1]

        scale = x * f
        return scale


# BNeck
class BNeck(nn.Module):
    # MobileNetV3 Block
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            exp_size: int,
            se: bool,
            act: torch.nn.modules.activation,
            stride: int
    ):
        super().__init__()

        self.add = in_channels == out_channels and stride == 1

        self.block = nn.Sequential(
            ConvBlock(in_channels, exp_size, 1, 1, act),
            ConvBlock(exp_size, exp_size, kernel_size, stride, act, exp_size),
            SEBlock(exp_size) if se is True else nn.Identity(),
            ConvBlock(exp_size, out_channels, 1, 1, activation=nn.Identity())
        )

    def forward(self, x: Tensor) -> Tensor:
        res = self.block(x)
        if self.add:
            res = res + x

        return res


""" MobileNetV3 """


class MobileNetV3(nn.Module):
    def __init__(
            self,
            config_name: str,
            in_channels=3,
            classes=1000
    ):
        super().__init__()
        config = self.config(config_name)

        # First convolution(conv2d) layer.
        self.conv = ConvBlock(in_channels, 16, 3, 2, nn.Hardswish())
        # Bneck blocks in a list.
        self.blocks = nn.ModuleList([])
        for c in config:
            kernel_size, exp_size, in_channels, out_channels, se, nl, s = c
            self.blocks.append(BNeck(in_channels, out_channels, kernel_size, exp_size, se, nl, s))

        # Classifier
        last_out_channel = config[-1][3]
        last_exp = config[-1][1]
        out = 1280 if config_name == "large" else 1024
        self.classifier = nn.Sequential(
            ConvBlock(last_out_channel, last_exp, 1, 1, nn.Hardswish()),
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvBlock(last_exp, out, 1, 1, nn.Hardswish(), batch_norm=False, bias=True),
            nn.Dropout(0.8),
            nn.Conv2d(out, classes, 1, 1)
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        for block in self.blocks:
            x = block(x)

        x = self.classifier(x)
        return torch.flatten(x, 1)

    def config(self, variant_name):
        hs, re = nn.Hardswish(), nn.ReLU()
        # [kernel, exp size, in_channels, out_channels, SEBlock(SE), activation function(NL), stride(s)]
        large = [
            [3, 16, 16, 16, False, re, 1],
            [3, 64, 16, 24, False, re, 2],
            [3, 72, 24, 24, False, re, 1],
            [5, 72, 24, 40, True, re, 2],
            [5, 120, 40, 40, True, re, 1],
            [5, 120, 40, 40, True, re, 1],
            [3, 240, 40, 80, False, hs, 2],
            [3, 200, 80, 80, False, hs, 1],
            [3, 184, 80, 80, False, hs, 1],
            [3, 184, 80, 80, False, hs, 1],
            [3, 480, 80, 112, True, hs, 1],
            [3, 672, 112, 112, True, hs, 1],
            [5, 672, 112, 160, True, hs, 2],
            [5, 960, 160, 160, True, hs, 1],
            [5, 960, 160, 160, True, hs, 1]
        ]

        small = [
            [3, 16, 16, 16, True, re, 2],
            [3, 72, 16, 24, False, re, 2],
            [3, 88, 24, 24, False, re, 1],
            [5, 96, 24, 40, True, hs, 2],
            [5, 240, 40, 40, True, hs, 1],
            [5, 240, 40, 40, True, hs, 1],
            [5, 120, 40, 48, True, hs, 1],
            [5, 144, 48, 48, True, hs, 1],
            [5, 288, 48, 96, True, hs, 2],
            [5, 576, 96, 96, True, hs, 1],
            [5, 576, 96, 96, True, hs, 1]
        ]

        if variant_name == "large":
            return large
        if variant_name == "small":
            return small


if __name__ == "__main__":
    name = "large"
    rho = 1
    resolution = int(rho * 224)

    net = MobileNetV3(name)
    print(net(torch.rand(1, 3, resolution, resolution)).shape)
