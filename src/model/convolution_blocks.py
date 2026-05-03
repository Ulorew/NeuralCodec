from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super().__init__()

        self.lpad = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            padding=0,
        )

    def forward(self, x):
        x = F.pad(x, (self.lpad, 0))
        x = self.conv(x)
        return x


class CausalConvTranspose1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super().__init__()

        self.rcrop = kernel_size - stride

        self.tconv = nn.ConvTranspose1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
        )

    def forward(self, x):
        x = self.tconv(x)
        if self.rcrop > 0:
            x = x[..., : -self.rcrop]
        return x


class ResidualUnit(nn.Module):
    def __init__(self, N, dilation):
        super().__init__()
        self.act = nn.ELU()
        self.conv1 = CausalConv1d(N, N, kernel_size=7, dilation=dilation)
        self.conv2 = CausalConv1d(N, N, kernel_size=1)

    def forward(self, x):
        y = self.conv1(self.act(x))
        y = self.conv2(self.act(y))
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
        self.ru1 = ResidualUnit(N // 2, dilation=1)
        self.ru2 = ResidualUnit(N // 2, dilation=3)
        self.ru3 = ResidualUnit(N // 2, dilation=9)
        self.act = nn.ELU()
        self.conv = CausalConv1d(N // 2, N, kernel_size=2 * S, stride=S)

    def forward(self, x):
        x = self.ru1(x)
        x = self.ru2(x)
        x = self.ru3(x)
        x = self.conv(self.act(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, N, S):
        super().__init__()
        self.act = nn.ELU()
        self.tconv = CausalConvTranspose1d(2 * N, N, kernel_size=2 * S, stride=S)
        self.ru1 = ResidualUnit(N, dilation=1)
        self.ru2 = ResidualUnit(N, dilation=3)
        self.ru3 = ResidualUnit(N, dilation=9)

    def forward(self, x):
        x = self.tconv(self.act(x))
        x = self.ru1(x)
        x = self.ru2(x)
        x = self.ru3(x)
        return x
