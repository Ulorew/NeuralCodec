from torch import nn

from model.convolution_blocks import CausalConv1d, DecoderBlock, EncoderBlock


class Encoder(nn.Module):
    def __init__(self, C, K):
        super().__init__()

        self.conv1 = CausalConv1d(1, C, kernel_size=7)
        self.eb1 = EncoderBlock(2 * C, S=2)
        self.eb2 = EncoderBlock(4 * C, S=4)
        self.eb3 = EncoderBlock(8 * C, S=5)
        self.eb4 = EncoderBlock(16 * C, S=5)
        self.conv2 = CausalConv1d(16 * C, K, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.eb1(x)
        x = self.eb2(x)
        x = self.eb3(x)
        x = self.eb4(x)
        x = self.conv2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, C, K):
        super().__init__()

        self.conv1 = CausalConv1d(K, 16 * C, kernel_size=7)

        self.db1 = DecoderBlock(8 * C, S=5)
        self.db2 = DecoderBlock(4 * C, S=5)
        self.db3 = DecoderBlock(2 * C, S=4)
        self.db4 = DecoderBlock(C, S=2)

        self.conv2 = CausalConv1d(C, 1, kernel_size=7)

    def forward(self, x):
        x = self.conv1(x)
        x = self.db1(x)
        x = self.db2(x)
        x = self.db3(x)
        x = self.db4(x)
        x = self.conv2(x)
        return x


class SoundStream(nn.Module):
    def __init__(self, base_channels, lat_dim):
        super().__init__()

        self.enc = Encoder(base_channels, lat_dim)
        self.dec = Decoder(base_channels, lat_dim)

    def forward(self, x):
        y = self.enc(x)
        z = self.dec(y)
        return z
