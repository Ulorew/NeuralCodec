import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm

from src.model.convolution_blocks import (
    CausalConv1d,
    DecoderBlock,
    DiscrResidualUnit,
    EncoderBlock,
    SamePadConv2d,
)
from src.model.rvq import RVQ
from src.utils.misc import wrap_keys


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


class SoundStreamGenerator(nn.Module):
    def __init__(self, base_channels, cb_cnt, cb_size, code_dim, rvq_eta, min_nb_ratio):
        super().__init__()

        self.enc = Encoder(base_channels, code_dim)
        self.dec = Decoder(base_channels, code_dim)
        self.rvq = RVQ(
            cb_cnt, cb_size, code_dim, eta=rvq_eta, min_nb_ratio=min_nb_ratio
        )

    def forward(self, orig, update_codebook=False, **batch):
        lat_raw = self.enc(orig)
        out = self.rvq(lat_raw, update_codebook=update_codebook)
        recon = self.dec(out["lat_quant"])

        out.update({"lat_raw": lat_raw, "recon": recon})
        assert out["lat_raw"].shape == out["lat_quant"].shape
        return out

    def encode(self, orig, update_codebook=False, **batch):
        lat_raw = self.enc(orig)
        out = self.rvq(lat_raw, update_codebook=update_codebook)
        out["lat_raw"] = lat_raw
        return out

    def decode(self, codes=None, lat_quant=None, **batch):
        out = {}

        if lat_quant is None:
            out = self.rvq.reconstruct(codes)
            lat_quant = out["lat_quant"]

        recon = self.dec(lat_quant)

        out["recon"] = recon

        return out


class WaveDiscriminator(nn.Module):
    """
    check https://github.com/descriptinc/melgan-neurips/blob/master/melgan_slides.pdf
    """

    def __init__(self, in_channels=1, C=1):  # C = ?
        super().__init__()

        self.layers = nn.ModuleList(
            [
                # plain projection
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            in_channels, C * 16, kernel_size=15, padding=7, stride=1
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                # 4 grouped
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            C * 16,
                            C * 64,
                            kernel_size=41,
                            stride=4,
                            padding=20,
                            groups=C * 4,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            C * 64,
                            C * 256,
                            kernel_size=41,
                            stride=4,
                            padding=20,
                            groups=C * 16,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            C * 256,
                            C * 1024,
                            kernel_size=41,
                            stride=4,
                            padding=20,
                            groups=C * 64,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            C * 1024,
                            C * 1024,
                            kernel_size=41,
                            stride=4,
                            padding=20,
                            groups=C * 256,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                # plain FF & projection
                nn.Sequential(
                    weight_norm(
                        nn.Conv1d(
                            C * 1024,
                            C * 1024,
                            kernel_size=5,
                            stride=1,
                            padding=2,
                        )
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                ),
                weight_norm(nn.Conv1d(C * 1024, 1, kernel_size=3, stride=1, padding=1)),
            ]
        )

    def forward(self, x):
        feats = []
        for layer in self.layers:
            x = layer(x)
            feats.append(x)

        out = feats[-1]
        return {
            "logits": out,
            "features": feats[:-1],
        }


class STFTDiscriminator(nn.Module):
    def __init__(self, W: int = 1024, H: int = 256, C: int = 1, F: int = None):
        super().__init__()
        self.W = W
        self.H = H

        if F is None:
            F = W // 2
        self.F = F
        if F % 64 != 0:
            raise ValueError("F should be divisible by 64")

        self.layers = nn.ModuleList(
            [
                SamePadConv2d(2, C, kernel_size=(7, 7)),
                DiscrResidualUnit(
                    in_channels=C, mid_channels=C, out_channels=2 * C, stride=(1, 2)
                ),
                DiscrResidualUnit(
                    in_channels=2 * C,
                    out_channels=4 * C,
                    stride=(2, 2),
                ),
                DiscrResidualUnit(
                    in_channels=4 * C,
                    out_channels=4 * C,
                    stride=(1, 2),
                ),
                DiscrResidualUnit(
                    in_channels=4 * C,
                    out_channels=8 * C,
                    stride=(2, 2),
                ),
                DiscrResidualUnit(
                    in_channels=8 * C,
                    out_channels=8 * C,
                    stride=(1, 2),
                ),
                DiscrResidualUnit(
                    in_channels=8 * C,
                    out_channels=16 * C,
                    stride=(2, 2),
                ),
                nn.Conv2d(in_channels=16 * C, out_channels=1, kernel_size=(1, F // 64)),
            ]
        )

    def forward(self, x):
        if x.shape[1] != 1:
            raise ValueError(f"Expected signal to be single channel, got {x.shape[1]}")
        x = x.squeeze(1)

        window = torch.hann_window(self.W, device=x.device, dtype=x.dtype)
        x = torch.stft(
            x,
            n_fft=self.W,
            hop_length=self.H,
            window=window,
            return_complex=True,
        )
        x = x[:, : self.F, :]  # cut last
        x = torch.view_as_real(x)  # (B, bins, frames, 2)
        x = x.permute(0, 3, 2, 1)  # (B, 2, frames, bins)

        feats = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                feats.append(x)

        out = x.squeeze(-1)
        return {
            "logits": out,
            "features": feats,
        }


class SoundStreamDiscriminator(nn.Module):
    def __init__(self, stft_channels=32, wave_width=1):
        super().__init__()
        self.stft = STFTDiscriminator(W=1024, C=stft_channels)
        self.waves = nn.ModuleList([WaveDiscriminator(C=wave_width) for _ in range(3)])

        self.downsample = nn.AvgPool1d(
            kernel_size=4, stride=2, padding=1, count_include_pad=False
        )

    def forward(self, x):
        mid = self.stft(x)
        output = {k: [v] for k, v in mid.items()}

        for i, d in enumerate(self.waves):
            mid = d(x)
            for k, v in mid.items():
                output[k].append(v)

            if i < len(self.waves) - 1:
                x = self.downsample(x)

        return output


class SoundStreamGAN(nn.Module):
    def __init__(
            self,
            gen_base_channels,
            discr_stft_channels,
            discr_wave_width,
            cb_cnt,
            cb_size,
            code_dim,
            rvq_eta,
            min_nb_ratio,
    ):
        super().__init__()
        self.gen = SoundStreamGenerator(
            gen_base_channels,
            cb_cnt,
            cb_size,
            code_dim,
            rvq_eta,
            min_nb_ratio=min_nb_ratio,
        )
        self.discr = SoundStreamDiscriminator(discr_stft_channels, discr_wave_width)

    def forward(self, **batch):
        return self.gen(**batch)

    def discriminate(self, x, key_pref="", key_suff=""):
        out = self.discr(x)

        wrap_keys(out, ["logits", "features"], pref=key_pref, suff=key_suff)
        return out
