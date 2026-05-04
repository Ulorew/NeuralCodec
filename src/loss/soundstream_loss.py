import math

import torch
import torch.nn.functional as F
import torchaudio
from torch import nn


class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_recon: torch.Tensor, **batch):
        mean_logit = [
            torch.max(torch.zeros_like(outs), 1 - outs).mean() for outs in logits_recon
        ]
        tot = torch.stack(mean_logit).mean()

        return {"adv_loss": tot}


class ReconstructionLoss(nn.Module):
    def __init__(self, n_mels=64):
        super().__init__()
        self.ss = [64, 128, 256, 512, 1024, 2048]
        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    n_mels=n_mels, n_fft=s, win_length=s, hop_length=s // 4
                )
                for s in self.ss
            ]
        )

    def forward(self, orig, recon, eps=1e-8, **batch):
        loss = 0
        for s, tf in zip(self.ss, self.transforms):
            mel_orig = tf(orig)
            mel_recon = tf(recon)

            alpha = math.sqrt(s / 2)

            loss += F.l1_loss(mel_orig, mel_recon)

            # loss_rec += alpha * F.mse_loss(
            #     torch.log(mel_recon + eps), torch.log(mel_orig + eps)
            # )

            loss += (
                alpha
                * torch.linalg.norm(
                    torch.log(mel_recon + eps) - torch.log(mel_orig + eps),
                    dim=-2,
                    ord=2,
                ).mean()
            )
            # TODO: return to simpler loss?
        return {"rec_loss": loss}


class FeatureLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, features_orig, features_recon, **batch):
        parts = []
        for layer_orig, layer_recon in zip(features_orig, features_recon):
            for f_orig, f_recon in zip(layer_orig, layer_recon):
                parts.append(F.l1_loss(f_recon, f_orig.detach()))

        loss = torch.stack(parts).mean()
        return {"feat_loss": loss}


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits_orig, logits_recon, **batch):
        mean_orig = [
            torch.max(torch.zeros_like(outs), 1 - outs).mean() for outs in logits_orig
        ]
        mean_fake = [
            torch.max(torch.zeros_like(outs), 1 + outs).mean() for outs in logits_recon
        ]

        tot = torch.stack(mean_orig).mean() + torch.stack(mean_fake).mean()
        return {"discr_loss": tot}
