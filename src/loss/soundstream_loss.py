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
        self.ss = [256, 512, 1024, 2048]  # TODO [64, 128] too small for n_mels=64

        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    n_mels=n_mels, n_fft=s, win_length=s, hop_length=s // 4, power=1
                )
                for s in self.ss
            ]
        )

    def forward(self, orig, recon, eps=1e-8, **batch):
        # loss = 0
        # loss += F.l1_loss(orig, recon) * 100
        #
        # tf = self.transforms[3]
        # mel_orig = tf(orig)
        # mel_recon = tf(recon)
        #
        # loss += F.l1_loss(mel_orig, mel_recon)

        out = {}

        for s, tf in zip(self.ss, self.transforms):
            mel_orig = tf(orig)
            mel_recon = tf(recon)

            mel_loss = F.l1_loss(mel_orig, mel_recon)
            out[f"rec_loss_mel_{s}"] = mel_loss

            # loss += alpha * F.mse_loss(
            #     torch.log(mel_recon + eps), torch.log(mel_orig + eps)
            # )

            # alpha = math.sqrt(s / 2)
            # loss += (
            #     alpha
            #     * torch.linalg.norm(
            #         torch.log(mel_recon + eps) - torch.log(mel_orig + eps),
            #         dim=-2,
            #         ord=2,
            #     ).mean()
            # )
        # TODO: return to simpler loss?

        loss = torch.stack(list(out.values())).mean()
        out["rec_loss"] = loss

        return out


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


class GeneratorLoss(nn.Module):
    def __init__(self, l_adv=1, l_feat=100, l_rec=1, rec_n_mels=64):
        super().__init__()
        self.rec_loss = ReconstructionLoss(n_mels=rec_n_mels)
        self.feat_loss = FeatureLoss()
        self.adv_loss = AdversarialLoss()

        self.l_adv = l_adv
        self.l_feat = l_feat
        self.l_rec = l_rec

    def forward(self, **batch):
        out = {}
        out.update(self.rec_loss(**batch))
        out.update(self.feat_loss(**batch))
        out.update(self.adv_loss(**batch))

        loss = (
            self.l_adv * out["adv_loss"]
            + self.l_feat * out["feat_loss"]
            + self.l_rec * out["rec_loss"]
        )
        out["gen_loss"] = loss
        return out


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

        orig = torch.stack(mean_orig).mean()
        fake = torch.stack(mean_fake).mean()
        tot = orig + fake
        return {"discr_loss": tot, "discr_orig_loss": orig, "discr_fake_loss": fake}
