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
    def __init__(
        self,
        n_mels=64,
        scales=(64, 128, 256, 512, 1024, 2048),
        log_weight=1.0,
        eps=1e-5,
    ):
        super().__init__()

        self.ss = list(scales)
        self.log_weight = log_weight
        self.eps = eps

        self.transforms = nn.ModuleList(
            [
                torchaudio.transforms.MelSpectrogram(
                    n_mels=n_mels,
                    n_fft=s,
                    win_length=s,
                    hop_length=s // 4,
                    power=1.0,
                )
                for s in self.ss
            ]
        )

    def forward(self, orig, recon, **batch):
        out = {}
        losses = []

        for s, tf in zip(self.ss, self.transforms):
            mel_orig = tf(orig)
            mel_recon = tf(recon)

            mel_l1 = F.l1_loss(mel_recon, mel_orig)

            log_diff = torch.log(mel_recon.clamp_min(self.eps)) - torch.log(
                mel_orig.clamp_min(self.eps)
            )

            log_l2 = torch.linalg.vector_norm(log_diff, ord=2, dim=-2).mean()

            alpha = math.sqrt(s / 2)

            loss_s = mel_l1 + self.log_weight * alpha * log_l2

            out[f"rec_loss_mel_{s}"] = mel_l1
            out[f"rec_loss_logmel_{s}"] = log_l2
            out[f"rec_loss_scale_{s}"] = loss_s

            losses.append(loss_s)

        out["rec_loss"] = torch.stack(losses).mean()
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


class CommitmentLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lat_raw, lat_quant, **batch):
        loss = F.mse_loss(lat_raw, lat_quant.detach())
        return {"comm_loss": loss}


class GeneratorLoss(nn.Module):
    def __init__(
        self, l_adv=1, l_feat=100, l_rec=1, l_comm=1, rec_n_mels=64, rec_log_weight=1.0
    ):
        super().__init__()
        self.rec_loss = ReconstructionLoss(n_mels=rec_n_mels, log_weight=rec_log_weight)
        self.feat_loss = FeatureLoss()
        self.adv_loss = AdversarialLoss()
        self.comm_loss = CommitmentLoss()

        self.l_adv = l_adv
        self.l_feat = l_feat
        self.l_rec = l_rec
        self.l_comm = l_comm
        self.progress = 0
        self.alpha = 0.0

    def update_progress(self, prog: float):
        self.progress = prog
        self.alpha = min(1.0, prog / 0.5)

    def forward(self, **batch):
        out = {}
        out.update(self.rec_loss(**batch))
        out.update(self.feat_loss(**batch))
        out.update(self.adv_loss(**batch))
        out.update(self.comm_loss(**batch))

        loss = (
            self.l_adv * self.alpha * out["adv_loss"]
            + self.l_feat * self.alpha * out["feat_loss"]
            + self.l_rec * out["rec_loss"]
            + self.l_comm * out["comm_loss"]
        )
        out["gen_loss"] = loss
        out["gan_alpha"] = torch.tensor(self.alpha, device=out["rec_loss"].device)
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
