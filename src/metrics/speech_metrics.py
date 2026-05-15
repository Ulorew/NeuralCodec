import torch
from torchmetrics.audio import (
    NonIntrusiveSpeechQualityAssessment,
    ShortTimeObjectiveIntelligibility,
)

from src.metrics.base_metric import BaseMetric


class STOIMetric(BaseMetric):
    def __init__(self, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = ShortTimeObjectiveIntelligibility(sampling_rate)

    def __call__(self, orig: torch.Tensor, recon: torch.Tensor, length : torch.Tensor, **batch):
        scores=[]
        for orig_s, recon_s, l in zip(orig, recon, length):
            orig_s = orig_s[:l].detach()
            recon_s = recon_s[:l].detach()
            stoi_s = self.metric(orig_s, recon_s)
            scores.append(stoi_s)

        return torch.stack(scores).mean().item()


class NISQAMetric(BaseMetric):
    def __init__(self, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = NonIntrusiveSpeechQualityAssessment(sampling_rate)

    def __call__(self, recon: torch.Tensor, length:torch.Tensor, **batch):
        scores = []
        for recon_s, l in zip(recon, length):
            recon_s = recon_s[:l].detach()
            nisqa_s = self.metric(recon_s)
            scores.append(nisqa_s)

        return torch.stack(scores).mean().item()
