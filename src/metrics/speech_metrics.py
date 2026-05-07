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

    def __call__(self, orig: torch.Tensor, recon: torch.Tensor, **batch):
        stoi = self.metric(recon, orig)
        return stoi


class NISQAMetric(BaseMetric):
    def __init__(self, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = NonIntrusiveSpeechQualityAssessment(sampling_rate)

    def __call__(self, recon: torch.Tensor, **batch):
        recon = recon.detach()
        if recon.dim() == 3:
            recon = recon.squeeze(1)
        nisqa = self.metric(recon.detach()).mean()
        return nisqa
