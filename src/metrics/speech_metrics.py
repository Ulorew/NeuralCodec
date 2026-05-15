import warnings

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

    def __call__(self, orig: torch.Tensor, recon: torch.Tensor, length: torch.Tensor, **batch):
        scores = []
        for orig_s, recon_s, l in zip(orig, recon, length):
            orig_s = orig_s[:l].detach()
            recon_s = recon_s[:l].detach()
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning)
                stoi_s = self.metric(orig_s, recon_s)

            if _has_short_stoi_warning(caught_warnings):
                continue

            scores.append(stoi_s)

        return _mean_or_none(scores)


class NISQAMetric(BaseMetric):
    def __init__(self, sampling_rate, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric = NonIntrusiveSpeechQualityAssessment(sampling_rate)

    def __call__(self, recon: torch.Tensor, length: torch.Tensor, **batch):
        scores = []
        for recon_s, l in zip(recon, length):
            recon_s = recon_s[:l].detach()
            try:
                nisqa_s = self.metric(recon_s)
            except RuntimeError as err:
                if "Input signal is too short" not in str(err):
                    raise
                continue
            scores.append(nisqa_s)

        return _mean_or_none(scores)


def _has_short_stoi_warning(caught_warnings):
    return any(
        warning.category is RuntimeWarning
        and "Not enough STFT frames" in str(warning.message)
        for warning in caught_warnings
    )


def _mean_or_none(scores):
    if not scores:
        return None
    return torch.stack(scores).mean().item()
