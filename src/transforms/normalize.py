import torch
from torch import nn


class Normalize1D(nn.Module):
    """
    Batch-version of Normalize for audio input.
    """

    def __init__(self, eps=1e-4):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        peak_amp = x.abs().max(dim=-1)[0].unsqueeze(-1).clamp_min_(self.eps)
        x /= peak_amp
        return x
