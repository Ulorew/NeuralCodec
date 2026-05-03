import torch
from torch import nn


class Normalize1D(nn.Module):
    """
    Batch-version of Normalize for audio input.
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        peak_amp = x.abs().max()
        x /= peak_amp
        return x
