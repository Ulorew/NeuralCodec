import torch
import torch.nn.functional as F

from src.metrics.base_metric import BaseMetric


class QuantDistMetric(BaseMetric):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, lat_raw: torch.Tensor, lat_quant: torch.Tensor, **batch):
        return F.mse_loss(lat_raw, lat_quant)


# class MeanNBCnt(BaseMetric):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#     def __call__(self, lat_raw: torch.Tensor, lat_quant: torch.Tensor, **batch):
#         return F.mse_loss(lat_raw, lat_quant)
