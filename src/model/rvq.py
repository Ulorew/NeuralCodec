import torch
from torch import nn
from torch.nn import functional as F


class RVQCodebook(nn.Module):
    def __init__(self, size, dim, eps=1e-5, eta=0.99):
        super().__init__()
        # TODO: dead updates, knn init

        self.size = size
        self.dim = dim
        self.eta = eta
        self.eps = eps

        book = torch.randn(size, dim)

        self.register_buffer("book", book)
        self.register_buffer("ema_sum", book.clone())
        self.register_buffer("ema_cnt", torch.ones(size))

    @torch.no_grad()
    def forward(self, x, update_codebook=False):
        # x : (B, T, D)
        B, T, D = x.shape
        y = x.reshape(-1, D)

        cross = y @ self.book.transpose(0, 1)  # TODO: matrices
        dist = (
            (y**2).sum(dim=1).unsqueeze(1)
            + (self.book**2).sum(dim=1).unsqueeze(0)
            - 2 * cross
        )
        pivot_id = dist.argmin(dim=-1)

        pivot = self.book[pivot_id].view(B, T, D)

        if self.training and update_codebook:
            mask = F.one_hot(pivot_id, self.size).to(x.dtype)  # (B*T, C)

            nbcnt = mask.sum(dim=0)

            nbsum = mask.transpose(0, 1) @ y  # (C, D)

            self.ema_cnt.mul_(self.eta).add_(nbcnt * (1 - self.eta))
            self.ema_sum.mul_(self.eta).add_(nbsum * (1 - self.eta))

            self.book.copy_(
                self.ema_sum / self.ema_cnt.clamp_min(self.eps).unsqueeze(1)
            )  # TODO: switch to eq?

        return pivot, pivot_id.view(B, T)


class RVQ(nn.Module):
    def __init__(self, cb_cnt, cb_size, dim, eta=0.99):
        super().__init__()

        self.dim = dim
        self.books = nn.ModuleList(
            [RVQCodebook(cb_size, dim, eta=eta) for _ in range(cb_cnt)]
        )

    def forward(self, lat_raw, update_codebook=False, **batch):
        y = lat_raw.clone().transpose(1, 2)  # -> (B, T, D)
        z = torch.zeros_like(y)
        codes = []

        for book in self.books:
            rec, ids = book(y, update_codebook=update_codebook)
            codes.append(ids)

            y -= rec
            z += rec

        codes = torch.stack(codes).permute(1, 2, 0)

        z = z.transpose(1, 2)  # -> (B, D, T)
        skipgrad = lat_raw + (z - lat_raw).detach()

        return {"residual": y, "codes": codes, "lat_quant": skipgrad}
