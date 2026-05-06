import torch
from torch import nn
from torch.nn import functional as F


class RVQCodebook(nn.Module):
    def __init__(self, size, dim, min_nb_ratio=5e-4, eps=1e-6, eta=0.99):
        super().__init__()
        # TODO: dead updates, knn init

        self.size = size
        self.dim = dim
        self.eta = eta
        self.eps = eps
        self.min_nb_ratio = min_nb_ratio

        book = torch.randn(size, dim)

        self.register_buffer("book", book)
        self.register_buffer("ema_sum", book.clone() * min_nb_ratio)
        self.register_buffer(
            "ema_cnt", torch.full((size,), min_nb_ratio, dtype=torch.float)
        )

    @torch.no_grad()
    def forward(self, x, update_codebook=False):
        # x : (B, T, D)
        B, T, D = x.shape
        N = B * T
        y = x.reshape(N, D)

        cross = y @ self.book.transpose(0, 1)  # TODO: matrices
        dist = (
            (y**2).sum(dim=1).unsqueeze(1)
            + (self.book**2).sum(dim=1).unsqueeze(0)
            - 2 * cross
        )
        pivot_id = dist.argmin(dim=-1)

        pivot = self.book[pivot_id].view(B, T, D)
        dead_cnt = torch.zeros((), dtype=torch.long, device=x.device)
        perp = torch.zeros((), dtype=torch.float, device=x.device)

        if self.training and update_codebook:
            mask = F.one_hot(pivot_id, self.size).to(x.dtype)  # (N, C)

            nbcnt = mask.sum(dim=0) / N

            nbsum = (mask.transpose(0, 1) @ y) / N  # (C, D)

            self.ema_cnt.mul_(self.eta).add_(nbcnt * (1 - self.eta))
            self.ema_sum.mul_(self.eta).add_(nbsum * (1 - self.eta))

            p = self.ema_cnt / self.ema_cnt.sum().clamp_min(self.eps)
            perp = (-(p * p.clamp_min(self.eps).log()).sum()).exp()

            self.book.copy_(
                self.ema_sum / self.ema_cnt.clamp_min(self.eps).unsqueeze(1)
            )  # TODO: switch to eq?

            dead_mask = self.ema_cnt < self.min_nb_ratio
            dead_cnt = dead_mask.sum().to(torch.long)
            samples = y[
                torch.randint(0, B * T, size=(dead_cnt.item(),), device=x.device)
            ]

            self.ema_cnt[dead_mask] = self.min_nb_ratio
            self.ema_sum[dead_mask] = samples * self.min_nb_ratio
            self.book[dead_mask] = samples

        return pivot, pivot_id.view(B, T), dead_cnt, perp


class RVQ(nn.Module):
    def __init__(self, cb_cnt, cb_size, dim, eta=0.99):
        super().__init__()

        self.dim = dim
        self.cb_cnt = cb_cnt
        self.cb_size = cb_size
        self.books = nn.ModuleList(
            [RVQCodebook(cb_size, dim, eta=eta) for _ in range(cb_cnt)]
        )

    def forward(self, lat_raw, update_codebook=False, **batch):
        y = lat_raw.clone().transpose(1, 2)  # -> (B, T, D)
        z = torch.zeros_like(y)
        codes = []
        dead_cnt = torch.zeros((), device=lat_raw.device)
        perps = []

        for book in self.books:
            rec, ids, dc, perp = book(y, update_codebook=update_codebook)
            codes.append(ids)
            perps.append(perp)
            dead_cnt += dc

            y -= rec
            z += rec

        codes = torch.stack(codes).permute(1, 2, 0)
        perps = torch.stack(perps)

        z = z.transpose(1, 2)  # -> (B, D, T)
        skipgrad = lat_raw + (z - lat_raw).detach()

        dead_ratio = dead_cnt / (self.cb_cnt * self.cb_size)
        out = {f"perplexity_book_{i}": perp for i, perp in enumerate(perps)}
        out.update(
            {
                "residual": y,
                "codes": codes,
                "lat_quant": skipgrad,
                "dead_code_ratio": dead_ratio,
                "mean_perplexity": perps.mean(),
            }
        )

        return out
