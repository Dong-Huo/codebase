import torch
import torch.nn as nn
import torch.nn.functional as F

class MonotoneLinear(nn.Module):
    """
    Linear layer with provably nonnegative weights via softplus parameterization.
    Ensures monotonicity w.r.t. nonnegative inputs when used with monotone activations.
    """
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.W_raw = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        nn.init.normal_(self.W_raw, mean=0.0, std=0.02)

    def forward(self, x):
        W = F.softplus(self.W_raw)  # >= 0
        y = x @ W.t()
        if self.bias is not None:
            y = y + self.bias
        return y

class MonotoneMLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=128, num_layers=3, act="softplus"):
        super().__init__()
        assert num_layers >= 2
        acts = {
            "relu": nn.ReLU(),
            "silu": nn.SiLU(),
            "softplus": nn.Softplus(),
        }
        self.act = acts[act]

        layers = []
        d = in_dim
        for i in range(num_layers - 1):
            d_next = hidden_dim if i < num_layers - 2 else out_dim
            layers.append(MonotoneLinear(d, d_next, bias=True))
            if i < num_layers - 2:
                layers.append(self.act)
            d = d_next
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def split_rggb(raw):
    # raw: (B, H, W)
    return torch.stack([
        raw[:, 0::2, 0::2],  # R
        raw[:, 0::2, 1::2],  # G1
        raw[:, 1::2, 0::2],  # G2
        raw[:, 1::2, 1::2],  # B
    ], dim=1)  # (B, 4, H/2, W/2)

def mad_variance(x, eps=1e-8):
    """
    x: (B, H, W)
    returns: (B,) robust variance estimate using MAD
    """
    med = x.median(dim=-1).values.median(dim=-1).values
    mad = (x - med[:, None, None]).abs()
    mad = mad.median(dim=-1).values.median(dim=-1).values
    sigma = 1.4826 * mad
    return sigma * sigma + eps

class ResidualUncertaintyEncoder(nn.Module):
    """
    Residual-based uncertainty encoder with MAD.
    Provably monotone w.r.t. distributional residual scale.
    """
    def __init__(self, emb_dim, hidden_dim=128):
        super().__init__()
        # stats per plane:
        # mean(|r|), mean(r^2), MAD(r), max(|r|)
        # -> 4 stats × 4 planes = 16 dims
        in_dim = 16

        self.mlp_emb = MonotoneMLP(
            in_dim, emb_dim, hidden_dim=hidden_dim, num_layers=3, act="softplus"
        )
        self.mlp_u = MonotoneMLP(
            in_dim, 1, hidden_dim=hidden_dim, num_layers=3, act="softplus"
        )

        self.gain = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    @torch.no_grad()
    def _pool_stats(self, r_raw):
        # r_raw: (B, H, W)
        r4 = split_rggb(r_raw)   # (B, 4, h, w)

        absr = r4.abs()
        r2 = r4 * r4

        mean_abs = absr.mean(dim=(2, 3))     # (B,4)
        mean_r2  = r2.mean(dim=(2, 3))       # (B,4)
        max_abs  = absr.amax(dim=(2, 3))     # (B,4)

        # MAD variance per plane
        mad_vars = []
        for c in range(4):
            mad_vars.append(mad_variance(r4[:, c]))
        mad_vars = torch.stack(mad_vars, dim=1)  # (B,4)

        feats = torch.cat(
            [mean_abs, mean_r2, mad_vars, max_abs], dim=1
        )  # (B,16), all >= 0

        return feats

    def forward(self, r_raw):
        feats = self._pool_stats(r_raw).float()
        emb = self.gain * self.mlp_emb(feats)
        u = self.mlp_u(feats)
        return emb, u