import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Bayer helpers (RGGB)
# -------------------------
def split_rggb_4ch(raw_bhw: torch.Tensor) -> torch.Tensor:
    """
    raw_bhw: (B,H,W) mosaiced Bayer (RGGB)
    returns: (B,4,H/2,W/2) packed planes [R,G1,G2,B]
    """
    return torch.stack(
        [
            raw_bhw[:, 0::2, 0::2],  # R
            raw_bhw[:, 0::2, 1::2],  # G1
            raw_bhw[:, 1::2, 0::2],  # G2
            raw_bhw[:, 1::2, 1::2],  # B
        ],
        dim=1,
    )

@torch.no_grad()
def mad_sigma_bhw(x_bhw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Robust sigma via MAD for (B,H,W). Returns (B,).
    """
    med = x_bhw.median(dim=-1).values.median(dim=-1).values  # (B,)
    dev = (x_bhw - med[:, None, None]).abs()
    mad = dev.median(dim=-1).values.median(dim=-1).values    # (B,)
    return 1.4826 * mad + eps


# -------------------------
# Monotone gate g(u)
# -------------------------
class MonotoneGate(nn.Module):
    """
    g(u) = sigmoid( sign * (softplus(k_raw) * u + b) ), u>=0
      - if negative=True, sign = -1 => higher u -> smaller g (more uncertainty -> less injection)
      - if negative=False, sign = +1 => higher u -> larger g
    """
    def __init__(self, init_k: float = 1.0, init_b: float = -2.0, negative: bool = True):
        super().__init__()
        self.k_raw = nn.Parameter(torch.tensor(float(init_k)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.negative = negative

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        # u: (B,1) >= 0
        s = F.softplus(self.k_raw)  # slope >= 0
        x = s * u + self.b
        if self.negative:
            x = -x
        return torch.sigmoid(x)     # (B,1) in (0,1)


# -------------------------
# Stats head for u (stable + order-preserving proxy)
# -------------------------
class ResidualStatsU(nn.Module):
    """
    Computes a scalar u from robust, nonnegative residual stats.
    You can keep this simple; u then drives the monotone gate.

    Features (per plane, 4 planes):
      mean(|r|), mean(r^2), MAD_sigma(r)  -> 3*4=12 dims
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        feat_dim = 12
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensure u >= 0
        )

    @torch.no_grad()
    def _feat(self, r_raw: torch.Tensor) -> torch.Tensor:
        r4 = split_rggb_4ch(r_raw)                 # (B,4,h,w)
        absr = r4.abs()
        r2 = r4 * r4

        mean_abs = absr.mean(dim=(2, 3))           # (B,4)
        mean_r2 = r2.mean(dim=(2, 3))              # (B,4)

        mad_sig = []
        for c in range(4):
            mad_sig.append(mad_sigma_bhw(r4[:, c]))  # (B,)
        mad_sig = torch.stack(mad_sig, dim=1)      # (B,4)

        feats = torch.cat([mean_abs, mean_r2, mad_sig], dim=1)  # (B,12) >=0
        return feats

    def forward(self, r_raw: torch.Tensor) -> torch.Tensor:
        feats = self._feat(r_raw).float()
        u = self.net(feats)  # (B,1) >= 0
        return u


# -------------------------
# CNN head for e_vec (expressive)
# -------------------------
class ResidualCNNEvec(nn.Module):
    """
    Encodes residual raw into e_vec in timestep projection space (B,in_channels).
    Input: residual raw (B,H,W) -> pack RGGB -> CNN -> global pool -> linear.
    """
    def __init__(self, in_channels: int, base_ch: int = 32, init_vec_scale: float = 0.1):
        super().__init__()
        self.in_channels = in_channels

        def block(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, stride=stride, padding=1),
                nn.GroupNorm(num_groups=8, num_channels=cout),
                nn.SiLU(),
            )

        self.conv = nn.Sequential(
            block(4, base_ch, stride=2),           # /2
            block(base_ch, base_ch, stride=1),
            block(base_ch, base_ch * 2, stride=2), # /4
            block(base_ch * 2, base_ch * 2, stride=1),
            block(base_ch * 2, base_ch * 4, stride=2), # /8
            block(base_ch * 4, base_ch * 4, stride=1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

        feat_dim = base_ch * 4
        self.head = nn.Linear(feat_dim, in_channels)

        # small gain so e_vec doesn't overpower time_proj early
        self.vec_gain = nn.Parameter(torch.tensor(init_vec_scale, dtype=torch.float32))

        nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.head.weight, mean=0.0, std=1e-4)

    def forward(self, r_raw: torch.Tensor) -> torch.Tensor:
        x = split_rggb_4ch(r_raw).float()       # (B,4,H/2,W/2)
        h = self.conv(x)                        # (B,C,h,w)
        h = self.pool(h).flatten(1)             # (B,C)
        e_vec = self.vec_gain * self.head(h)    # (B,in_channels)
        return e_vec


# -------------------------
# Combined encoder: stats->u, cnn->e_vec
# -------------------------
class ResidualConditionEncoder(nn.Module):
    """
    Returns:
      e_vec: (B,in_channels)
      u:     (B,1) >=0
    """
    def __init__(self, in_channels: int, u_hidden: int = 64, base_ch: int = 32):
        super().__init__()
        self.u_head = ResidualStatsU(hidden_dim=u_hidden)
        self.e_head = ResidualCNNEvec(in_channels=in_channels, base_ch=base_ch)

    def forward(self, r_raw: torch.Tensor):
        u = self.u_head(r_raw)       # (B,1)
        e_vec = self.e_head(r_raw)   # (B,in_channels)
        return e_vec, u


# -------------------------
# Patch/wrapper for your TimestepEmbedding
# -------------------------
class TimestepEmbeddingResidualWrapper(nn.Module):
    """
    Implements:
      time_emb = base( time_proj + g(u) * e_vec )

    This does NOT use base.cond_proj at all; it injects directly into `sample` space (in_channels).
    So set cond_proj_dim=None in your base TimestepEmbedding (or just ignore its condition argument).
    """
    def __init__(self, base_timestep_embedding: nn.Module, in_channels: int,
                 gate_negative: bool = True, gate_init_k: float = 1.0, gate_init_b: float = -2.0,
                 u_hidden: int = 64, base_ch: int = 32):
        super().__init__()
        self.base = base_timestep_embedding
        self.enc = ResidualConditionEncoder(in_channels=in_channels, u_hidden=u_hidden, base_ch=base_ch)
        self.gate = MonotoneGate(init_k=gate_init_k, init_b=gate_init_b, negative=gate_negative)

    def forward(self, time_proj: torch.Tensor, residual_raw: torch.Tensor):
        """
        time_proj: (B,in_channels) output of get_timestep_embedding(t, in_channels)
        residual_raw: (B,H,W) = noisy_raw - clean_raw_proxy (same normalized raw domain)
        """
        e_vec, u = self.enc(residual_raw)  # (B,in_channels), (B,1)
        g = self.gate(u)                   # (B,1)
        time_proj2 = time_proj + g * e_vec # broadcast (B,1)*(B,C)

        # Call base without condition (we already injected)
        return self.base(time_proj2, condition=None), {"u": u, "g": g}


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    import math
    from typing import Optional

    # your get_timestep_embedding function (as provided)
    def get_timestep_embedding(timesteps, embedding_dim, max_period: int = 10000):
        assert len(timesteps.shape) == 1
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(
            start=0, end=half_dim, dtype=torch.float32, device=timesteps.device
        )
        exponent = exponent / (half_dim - 1)
        emb = torch.exp(exponent)
        emb = timesteps[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if embedding_dim % 2 == 1:
            emb = F.pad(emb, (0, 1, 0, 0))
        return emb

    # base module: in_channels must match time_proj dim
    class TimestepEmbedding(nn.Module):
        def __init__(self, in_channels, time_embed_dim, act_fn="silu", out_dim=None, cond_proj_dim=None, sample_proj_bias=True):
            super().__init__()
            self.linear_1 = nn.Linear(in_channels, time_embed_dim, sample_proj_bias)
            self.cond_proj = None
            self.act = nn.SiLU()
            time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
            self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out, sample_proj_bias)
            self.post_act = None

        def forward(self, sample, condition=None):
            if condition is not None:
                sample = sample + condition
            sample = self.linear_1(sample)
            sample = self.act(sample)
            sample = self.linear_2(sample)
            return sample

    B, H, W = 2, 512, 512
    in_channels = 320
    time_embed_dim = 1024

    base = TimestepEmbedding(in_channels=in_channels, time_embed_dim=time_embed_dim)
    wrapper = TimestepEmbeddingResidualWrapper(
        base_timestep_embedding=base,
        in_channels=in_channels,
        gate_negative=True,    # higher u => smaller g (more uncertainty => less injection)
        gate_init_k=1.0,
        gate_init_b=-2.0,
        u_hidden=64,
        base_ch=32,
    )

    t = torch.randint(low=0, high=1000, size=(B,))
    time_proj = get_timestep_embedding(t, in_channels)         # (B,in_channels)
    residual_raw = torch.randn(B, H, W) * 0.01                 # toy residual

    time_emb, aux = wrapper(time_proj, residual_raw)
    print(time_emb.shape, aux["u"].shape, aux["g"].shape)