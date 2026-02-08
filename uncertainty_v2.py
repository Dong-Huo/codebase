import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers ----------
def split_rggb(raw_bhw: torch.Tensor) -> torch.Tensor:
    """
    raw_bhw: (B,H,W) mosaiced Bayer RGGB
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
def mad_sigma(x_bhw: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x_bhw: (B,H,W)
    returns: (B,) robust sigma using MAD (Gaussian-consistent)
    """
    # median over H then W (works on PyTorch without needing flatten)
    med = x_bhw.median(dim=-1).values.median(dim=-1).values  # (B,)
    dev = (x_bhw - med[:, None, None]).abs()
    mad = dev.median(dim=-1).values.median(dim=-1).values    # (B,)
    return 1.4826 * mad + eps

# ---------- residual uncertainty encoder ----------
class ResidualUncertaintyEncoder(nn.Module):
    """
    Produces:
      - e_vec: (B, in_channels) vector to be added into time_proj space
      - u:     (B, 1) scalar uncertainty (nonnegative), used to build a monotone gate g(u)

    The feature extraction uses nonnegative pooled stats (monotone in residual scale):
      per plane: mean(|r|), mean(r^2), MAD_sigma(r), max(|r|)
      => 4 stats x 4 planes = 16 dims
    """
    def __init__(self, in_channels: int, hidden_dim: int = 128, init_vec_scale: float = 0.1):
        super().__init__()
        self.in_channels = in_channels

        feat_dim = 16
        self.mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, in_channels),
        )

        # small initial scale so this doesn't overpower time_proj early
        self.vec_gain = nn.Parameter(torch.tensor(init_vec_scale, dtype=torch.float32))

        # optional: init last layer small
        last = self.mlp[-1]
        nn.init.zeros_(last.bias)
        nn.init.normal_(last.weight, mean=0.0, std=1e-4)

        # a separate head to produce scalar uncertainty u (>=0)
        self.u_head = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # ensures u >= 0
        )

    @torch.no_grad()
    def _features(self, r_raw: torch.Tensor) -> torch.Tensor:
        """
        r_raw: (B,H,W) residual in the SAME normalized raw domain.
        returns: (B,16) nonnegative feature vector
        """
        r4 = split_rggb(r_raw)                      # (B,4,h,w)
        absr = r4.abs()
        r2 = r4 * r4

        mean_abs = absr.mean(dim=(2, 3))            # (B,4)
        mean_r2  = r2.mean(dim=(2, 3))              # (B,4)
        max_abs  = absr.amax(dim=(2, 3))            # (B,4)

        # MAD sigma per plane
        mad_sig = []
        for c in range(4):
            mad_sig.append(mad_sigma(r4[:, c]))     # (B,)
        mad_sig = torch.stack(mad_sig, dim=1)       # (B,4)

        feats = torch.cat([mean_abs, mean_r2, mad_sig, max_abs], dim=1)  # (B,16), all >= 0
        return feats

    def forward(self, r_raw: torch.Tensor):
        feats = self._features(r_raw).float()
        e_vec = self.vec_gain * self.mlp(feats)     # (B,in_channels)
        u = self.u_head(feats)                      # (B,1), >=0
        return e_vec, u

# ---------- monotone gate ----------
class MonotoneGate(nn.Module):
    """
    g(u) = sigmoid( s * u + b ), with s >= 0 enforced by softplus(k_raw)

    - u is (B,1) nonnegative uncertainty
    - s controls sensitivity: how fast g changes with u
    - b controls baseline: g(u=0)

    If you want higher uncertainty -> *less* injection, flip sign by using negative=True.
    """
    def __init__(self, init_s: float = 1.0, init_b: float = 0.0, negative: bool = True):
        super().__init__()
        # k_raw is an unconstrained parameter; we pass it through softplus so slope s is guaranteed >= 0.
        # This avoids accidentally learning a negative slope that would invert the intended ordering.
        self.k_raw = nn.Parameter(torch.tensor(float(init_s)))
        self.b = nn.Parameter(torch.tensor(float(init_b)))
        self.negative = negative

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        u: (B,1) >=0
        returns: g(u) in (0,1)
        """
        s = F.softplus(self.k_raw)  # s >= 0
        x = s * u + self.b
        if self.negative:
            x = -x
        return torch.sigmoid(x)     # (B,1)

# ---------- patched timestep embedding ----------
class TimestepEmbeddingWithResidual(nn.Module):
    """
    Wraps your TimestepEmbedding:
      time_emb = TimeMLP( time_proj + g(u) * e_vec )
    """
    def __init__(self, base_timestep_embedding: nn.Module, in_channels: int, gate_negative: bool = True):
        super().__init__()
        self.base = base_timestep_embedding
        self.res_enc = ResidualUncertaintyEncoder(in_channels=in_channels)
        self.gate = MonotoneGate(negative=gate_negative)

    def forward(self, time_proj: torch.Tensor, residual_raw: torch.Tensor):
        """
        time_proj: (B, in_channels) output of get_timestep_embedding(...)
        residual_raw: (B,H,W) = noisy_raw - clean_raw_proxy (same domain)
        """
        e_vec, u = self.res_enc(residual_raw)       # (B,in_channels), (B,1)
        g = self.gate(u)                             # (B,1) in (0,1)
        cond = g * e_vec                             # broadcast (B,1)*(B,C)
        return self.base(time_proj, condition=cond)  # uses base.cond_proj if present, or can set cond_proj=None and add directly