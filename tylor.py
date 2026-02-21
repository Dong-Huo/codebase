import torch
from typing import Tuple, Optional, Literal


# -----------------------------
# ISP forward + Jacobian pieces
# -----------------------------

def isp_forward_wb_ccm_gamma_tone_modified(
    x_lin: torch.Tensor,
    wb: torch.Tensor,
    ccm: torch.Tensor,
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ISP (matches your current clamp rules):
      WB:    u = diag(wb) @ x
      CCM:   v = ccm @ u
      Gamma: w = clamp_min(v, gamma_min)^(1/gamma)      (NO clamp_max)
      Tone:  s_raw = 3*w^2 - 2*w^3                       (NO clamp on input)
      Clamp: s = clamp(s_raw, 0, 1)                      (clamp OUTPUT only)

    Returns: (s, v, w, s_raw)
      s:     (..., 3, H, W) clamped output
      v:     (..., 3, H, W) pre-gamma
      w:     (..., 3, H, W) post-gamma
      s_raw: (..., 3, H, W) pre-clamp tone output
    """
    wb_ = wb[..., :, None, None]
    u = x_lin * wb_
    v = torch.einsum("...ij,...jhw->...ihw", ccm, u)

    alpha = 1.0 / gamma
    w = v.clamp_min(gamma_min).pow(alpha)

    s_raw = 3.0 * w * w - 2.0 * w * w * w
    s = s_raw.clamp(0.0, 1.0)
    return s, v, w, s_raw


def isp_jacobian_factor_k_and_A(
    x_lin: torch.Tensor,
    wb: torch.Tensor,
    ccm: torch.Tensor,
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes:
      - s(x) (clamped)
      - k per channel: k_c = t'(w_c) * g'(v_c) with output-clamp mask
      - A = M * G  (3x3) where columns scaled by wb (camera meta)

    Such that local Jacobian: J = diag(k) * A.

    Returns:
      s: (...,3,H,W)
      k: (...,3,H,W)
      A: (...,3,3)
    """
    s, v, w, s_raw = isp_forward_wb_ccm_gamma_tone_modified(
        x_lin, wb, ccm, gamma=gamma, gamma_min=gamma_min
    )

    alpha = 1.0 / gamma

    # g'(v): derivative of clamp_min(v,gamma_min)^(alpha)
    # zero when v < gamma_min
    mg = (v >= gamma_min).to(x_lin.dtype)
    v_safe = v.clamp_min(gamma_min)
    gprime = (alpha * v_safe.pow(alpha - 1.0)) * mg

    # t'(w): derivative of 3w^2 - 2w^3, then masked by output clamp
    mt = ((s_raw > 0.0) & (s_raw < 1.0)).to(x_lin.dtype)
    tprime = (6.0 * w - 6.0 * w * w) * mt

    k = tprime * gprime

    # A = CCM * diag(wb)  (i.e., scale CCM columns by wb)
    A = ccm * wb[..., None, :]

    return s, k, A


def analytic_gate_from_k(
    k: torch.Tensor,
    tau: float = 0.02,
    eps: float = 1e-6,
    clamp_min: Optional[float] = None,
) -> torch.Tensor:
    """
    Soft gate g in [0,1] from k (per pixel).
    k: (...,3,H,W) -> gate (...,1,H,W)
    """
    k_norm = k.abs().sum(dim=-3, keepdim=True)  # L1 over channels
    g = k_norm / (k_norm + tau + eps)
    if clamp_min is not None:
        g = g.clamp(min=clamp_min)
    return g


# -----------------------------
# Option 1: Gauss-Newton ISP refinement
# -----------------------------

@torch.no_grad()
def _trust_region_clip(dx: torch.Tensor, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    dx: (...,3,H,W), r: (...,1,H,W)
    """
    n = torch.linalg.vector_norm(dx, ord=2, dim=-3, keepdim=True)
    scale = torch.minimum(torch.ones_like(n), r / (n + eps))
    return dx * scale


def refine_linear_with_isp_gauss_newton(
    x_init: torch.Tensor,               # (B,3,H,W) linear RGB
    s_star: torch.Tensor,               # (B,3,H,W) diffusion output in sRGB domain (clamped-like)
    wb: torch.Tensor,                   # (B,3)
    ccm: torch.Tensor,                  # (B,3,3)
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
    num_iters: int = 2,
    beta: float = 1e-3,                 # LM damping (bigger => smaller updates)
    gate_tau: float = 0.02,
    gate_clamp_min: Optional[float] = None,
    apply_gate_to_residual: bool = True,
    trust_region: Literal["none", "k", "fixed"] = "k",
    r_min: float = 0.05,
    r_max: float = 0.35,
    tau_knorm: float = 3.0,             # on k_norm (L1) scale
    r_fixed: float = 0.25,
    max_delta_per_iter: Optional[float] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Performs a few Gauss-Newton (Levenberg–Marquardt) iterations to refine linear RGB x
    by minimizing (optionally gated):
        || g * (f_ISP(x) - s_star) ||^2 + beta ||dx||^2

    Update:
        dx = (J^T J + beta I)^-1 J^T (g * (s_star - s(x)))

    with J = diag(k) * A, computed at current x.

    Returns:
      x:   refined linear RGB (B,3,H,W)
      info: dict with debug maps (k, gate, last_dx_norm, s_last)
    """
    assert x_init.shape == s_star.shape and x_init.shape[1] == 3
    x = x_init.clone()

    info = {}
    for _ in range(num_iters):
        # forward + jacobian factors at current x
        s, k, A = isp_jacobian_factor_k_and_A(
            x, wb, ccm, gamma=gamma, gamma_min=gamma_min, eps=eps
        )

        # residual in sRGB space
        r_s = (s_star - s)

        # gate from k (optional)
        gate = analytic_gate_from_k(k, tau=gate_tau, eps=eps, clamp_min=gate_clamp_min)
        if apply_gate_to_residual:
            r_s = gate * r_s

        # Build per-pixel linear system:
        # dx = (A^T D^2 A + beta I)^-1 A^T (D r_s)
        # where D=diag(k)
        Dr = k * r_s  # (B,3,H,W)

        # rhs: A^T (D r_s)
        rhs = torch.einsum("bij,bjhw->bihw", A.transpose(-1, -2), Dr)  # (B,3,H,W)

        # mat: A^T D^2 A
        w2 = k * k  # (B,3,H,W)
        D2A = A[:, :, :, None, None] * w2[:, :, None, :, :]           # (B,3,3,H,W)
        Bmat = torch.einsum("bim, bmnhw->binhw", A, D2A)              # (B,3,3,H,W)

        # reshape to (B,H,W,3,3) and (B,H,W,3)
        B_hw = Bmat.permute(0, 3, 4, 1, 2).contiguous()
        rhs_hw = rhs.permute(0, 2, 3, 1).contiguous()

        I = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 1, 1, 3, 3)
        M = B_hw + beta * I

        dx_hw = torch.linalg.solve(M, rhs_hw[..., None]).squeeze(-1)  # (B,H,W,3)
        dx = dx_hw.permute(0, 3, 1, 2).contiguous()                   # (B,3,H,W)

        if max_delta_per_iter is not None:
            dx = dx.clamp(min=-max_delta_per_iter, max=max_delta_per_iter)

        # trust region clip (optional)
        if trust_region != "none":
            if trust_region == "fixed":
                r = torch.full_like(gate, r_fixed)
            elif trust_region == "k":
                k_norm = k.abs().sum(dim=1, keepdim=True)  # (B,1,H,W)
                r = r_min + (r_max - r_min) * (k_norm / (k_norm + tau_knorm + eps))
            else:
                raise ValueError(f"Unknown trust_region={trust_region}")

            dx = _trust_region_clip(dx, r, eps=eps)

        # update x
        x = x + dx

        # store debug (last iter)
        info = {
            "s": s,
            "k": k,
            "gate": gate,
            "dx": dx,
            "dx_norm": torch.linalg.vector_norm(dx, ord=2, dim=1, keepdim=True),
        }

    return x, info


# -----------------------------
# Example usage
# -----------------------------
if __name__ == "__main__":
    B, H, W = 2, 512, 512
    x_init = torch.rand(B, 3, H, W, device="cpu")
    s_star = torch.rand(B, 3, H, W, device="cpu")  # diffusion output in sRGB domain (assumed clamped-ish)

    wb = torch.tensor([[2.0, 1.0, 1.5],
                       [1.8, 1.0, 1.2]], dtype=x_init.dtype)
    ccm = torch.eye(3, dtype=x_init.dtype).unsqueeze(0).repeat(B, 1, 1)

    x_ref, dbg = refine_linear_with_isp_gauss_newton(
        x_init=x_init,
        s_star=s_star,
        wb=wb,
        ccm=ccm,
        num_iters=2,
        beta=1e-3,
        gate_tau=0.02,
        trust_region="k",
        r_min=0.05,
        r_max=0.35,
        tau_knorm=3.0,
        max_delta_per_iter=0.2,
    )
    print(x_ref.shape, dbg["dx_norm"].min().item(), dbg["dx_norm"].max().item())