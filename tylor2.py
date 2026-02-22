import torch
from typing import Tuple, Optional, Literal


# ============================================================
# ISP with WB clip-to-[0,1] (as you observed)
#   u = clamp(wb * x, 0, 1)
#   v = ccm @ u
#   w = clamp_min(v, gamma_min)^(1/gamma)
#   s_raw = 3 w^2 - 2 w^3
#   s = clamp(s_raw, 0, 1)
# ============================================================

def isp_forward_wbclip_ccm_gamma_tone(
    x_lin: torch.Tensor,           # (B,3,H,W)
    wb: torch.Tensor,              # (B,3)
    ccm: torch.Tensor,             # (B,3,3)
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      s:     (B,3,H,W) clamped output after tone
      s_raw: (B,3,H,W) unclamped tone output
      v:     (B,3,H,W) pre-gamma
      w:     (B,3,H,W) post-gamma
      u:     (B,3,H,W) after WB + clip
      u_pre: (B,3,H,W) before WB clip
    """
    wb_ = wb[..., :, None, None]
    u_pre = x_lin * wb_
    u = u_pre.clamp(0.0, 1.0)

    v = torch.einsum("bij,bjhw->bihw", ccm, u)

    alpha = 1.0 / gamma
    w = v.clamp_min(gamma_min).pow(alpha)

    s_raw = 3.0 * w * w - 2.0 * w * w * w
    s = s_raw.clamp(0.0, 1.0)

    return s, s_raw, v, w, u, u_pre


def isp_jacobian_factor_k_and_Aeff_wbclip(
    x_lin: torch.Tensor,           # (B,3,H,W)
    wb: torch.Tensor,              # (B,3)
    ccm: torch.Tensor,             # (B,3,3)
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes local Jacobian in the same factor form, but with WB clipping:

      u = clip(wb*x, 0, 1)
      du/dx = wb * 1(0 < wb*x < 1)

      J(p) = diag(k(p)) * CCM * diag(wb_eff(p))
      where wb_eff(p) = wb * m_wb(p)

      k(p) = t'(w(p)) * g'(v(p)) * 1(0<s_raw(p)<1)
      g'(v) = 0                      if v < gamma_min
            = alpha * v^(alpha-1)    otherwise

    Returns:
      s:      (B,3,H,W) clamped ISP output at x
      k:      (B,3,H,W) per-channel slope from tone+gamma+output-clamp
      Aeff:   (B,H,W,3,3) per-pixel effective A = CCM * diag(wb_eff(p))
      wb_eff: (B,3,H,W) per-channel effective WB derivative term
      gate:   (B,1,H,W) soft gate derived from k (useful for weighting)
    """
    s, s_raw, v, w, u, u_pre = isp_forward_wbclip_ccm_gamma_tone(
        x_lin, wb, ccm, gamma=gamma, gamma_min=gamma_min
    )

    alpha = 1.0 / gamma

    # WB clip derivative mask: 1 if inside (0,1), else 0
    m_wb = ((u_pre > 0.0) & (u_pre < 1.0)).to(x_lin.dtype)  # (B,3,H,W)
    wb_ = wb[..., :, None, None]
    wb_eff = wb_ * m_wb  # (B,3,H,W)

    # gamma derivative (clamp_min only)
    mg = (v >= gamma_min).to(x_lin.dtype)
    v_safe = v.clamp_min(gamma_min)
    gprime = (alpha * v_safe.pow(alpha - 1.0)) * mg  # (B,3,H,W)

    # tone derivative masked by output clamp on s_raw
    mt = ((s_raw > 0.0) & (s_raw < 1.0)).to(x_lin.dtype)
    tprime = (6.0 * w - 6.0 * w * w) * mt            # (B,3,H,W)

    k = tprime * gprime                               # (B,3,H,W)

    # Per-pixel Aeff = CCM * diag(wb_eff(p))
    # Build diag(wb_eff) in (B,H,W,3,3), multiply CCM.
    wb_eff_hw = wb_eff.permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)
    D = torch.diag_embed(wb_eff_hw)                       # (B,H,W,3,3)
    CCM_hw = ccm[:, None, None, :, :]                     # (B,1,1,3,3)
    Aeff = torch.matmul(CCM_hw, D)                        # (B,H,W,3,3)

    # Optional soft gate from k
    k_norm = k.abs().sum(dim=1, keepdim=True)             # (B,1,H,W)
    gate_tau = 0.02
    gate = k_norm / (k_norm + gate_tau + eps)

    return s, k, Aeff, wb_eff, gate


# ============================================================
# Gauss–Newton ISP refinement (Option 1) with WB clipping
# ============================================================

@torch.no_grad()
def _trust_region_clip(dx: torch.Tensor, r: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    n = torch.linalg.vector_norm(dx, ord=2, dim=1, keepdim=True)  # (B,1,H,W)
    scale = torch.minimum(torch.ones_like(n), r / (n + eps))
    return dx * scale


def refine_linear_with_isp_gauss_newton_wbclip(
    x_init: torch.Tensor,               # (B,3,H,W) linear RGB
    s_star: torch.Tensor,               # (B,3,H,W) target sRGB (diffusion output)
    wb: torch.Tensor,                   # (B,3)
    ccm: torch.Tensor,                  # (B,3,3)
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
    num_iters: int = 2,
    beta: float = 1e-3,                 # LM damping
    apply_gate_to_residual: bool = True,
    trust_region: Literal["none", "k", "fixed"] = "k",
    r_min: float = 0.05,
    r_max: float = 0.35,
    tau_knorm: float = 3.0,             # on k_norm (L1) scale
    r_fixed: float = 0.25,
    max_delta_per_iter: Optional[float] = 0.2,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, dict]:
    """
    Minimize (optionally gated):
        || g * (f_ISP(x) - s_star) ||^2 + beta ||dx||^2
    with Gauss–Newton / LM updates:
        dx = (J^T J + beta I)^-1 J^T (g * (s_star - s(x)))

    Here, because WB is clipped, A is per-pixel:
        J(p) = diag(k(p)) * Aeff(p)
        Aeff(p) = CCM * diag(wb_eff(p))

    Returns:
      x: refined linear RGB
      info: debug maps from last iteration
    """
    assert x_init.shape == s_star.shape and x_init.shape[1] == 3
    x = x_init.clone()

    info = {}
    I3 = torch.eye(3, device=x.device, dtype=x.dtype).view(1, 1, 1, 3, 3)

    for _ in range(num_iters):
        s, k, Aeff, wb_eff, gate = isp_jacobian_factor_k_and_Aeff_wbclip(
            x, wb, ccm, gamma=gamma, gamma_min=gamma_min, eps=eps
        )

        r_s = (s_star - s)  # (B,3,H,W)
        if apply_gate_to_residual:
            r_s = gate * r_s

        # Dr = diag(k) * r_s
        Dr = k * r_s  # (B,3,H,W)

        # Convert to HW layout
        Dr_hw = Dr.permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)

        # rhs = Aeff^T * Dr
        rhs_hw = torch.matmul(Aeff.transpose(-1, -2), Dr_hw[..., None]).squeeze(-1)  # (B,H,W,3)

        # B = Aeff^T * diag(k^2) * Aeff
        k2_hw = (k * k).permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)
        D2 = torch.diag_embed(k2_hw)                       # (B,H,W,3,3)
        B_hw = torch.matmul(Aeff.transpose(-1, -2), torch.matmul(D2, Aeff))  # (B,H,W,3,3)

        # Solve (B + beta I) dx = rhs
        M = B_hw + beta * I3
        dx_hw = torch.linalg.solve(M, rhs_hw[..., None]).squeeze(-1)  # (B,H,W,3)
        dx = dx_hw.permute(0, 3, 1, 2).contiguous()                   # (B,3,H,W)

        if max_delta_per_iter is not None:
            dx = dx.clamp(min=-max_delta_per_iter, max=max_delta_per_iter)

        # Trust region
        if trust_region != "none":
            if trust_region == "fixed":
                r = torch.full_like(gate, r_fixed)
            elif trust_region == "k":
                k_norm = k.abs().sum(dim=1, keepdim=True)  # (B,1,H,W)
                r = r_min + (r_max - r_min) * (k_norm / (k_norm + tau_knorm + eps))
            else:
                raise ValueError(f"Unknown trust_region={trust_region}")
            dx = _trust_region_clip(dx, r, eps=eps)

        x = x + dx

        info = {
            "s": s,
            "k": k,
            "wb_eff": wb_eff,
            "gate": gate,
            "dx": dx,
            "dx_norm": torch.linalg.vector_norm(dx, ord=2, dim=1, keepdim=True),
        }

    return x, info


# ============================================================
# Example
# ============================================================
if __name__ == "__main__":
    B, H, W = 2, 512, 512
    x_init = torch.rand(B, 3, H, W)
    s_star = torch.rand(B, 3, H, W)  # diffusion output in the same sRGB domain as ISP output

    wb = torch.tensor([[2.0, 1.0, 1.5],
                       [1.8, 1.0, 1.2]], dtype=x_init.dtype)
    ccm = torch.eye(3, dtype=x_init.dtype).unsqueeze(0).repeat(B, 1, 1)

    x_ref, dbg = refine_linear_with_isp_gauss_newton_wbclip(
        x_init=x_init,
        s_star=s_star,
        wb=wb,
        ccm=ccm,
        num_iters=2,
        beta=1e-3,
        apply_gate_to_residual=True,
        trust_region="k",
        r_min=0.05,
        r_max=0.35,
        tau_knorm=3.0,
        max_delta_per_iter=0.2,
    )

    print("x_ref:", x_ref.shape, "dx_norm min/max:", dbg["dx_norm"].min().item(), dbg["dx_norm"].max().item())