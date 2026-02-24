import torch
from typing import Tuple, Optional, Dict, Literal


# ============================================================
# ISP + Jacobian builder (WITH WB clip after WB)
#   u = clamp(wb*x, 0, 1)
#   v = ccm @ u
#   w = clamp_min(v, gamma_min)^(1/gamma)
#   s_raw = 3 w^2 - 2 w^3
#   s = clamp(s_raw, 0, 1)
# ============================================================

def isp_forward_wbclip_ccm_gamma_tone(
    x_lin: torch.Tensor,   # (B,3,H,W)
    wb: torch.Tensor,      # (B,3)
    ccm: torch.Tensor,     # (B,3,3)
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wb_ = wb[..., :, None, None]
    u_pre = x_lin * wb_
    u = u_pre.clamp(0.0, 1.0)

    v = torch.einsum("bij,bjhw->bihw", ccm, u)

    alpha = 1.0 / gamma
    w = v.clamp_min(gamma_min).pow(alpha)

    s_raw = 3.0 * w * w - 2.0 * w * w * w
    s = s_raw.clamp(0.0, 1.0)
    return s, s_raw, v, w, u, u_pre


def build_J_wbclip(
    x_lin: torch.Tensor,   # (B,3,H,W)
    wb: torch.Tensor,      # (B,3)
    ccm: torch.Tensor,     # (B,3,3)
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      J:      (B,H,W,3,3)
      k:      (B,3,H,W)
      wb_eff: (B,3,H,W)  (WB derivative term wb * 1(0<wb*x<1))
      s0:     (B,3,H,W)
    """
    s0, s_raw, v, w, u, u_pre = isp_forward_wbclip_ccm_gamma_tone(
        x_lin, wb, ccm, gamma=gamma, gamma_min=gamma_min
    )

    # WB clip derivative
    m_wb = ((u_pre > 0.0) & (u_pre < 1.0)).to(x_lin.dtype)  # (B,3,H,W)
    wb_eff = wb[..., :, None, None] * m_wb                  # (B,3,H,W)

    # gamma derivative (clamp_min only)
    alpha = 1.0 / gamma
    mg = (v >= gamma_min).to(x_lin.dtype)
    v_safe = v.clamp_min(gamma_min)
    gprime = (alpha * v_safe.pow(alpha - 1.0)) * mg

    # tone derivative, masked by output clamp on s_raw
    mt = ((s_raw > 0.0) & (s_raw < 1.0)).to(x_lin.dtype)
    tprime = (6.0 * w - 6.0 * w * w) * mt

    k = tprime * gprime  # (B,3,H,W)

    # Aeff(p) = CCM * diag(wb_eff(p))  -> (B,H,W,3,3)
    wb_eff_hw = wb_eff.permute(0, 2, 3, 1).contiguous()  # (B,H,W,3)
    Dwb = torch.diag_embed(wb_eff_hw)                    # (B,H,W,3,3)
    CCM_hw = ccm[:, None, None, :, :]                    # (B,1,1,3,3)
    Aeff = CCM_hw @ Dwb                                  # (B,H,W,3,3)

    # J(p) = diag(k(p)) * Aeff(p)
    k_hw = k.permute(0, 2, 3, 1).contiguous()            # (B,H,W,3)
    Dk = torch.diag_embed(k_hw)                          # (B,H,W,3,3)
    J = Dk @ Aeff

    return J, k, wb_eff, s0


# ============================================================
# FO solve (normal equations) at x0: dx_FO = (J^T J + lam I)^-1 J^T ds
# This mirrors your earlier FO implementations, but we form J explicitly (3x3).
# ============================================================

def fo_transport_from_J(
    J: torch.Tensor,                 # (B,H,W,3,3)
    delta_s: torch.Tensor,           # (B,3,H,W)
    lam: float = 1e-6,
) -> torch.Tensor:
    """
    Returns dx_FO: (B,3,H,W)
    """
    B, H, W = J.shape[:3]
    ds_hw = delta_s.permute(0, 2, 3, 1).contiguous()                # (B,H,W,3)
    JT = J.transpose(-1, -2)                                        # (B,H,W,3,3)
    rhs = (JT @ ds_hw[..., None]).squeeze(-1)                       # (B,H,W,3)
    M = JT @ J                                                      # (B,H,W,3,3)
    I = torch.eye(3, device=J.device, dtype=J.dtype).view(1,1,1,3,3)
    dx_hw = torch.linalg.solve(M + lam * I, rhs[..., None]).squeeze(-1)  # (B,H,W,3)
    dx = dx_hw.permute(0, 3, 1, 2).contiguous()                     # (B,3,H,W)
    return dx


# ============================================================
# One-step TSVD transport (anchored at x0): dx_TSVD = V_k * (σ/(σ^2+β)) * U^T ds
# ============================================================

def tsvd_transport_from_J(
    J: torch.Tensor,                 # (B,H,W,3,3)
    delta_s: torch.Tensor,           # (B,3,H,W)
    eps_svd: float = 1e-3,           # threshold on singular values
    beta: float = 1e-2,              # damping within retained subspace
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      dx_TSVD: (B,3,H,W)
      svals:   (B,H,W,3) singular values (descending)
      keep:    (B,H,W,3) bool mask per singular value (σ>eps_svd)
    """
    ds_hw = delta_s.permute(0, 2, 3, 1).contiguous()    # (B,H,W,3)
    U, S, Vh = torch.linalg.svd(J, full_matrices=False) # U:(B,H,W,3,3), S:(B,H,W,3), Vh:(B,H,W,3,3)
    keep = (S > eps_svd)

    # Compute y = U^T ds
    y = (U.transpose(-1, -2) @ ds_hw[..., None]).squeeze(-1)  # (B,H,W,3)

    # Per-component filter: σ/(σ^2 + beta) if kept else 0
    filt = torch.where(keep, S / (S * S + beta), torch.zeros_like(S))  # (B,H,W,3)

    # z = filt * y
    z = filt * y  # (B,H,W,3)

    # dx = V z  where V = Vh^T
    V = Vh.transpose(-1, -2)
    dx_hw = (V @ z[..., None]).squeeze(-1)  # (B,H,W,3)
    dx = dx_hw.permute(0, 3, 1, 2).contiguous()
    return dx, S, keep


# ============================================================
# Combined FO + one-step TSVD with reliability mask (anchored at x0)
#   dx = m_good * dx_FO + m_bad * alpha * dx_TSVD
# Mask can be from sigma_min(J) or from k_norm.
# ============================================================

def fo_plus_one_step_tsvd(
    x0_lin: torch.Tensor,                   # (B,3,H,W)
    s_star: torch.Tensor,                   # (B,3,H,W)
    wb: torch.Tensor,                       # (B,3)
    ccm: torch.Tensor,                      # (B,3,3)
    gamma: float = 2.2,
    gamma_min: float = 1e-8,
    # FO
    lam: float = 1e-6,
    # TSVD
    eps_svd: float = 1e-3,
    beta: float = 1e-2,
    alpha: float = 0.15,                    # trust-region scalar for TSVD step
    # Reliability mask
    mask_mode: Literal["smin", "knorm"] = "knorm",
    tau_smin: float = 1e-3,                 # ill-conditioned if σ_min(J) < tau_smin
    tau_knorm: float = 2.0,                 # ill-conditioned if ||k||_1 < tau_knorm
    # Safety
    max_dx_fo: Optional[float] = None,
    max_dx_tsvd: Optional[float] = None,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Returns:
      x_out: (B,3,H,W)
      debug: dict with maps (dx_fo, dx_tsvd, masks, singular values, etc.)
    """
    # Build J at the anchor x0 (IMPORTANT: anchored; no drift from x1)
    J0, k, wb_eff, s0 = build_J_wbclip(x0_lin, wb, ccm, gamma=gamma, gamma_min=gamma_min)
    delta_s = (s_star - s0)

    # FO update everywhere (you will mask it later)
    dx_fo = fo_transport_from_J(J0, delta_s, lam=lam)
    if max_dx_fo is not None:
        dx_fo = dx_fo.clamp(-max_dx_fo, max_dx_fo)

    # TSVD update everywhere (you will mask it later)
    dx_tsvd, S, keep = tsvd_transport_from_J(J0, delta_s, eps_svd=eps_svd, beta=beta)
    if max_dx_tsvd is not None:
        dx_tsvd = dx_tsvd.clamp(-max_dx_tsvd, max_dx_tsvd)

    # Reliability mask
    if mask_mode == "smin":
        smin = S[..., -1]                              # (B,H,W)
        m_bad = (smin < tau_smin).to(x0_lin.dtype)     # (B,H,W)
        m_bad = m_bad.unsqueeze(1)                     # (B,1,H,W)
        smin_map = smin.unsqueeze(1)
        knorm_map = k.abs().sum(dim=1, keepdim=True)
    elif mask_mode == "knorm":
        knorm_map = k.abs().sum(dim=1, keepdim=True)   # (B,1,H,W)
        m_bad = (knorm_map < tau_knorm).to(x0_lin.dtype)
        smin_map = S[..., -1].unsqueeze(1)
    else:
        raise ValueError(f"Unknown mask_mode={mask_mode}")

    m_good = 1.0 - m_bad

    # Combine (anchored): do NOT form x1 = x0 + dx_fo globally.
    dx = m_good * dx_fo + m_bad * (alpha * dx_tsvd)
    x_out = x0_lin + dx

    debug = {
        "s0": s0,
        "delta_s": delta_s,
        "J0_svals": S,                       # (B,H,W,3)
        "J0_smin": smin_map,                 # (B,1,H,W)
        "k": k,                              # (B,3,H,W)
        "k_norm": knorm_map,                 # (B,1,H,W)
        "wb_eff": wb_eff,                    # (B,3,H,W)
        "dx_fo": dx_fo,
        "dx_tsvd": dx_tsvd,
        "dx": dx,
        "mask_bad": m_bad,
        "mask_good": m_good,
        "tsvd_keep": keep,                   # (B,H,W,3) bool
    }
    return x_out, debug


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    B, H, W = 2, 256, 256
    x0 = torch.rand(B, 3, H, W, dtype=torch.float32)
    s_star = torch.rand(B, 3, H, W, dtype=torch.float32)  # diffusion output in *this* sRGB domain

    wb = torch.tensor([[2.0, 1.0, 1.5],
                       [1.8, 1.0, 1.2]], dtype=torch.float32)
    ccm = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(B, 1, 1)

    x_out, dbg = fo_plus_one_step_tsvd(
        x0_lin=x0,
        s_star=s_star,
        wb=wb,
        ccm=ccm,
        lam=1e-6,
        eps_svd=1e-3,
        beta=1e-2,
        alpha=0.15,
        mask_mode="knorm",
        tau_knorm=2.0,
        max_dx_fo=0.3,
        max_dx_tsvd=0.3,
    )

    print("x_out:", x_out.shape,
          "bad_frac:", float(dbg["mask_bad"].mean()),
          "smin_med:", float(torch.median(dbg["J0_smin"])))