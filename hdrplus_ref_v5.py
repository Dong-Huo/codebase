# hdrplus_full.py
# Reference HDR+ pipeline in PyTorch:
# - Hierarchical vectorized alignment (Eq. 2–4) with 3-candidate upsampling and subpixel quadratic fit
# - Windowed, frequency-domain temporal merge (Eq. 6–7) with noise model σ² = A*x + B
# - Simple demosaic + WB/CCM + tone/gamma to sRGB

import math
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn.functional as F

Tensor = torch.Tensor


# ============================================================
# Bayer helpers (RGGB by default)
# ============================================================

def split_bayer_planes(raw: Tensor, pattern: str = "RGGB") -> Dict[str, Tensor]:
    """Split RGGB-like mosaic into 4 half-res planes."""
    assert raw.dim() == 2
    H, W = raw.shape
    patt = {
        "RGGB": ((0, 0, 'R'), (0, 1, 'G1'), (1, 0, 'G2'), (1, 1, 'B')),
        "BGGR": ((0, 0, 'B'), (0, 1, 'G1'), (1, 0, 'G2'), (1, 1, 'R')),
        "GRBG": ((0, 0, 'G1'), (0, 1, 'R'), (1, 0, 'B'), (1, 1, 'G2')),
        "GBRG": ((0, 0, 'G1'), (0, 1, 'B'), (1, 0, 'R'), (1, 1, 'G2')),
    }[pattern]
    planes = {}
    for i, j, name in patt:
        planes[name] = raw[i::2, j::2].contiguous()
    return planes


def merge_bayer_planes(planes: Dict[str, Tensor], pattern: str = "RGGB") -> Tensor:
    """Re-mosaic 4 planes back to RAW mosaic."""
    h, w = planes['R'].shape
    out = torch.empty((2 * h, 2 * w), dtype=planes['R'].dtype, device=planes['R'].device)
    if pattern == "RGGB":
        out[0::2, 0::2] = planes['R'];
        out[0::2, 1::2] = planes['G1']
        out[1::2, 0::2] = planes['G2'];
        out[1::2, 1::2] = planes['B']
    elif pattern == "BGGR":
        out[0::2, 0::2] = planes['B'];
        out[0::2, 1::2] = planes['G1']
        out[1::2, 0::2] = planes['G2'];
        out[1::2, 1::2] = planes['R']
    elif pattern == "GRBG":
        out[0::2, 0::2] = planes['G1'];
        out[0::2, 1::2] = planes['R']
        out[1::2, 0::2] = planes['B'];
        out[1::2, 1::2] = planes['G2']
    elif pattern == "GBRG":
        out[0::2, 0::2] = planes['G1'];
        out[0::2, 1::2] = planes['B']
        out[1::2, 0::2] = planes['R'];
        out[1::2, 1::2] = planes['G2']
    else:
        raise ValueError("Unknown Bayer pattern")
    return out


def gray_from_bayer(raw: Tensor, pattern: str = "RGGB") -> Tensor:
    """2×2 average of the Bayer quad, used for alignment."""
    p = split_bayer_planes(raw, pattern)
    return (p['R'] + p['G1'] + p['G2'] + p['B']) * 0.25


# ============================================================
# Windows and small utils
# ============================================================

def raised_cosine_window_2d(n: int, device, dtype) -> Tensor:
    # w(x) = 0.5 - 0.5 cos(2π (x+0.5)/n)
    x = torch.arange(n, device=device, dtype=dtype)
    w1 = 0.5 - 0.5 * torch.cos(2 * math.pi * (x + 0.5) / n)
    return (w1[:, None] * w1[None, :]).contiguous()


def rms(x: Tensor) -> Tensor:
    return torch.sqrt((x * x).mean() + 1e-12)


# ============================================================
# Alignment helpers (Eq. 2–4)
# ============================================================

def _make_shifts(radius: int, device) -> Tensor:
    u = torch.arange(-radius, radius + 1, device=device)
    v = torch.arange(-radius, radius + 1, device=device)
    U, V = torch.meshgrid(u, v, indexing='ij')
    return torch.stack([U.flatten(), V.flatten()], dim=-1)  # [D,2]


def _grid_for_shifts(H, W, shifts_pix: Tensor, device, dtype) -> Tensor:
    """Build batched sampling grids (normalized) for grid_sample."""
    yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                            torch.arange(W, device=device, dtype=dtype),
                            indexing='ij')
    gx = (xx / (W - 1)) * 2 - 1
    gy = (yy / (H - 1)) * 2 - 1
    base = torch.stack([gx, gy], dim=-1)  # [H,W,2]
    du = shifts_pix[:, 1] * (2.0 / (W - 1))  # x
    dv = shifts_pix[:, 0] * (2.0 / (H - 1))  # y
    offs = torch.stack([du, dv], dim=-1)  # [D,2]
    return base[None] + offs[:, None, None, :]  # [D,H,W,2]


def _batched_quad_min(D3x3: Tensor) -> Tensor:
    """Fit 2D quadratic to 3×3 costs and return subpixel [du,dv] (Eq. 3–4)."""
    device, dtype = D3x3.device, D3x3.dtype
    coords = torch.tensor(
        [[-1, -1], [-1, 0], [-1, 1],
         [0, -1], [0, 0], [0, 1],
         [1, -1], [1, 0], [1, 1]], device=device, dtype=dtype)
    u = coords[:, 0];
    v = coords[:, 1]
    X = torch.stack([u * u, u * v, v * v, u, v, torch.ones_like(u)], dim=1)  # [9,6]
    X_pinv = torch.pinverse(X)  # [6,9]

    N = D3x3.shape[0]
    Y = D3x3.reshape(N, 9)
    theta = (X_pinv[None] @ Y.unsqueeze(-1)).squeeze(-1)  # [N,6]
    a11, a12, a22, b1, b2, _c = theta.unbind(dim=1)

    det = a11 * a22 - a12 * a12
    det = torch.where(det.abs() < 1e-12, det.sign() * 1e-12, det)
    invA11 = a22 / det
    invA12 = -a12 / det
    invA22 = a11 / det
    du = -(invA11 * b1 + invA12 * b2)
    dv = -(invA12 * b1 + invA22 * b2)
    return torch.stack([du.clamp(-1, 1), dv.clamp(-1, 1)], dim=1)  # [N,2]


# ============================================================
# Vectorized hierarchical alignment (Eq. 2–4 + 3-candidate upsampling)
# ============================================================

@torch.no_grad()
def align_burst(
        gray_burst: List[Tensor],
        ref_idx: int,
        tile_size: int = 16,
        levels: int = 4,
        coarse_search: int = 32,
        fine_search: int = 4,
) -> Tensor:
    """
    Vectorized HDR+ alignment.
    Returns disp [F,2,Ht,Wt] at the finest scale (pixels at half-res/Bayer-plane scale).
    """
    device = gray_burst[0].device
    dtype = gray_burst[0].dtype
    Fm = len(gray_burst)

    # Build Gaussian/box pyramids
    pyramids = []
    for f in range(Fm):
        lv = [gray_burst[f]]
        for _ in range(1, levels):
            lv.append(F.avg_pool2d(lv[-1][None, None], 2, 2)[0, 0])
        pyramids.append(lv)

    disp_levels: List[Optional[Tensor]] = [None] * levels

    for lev in reversed(range(levels)):
        imgs = [p[lev] for p in pyramids]
        ref = imgs[ref_idx]
        H, W = ref.shape

        stride = tile_size // 2
        pad = (tile_size - stride) // 2  # ensure edge coverage for unfold

        # Tile grid dims with padding
        Ht = math.ceil((H + 2 * pad - tile_size) / stride) + 1
        Wt = math.ceil((W + 2 * pad - tile_size) / stride) + 1
        Ntiles = Ht * Wt

        # Extract reference tiles once
        ref_unf = F.unfold(ref[None, None], kernel_size=tile_size, stride=stride, padding=pad)
        ref_tiles = ref_unf.squeeze(0).T  # [Ntiles, k^2]

        # ---- Upsample previous level with 3-candidate selection (paper) ----
        if lev < levels - 1:
            prev = disp_levels[lev + 1]  # [F,2,Ht_prev,Wt_prev]
            # map fine tiles to coarse indices
            ci = torch.div(torch.arange(Ht, device=device), 2, rounding_mode='floor')
            cj = torch.div(torch.arange(Wt, device=device), 2, rounding_mode='floor')
            CI, CJ = torch.meshgrid(ci, cj, indexing='ij')
            CIp = torch.clamp(CI + 1, max=prev.shape[2] - 1)
            CJp = torch.clamp(CJ + 1, max=prev.shape[3] - 1)

            disp_guess = torch.zeros((Fm, 2, Ht, Wt), dtype=dtype, device=device)

            for f in range(Fm):
                if f == ref_idx:  # ref stays 0
                    continue
                cand_uv = torch.stack([
                    prev[f, :, CI, CJ],  # nearest
                    prev[f, :, CIp, CJ],  # +x neighbor
                    prev[f, :, CI, CJp],  # +y neighbor
                ], dim=0) * 2.0  # scale to current level

                I = imgs[f][None, None]
                # Evaluate L1 residual for the 3 candidates (vectorized with grid_sample)
                costs = []
                for c in range(3):
                    u = cand_uv[c, 0]
                    v = cand_uv[c, 1]
                    shifts = torch.stack([u.flatten(), v.flatten()], dim=1)  # [Ntiles,2]
                    grids = _grid_for_shifts(H, W, shifts, device, dtype)  # [Ntiles,H,W,2]
                    I_b = I.expand(Ntiles, -1, -1, -1)
                    warped = F.grid_sample(I_b, grids, mode='bilinear',
                                           padding_mode='border', align_corners=True)
                    # L1 over full image is costly; compare on tile grid via unfold on warped
                    warped_unf = F.unfold(warped, kernel_size=tile_size, stride=stride,
                                          padding=pad)  # [Ntiles, k^2, Ntiles]
                    # pick diagonal tiles: tile i from warped i
                    idx = torch.arange(Ntiles, device=device)
                    tiles = warped_unf.permute(0, 2, 1)[idx, idx]  # [Ntiles, k^2]
                    l1 = (tiles - ref_tiles).abs().mean(dim=-1)  # [Ntiles]
                    costs.append(l1)
                costs = torch.stack(costs, dim=0)  # [3,Ntiles]
                best = costs.argmin(dim=0)
                disp_guess[f, 0] = cand_uv[:, 0].reshape(3, -1).gather(0, best[None]).view(Ht, Wt)
                disp_guess[f, 1] = cand_uv[:, 1].reshape(3, -1).gather(0, best[None]).view(Ht, Wt)
        else:
            disp_guess = torch.zeros((Fm, 2, Ht, Wt), dtype=dtype, device=device)

        # ---- Per-level search (Eq. 2) + subpixel fit (Eq. 3–4) ----
        disp_new = disp_guess.clone()
        if lev == 0:
            R = fine_search
            use_L1 = True
        else:
            R = coarse_search
            use_L1 = False

        shifts = _make_shifts(R, device)  # [D,2]
        grids = _grid_for_shifts(H, W, shifts, device, dtype)  # [D,H,W,2]

        ref_tiles_flat = ref_tiles  # [Ntiles, k^2]

        for f in range(Fm):
            if f == ref_idx:
                continue
            I = imgs[f][None, None]
            Irep = I.expand(len(shifts), -1, -1, -1)
            warped = F.grid_sample(Irep, grids, mode='bilinear',
                                   padding_mode='border', align_corners=True)  # [D,1,H,W]
            cand_unf = F.unfold(warped, kernel_size=tile_size, stride=stride, padding=pad)  # [D, k^2, Ntiles]
            cand_tiles = cand_unf.transpose(1, 2)  # [D, Ntiles, k^2]

            if use_L1:
                cost = (cand_tiles - ref_tiles_flat[None]).abs().mean(dim=-1)  # [D,Ntiles]
            else:
                cost = (cand_tiles - ref_tiles_flat[None]).pow(2).mean(dim=-1)  # [D,Ntiles]

            best_idx = cost.argmin(dim=0)  # [Ntiles]
            best_uv = shifts[best_idx]  # [Ntiles,2]

            if not use_L1:
                # Subpixel quad fit on 3x3 around integer min
                map_idx = torch.arange(cost.shape[0], device=device).view(2 * R + 1, 2 * R + 1)
                nu = torch.tensor([-1, 0, 1], device=device)
                nv = torch.tensor([-1, 0, 1], device=device)
                NU, NV = torch.meshgrid(nu, nv, indexing='ij')
                du0 = best_uv[:, 0]
                dv0 = best_uv[:, 1]
                iu = (du0 + R).to(torch.long)
                iv = (dv0 + R).to(torch.long)
                iu_nb = (iu[:, None, None] + NU).clamp(0, 2 * R)
                iv_nb = (iv[:, None, None] + NV).clamp(0, 2 * R)
                nb_idx = map_idx[iu_nb, iv_nb]  # [Ntiles,3,3]

                cost_T = cost.transpose(0, 1).contiguous()  # [Ntiles, D]
                nb_lin = nb_idx.view(Ntiles, -1).long()  # [Ntiles, 9]
                D3x3 = torch.gather(cost_T, 1, nb_lin)  # [Ntiles, 9]
                D3x3 = D3x3.view(Ntiles, 3, 3)  # [Ntiles, 3, 3]

                subpix = _batched_quad_min(D3x3)  # [Ntiles,2]
                du = (best_uv[:, 0] + subpix[:, 0]).view(Ht, Wt)
                dv = (best_uv[:, 1] + subpix[:, 1]).view(Ht, Wt)
            else:
                du = best_uv[:, 0].to(dtype).view(Ht, Wt)
                dv = best_uv[:, 1].to(dtype).view(Ht, Wt)

            disp_new[f, 0] = disp_guess[f, 0] + du
            disp_new[f, 1] = disp_guess[f, 1] + dv

        disp_levels[lev] = disp_new

    return disp_levels[0]  # [F,2,Ht,Wt] at finest level (alignment scale)


# ============================================================
# Merge (Eq. 6–7), vectorized with unfold/fold
# ============================================================

def _warp_burst_with_tilewise_flow(imgs: Tensor,  # [F,h,w]
                                   disp: Tensor,  # [F,2,Ht,Wt]
                                   tile_size: int) -> Tensor:
    """
    Build a piecewise-constant flow from tile displacements and warp each frame.
    Returns warped [F,h,w] aligned to the reference tiles grid.
    """
    Fm, h, w = imgs.shape
    _, _, Ht, Wt = disp.shape
    device, dtype = imgs.device, imgs.dtype

    # Upsample (nearest) tile displacements to per-pixel flow
    u = F.interpolate(disp[:, 0][:, None], size=(h, w), mode='nearest').squeeze(1)  # [F,h,w]
    v = F.interpolate(disp[:, 1][:, None], size=(h, w), mode='nearest').squeeze(1)  # [F,h,w]

    # Build base normalized grid
    yy, xx = torch.meshgrid(torch.arange(h, device=device, dtype=dtype),
                            torch.arange(w, device=device, dtype=dtype),
                            indexing='ij')
    gx = (xx / (w - 1)) * 2 - 1
    gy = (yy / (h - 1)) * 2 - 1

    # Normalize pixel shifts to [-1,1]
    du = u * (2.0 / (w - 1))
    dv = v * (2.0 / (h - 1))

    grid = torch.stack([gx[None].expand(Fm, -1, -1) + du,
                        gy[None].expand(Fm, -1, -1) + dv], dim=-1)  # [F,h,w,2]
    imgs_b = imgs[:, None]  # [F,1,h,w]
    warped = F.grid_sample(imgs_b, grid, mode='bilinear',
                           padding_mode='border', align_corners=True)[:, 0]  # [F,h,w]
    return warped


def merge_burst_bayer_planes(
        burst_planes: List[Dict[str, Tensor]],
        disp_field: Tensor,  # [F,2,Ht,Wt] at plane scale
        tile_size: int = 16,
        c_const: float = 8.0,
        A_param: float = 1.0,
        B_param: float = 0.0,
        ref_idx: int = 0,
) -> Dict[str, Tensor]:
    """
    Vectorized HDR+ merge per Bayer plane (Eq. 6–7), with windowed overlap-add.
    """
    Fm = len(burst_planes)
    device = disp_field.device
    dtype = burst_planes[0]['R'].dtype
    h, w = burst_planes[0]['R'].shape
    stride = tile_size // 2
    pad = (tile_size - stride) // 2

    win = raised_cosine_window_2d(tile_size, device, dtype)

    merged: Dict[str, Tensor] = {}

    for ch in ['R', 'G1', 'G2', 'B']:
        # Stack channel images [F,h,w]
        imgs = torch.stack([bp[ch] for bp in burst_planes], dim=0)

        # Warp each frame using the piecewise-constant tile displacement
        warped = _warp_burst_with_tilewise_flow(imgs, disp_field, tile_size)  # [F,h,w]

        # Extract overlapped tiles for all frames
        tiles = F.unfold(warped, kernel_size=tile_size, stride=stride, padding=pad)  # [F, n^2, Nt]
        Nt = tiles.shape[-1]
        tiles = tiles.permute(0, 2, 1).reshape(Fm, Nt, tile_size, tile_size)  # [F,Nt,n,n]

        # Apply window
        tiles_w = tiles * win

        # FFT per tile, per frame
        fft_tiles = torch.fft.fft2(tiles_w)  # [F,Nt,n,n]
        Tref = fft_tiles[ref_idx:ref_idx + 1]  # [1,Nt,n,n]
        Dz = fft_tiles - Tref
        mag2 = (Dz.real ** 2 + Dz.imag ** 2)  # [F,Nt,n,n]

        # Per-tile noise: σ² = A * x_rms + B (RMS of reference tile in spatial domain)
        ref_tiles_spatial = tiles[ref_idx]  # [Nt,n,n]
        sigma2_tile = (A_param * rms(ref_tiles_spatial.view(Nt, -1)).view(Nt, 1, 1) + B_param)  # [Nt,1,1]

        # Effective variance in DFT domain (window power and differencing)
        n = tile_size
        sigma2_eff = sigma2_tile * (n * n) * (1.0 / 16.0) ** 2 * 2.0  # match paper scaling

        # Eq. 7: A_z = |D_z|^2 / (|D_z|^2 + c * σ_eff²)
        Az = mag2 / torch.clamp(mag2 + c_const * sigma2_eff[None], min=1e-12)

        # Eq. 6: pairwise shrink towards reference in frequency domain, then average across frames
        merged_fft = fft_tiles + Az * (Tref - fft_tiles)
        merged_fft = merged_fft.mean(dim=0)  # [Nt,n,n]

        # Back to spatial
        merged_tiles = torch.fft.ifft2(merged_fft).real  # [Nt,n,n]

        # Fold (overlap-add) with window energy normalization
        merged_flat = (merged_tiles * win).reshape(Nt, -1).T[None]  # [1, n^2, Nt]
        img = F.fold(merged_flat, output_size=(h, w), kernel_size=tile_size, stride=stride, padding=pad)

        wflat = (win.reshape(1, -1, 1).repeat(1, 1, Nt))
        wsum = F.fold(wflat, output_size=(h, w), kernel_size=tile_size, stride=stride, padding=pad)

        merged[ch] = (img / torch.clamp(wsum, min=1e-8)).squeeze(0).squeeze(0)

    return merged


# ============================================================
# Simple demosaic + finishing to sRGB
# ============================================================

def demosaic_bilinear(planes: Dict[str, Tensor]) -> Tensor:
    """Bilinear demosaic to RGB (linear)."""
    h, w = planes['R'].shape
    R = F.interpolate(planes['R'][None, None], size=(2 * h, 2 * w), mode='bilinear', align_corners=False)[0, 0]
    G1 = F.interpolate(planes['G1'][None, None], size=(2 * h, 2 * w), mode='bilinear', align_corners=False)[0, 0]
    G2 = F.interpolate(planes['G2'][None, None], size=(2 * h, 2 * w), mode='bilinear', align_corners=False)[0, 0]
    B = F.interpolate(planes['B'][None, None], size=(2 * h, 2 * w), mode='bilinear', align_corners=False)[0, 0]
    G = 0.5 * (G1 + G2)
    return torch.stack([R, G, B], dim=-1)  # [H,W,3] linear


def apply_wb_ccm_gamma(rgb_lin: Tensor,
                       wb_gains: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                       ccm: Optional[Tensor] = None) -> Tensor:
    """White balance → CCM (sensor→sRGB) → mild tone → sRGB gamma."""
    rgb = rgb_lin * torch.tensor(wb_gains, dtype=rgb_lin.dtype, device=rgb_lin.device)
    if ccm is not None:
        H, W, _ = rgb.shape
        rgb = rgb.view(-1, 3) @ ccm.to(rgb.dtype).to(rgb.device).T
        rgb = rgb.view(H, W, 3)
    rgb = torch.clamp(rgb, 0.0, None)

    # mild contrast curve (simple S-curve)
    m, s = 0.5, 0.8
    rgb = torch.clamp((rgb - m) * s + m, 0.0, None)

    # sRGB gamma
    a = 0.055
    out = torch.where(rgb <= 0.0031308, 12.92 * rgb, (1 + a) * torch.pow(rgb, 1 / 2.4) - a)
    return torch.clamp(out, 0.0, 1.0)


# ============================================================
# Public API
# ============================================================

@torch.no_grad()
def hdrplus_process(
        burst_raw: List[Tensor],  # list of [H,W] float32, linear, black-level subtracted
        bayer_pattern: str = "RGGB",
        A_param: float = 1.0,
        B_param: float = 0.0,
        tile_size: int = 16,
        levels: int = 4,
        coarse_search: int = 32,
        fine_search: int = 4,
        wbgains: Tuple[float, float, float] = (1.0, 1.0, 1.0),
        ccm: Optional[Tensor] = None,
        c_const: float = 8.0,
        ref_idx: Optional[int] = None
) -> Tensor:
    """
    Run full HDR+ on a RAW burst and return sRGB in [0,1], shape [H,W,3].
    Notes:
      - Alignment & merge are done at half-res (Bayer plane scale).
      - Inputs must be linear RAW, black-level subtracted.
    """
    assert len(burst_raw) >= 2, "Need at least 2 frames"
    device = burst_raw[0].device
    dtype = burst_raw[0].dtype

    # Choose reference: sharpest among first 3 frames by green gradient (simple heuristic)
    if ref_idx is None:
        def green_grad(raw: Tensor) -> float:
            p = split_bayer_planes(raw, bayer_pattern)
            G = 0.5 * (p['G1'] + p['G2'])
            gx = F.pad(G[:, 1:] - G[:, :-1], (1, 0))
            gy = F.pad(G[1:, :] - G[:-1, :], (0, 0, 1, 0))
            return (gx.abs().mean() + gy.abs().mean()).item()

        k = min(3, len(burst_raw))
        scores = [green_grad(burst_raw[i]) for i in range(k)]
        ref_idx = int(torch.tensor(scores).argmax().item())

    # Alignment on 2×2-averaged grayscale (Bayer-plane resolution)
    gray_burst = [gray_from_bayer(r, bayer_pattern) for r in burst_raw]  # [h,w] half-res
    disp = align_burst(gray_burst, ref_idx,
                       tile_size=tile_size, levels=levels,
                       coarse_search=coarse_search, fine_search=fine_search)  # [F,2,Ht,Wt]

    # Split Bayer planes for merge (half-res)
    burst_planes = [split_bayer_planes(r, bayer_pattern) for r in burst_raw]

    # Merge per plane using Eq. (6–7)
    merged_planes = merge_burst_bayer_planes(
        burst_planes, disp, tile_size=tile_size, c_const=c_const,
        A_param=A_param, B_param=B_param, ref_idx=ref_idx
    )

    # Demosaic + finishing
    rgb_lin = demosaic_bilinear(merged_planes)  # [H,W,3] linear
    srgb = apply_wb_ccm_gamma(rgb_lin, wb_gains=wbgains, ccm=ccm)
    return srgb


# ============================================================
# Example (optional)
# ============================================================

if __name__ == "__main__":
    # Minimal smoke test with synthetic tensors
    H, W = 512, 768
    Fm = 4
    torch.manual_seed(0)
    # Fake RAW burst (normalized, black-level subtracted)
    # burst = [(torch.rand(H, W) * 0.2 + 0.1).float().cuda() for _ in range(Fm)]
    burst = [(torch.rand(H, W) * 0.2 + 0.1).float() for _ in range(Fm)]
    # Slight translations on later frames (toy)
    for i in range(1, Fm):
        burst[i] = torch.roll(burst[i], shifts=(i, -i), dims=(0, 1))  # just to have motion

    srgb = hdrplus_process(
        burst_raw=burst,
        bayer_pattern="RGGB",
        A_param=0.0005,  # example noise model
        B_param=0.0001,
        tile_size=16,
        # tile_size=4,
        levels=4,
        coarse_search=32,
        # coarse_search=4,
        fine_search=4,
        # fine_search=2,
        wbgains=(2.0, 1.0, 1.5),  # example WB
        ccm=None,  # optionally provide a 3x3 CCM tensor
        c_const=8.0,
    )
    print("sRGB shape:", srgb.shape, "min/max:", srgb.min().item(), srgb.max().item())
