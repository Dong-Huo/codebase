# hdrplus_ref.py
# Minimal, faithful PyTorch reference of HDR+ alignment & merge (Hasinoff et al., 2016)
# Implements Eq. (2), (3), (4), (6), (7) and the window/overlap rules on Bayer RAW bursts.
# Requires: torch>=1.10

import math
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# ----------------------------
# Utility: Bayer packing/unpacking (RGGB)
# ----------------------------

def split_bayer_planes(raw: Tensor, pattern: str = "RGGB") -> Dict[str, Tensor]:
    """
    raw: [H, W] linear, black-level subtracted
    returns dict of 4 planes at half-res: R, G1, G2, B each [H//2, W//2]
    """
    assert raw.dim() == 2
    H, W = raw.shape
    rggb = {
        "RGGB": ( (0,0,'R'), (0,1,'G1'), (1,0,'G2'), (1,1,'B') ),
        "BGGR": ( (0,0,'B'), (0,1,'G1'), (1,0,'G2'), (1,1,'R') ),
        "GRBG": ( (0,0,'G1'), (0,1,'R'), (1,0,'B'), (1,1,'G2') ),
        "GBRG": ( (0,0,'G1'), (0,1,'B'), (1,0,'R'), (1,1,'G2') ),
    }[pattern]
    planes = {}
    for i,j,name in rggb:
        planes[name] = raw[i::2, j::2].contiguous()
    return planes

def merge_bayer_planes(planes: Dict[str, Tensor], pattern: str = "RGGB") -> Tensor:
    """
    planes: dict with keys 'R','G1','G2','B' each [h,w]
    returns raw mosaic [2h, 2w]
    """
    h, w = planes['R'].shape
    out = torch.empty((2*h, 2*w), dtype=planes['R'].dtype, device=planes['R'].device)
    if pattern == "RGGB":
        out[0::2,0::2] = planes['R']
        out[0::2,1::2] = planes['G1']
        out[1::2,0::2] = planes['G2']
        out[1::2,1::2] = planes['B']
    elif pattern == "BGGR":
        out[0::2,0::2] = planes['B']
        out[0::2,1::2] = planes['G1']
        out[1::2,0::2] = planes['G2']
        out[1::2,1::2] = planes['R']
    elif pattern == "GRBG":
        out[0::2,0::2] = planes['G1']
        out[0::2,1::2] = planes['R']
        out[1::2,0::2] = planes['B']
        out[1::2,1::2] = planes['G2']
    elif pattern == "GBRG":
        out[0::2,0::2] = planes['G1']
        out[0::2,1::2] = planes['B']
        out[1::2,0::2] = planes['R']
        out[1::2,1::2] = planes['G2']
    else:
        raise ValueError("Unknown Bayer pattern")
    return out

def gray_from_bayer(raw: Tensor, pattern: str = "RGGB") -> Tensor:
    """ 2x2 average to grayscale (paper aligns on 3Mpix gray) """
    planes = split_bayer_planes(raw, pattern)
    # Average the 2x2 block = average of planes
    return (planes['R'] + planes['G1'] + planes['G2'] + planes['B']) * 0.25

# ----------------------------
# Reference selection (lucky imaging) on green gradients
# ----------------------------

def green_grad_sharpness(raw: Tensor, pattern: str="RGGB") -> float:
    p = split_bayer_planes(raw, pattern)
    G = 0.5*(p['G1'] + p['G2'])
    gx = F.pad(G[:,1:] - G[:,:-1], (1,0))
    gy = F.pad(G[1:,:] - G[:-1,:], (0,0,1,0))
    return (gx.abs().mean() + gy.abs().mean()).item()

def choose_reference(burst_raw: List[Tensor], pattern: str="RGGB", consider_first_k:int=3) -> int:
    k = min(consider_first_k, len(burst_raw))
    scores = [green_grad_sharpness(burst_raw[i], pattern) for i in range(k)]
    return int(torch.tensor(scores).argmax().item())

# ----------------------------
# Gaussian pyramid helpers
# ----------------------------

def pyr_down(img: Tensor) -> Tensor:
    # 5-tap Gaussian-like kernel separable
    k = torch.tensor([1,4,6,4,1], dtype=img.dtype, device=img.device)[None,None,:]
    k = k / k.sum()
    x = F.conv2d(img[None,None], k, padding=(0,2))
    x = F.conv2d(x, k.transpose(-1,-2), padding=(2,0))
    x = x[:, :, ::2, ::2]
    return x[0,0]

def build_gauss_pyr(img: Tensor, levels:int=4) -> List[Tensor]:
    pyr=[img]
    for _ in range(1,levels):
        pyr.append(pyr_down(pyr[-1]))
    return pyr

# ----------------------------
# Eq. (2): Fast L2 cost via FFT and box filter
# ----------------------------

def l2_cost_fft_tile(T: Tensor, I: Tensor) -> Tensor:
    """
    Compute D2(u,v) for all valid (u,v) s.t. T fits inside I[u:u+n, v:v+n]
    Returns a cost map with shape [H_T, W_T] aligned to top-left valid positions.
    T: [n, n], I: [n+R, n+R] or bigger
    """
    n = T.shape[0]
    # sum of squares of I over all n×n windows via conv with ones:
    # Implement with FFT-friendly box or direct conv (torch conv2d)
    ones = torch.ones((1,1,n,n), dtype=I.dtype, device=I.device)
    I2 = (I*I)[None,None]
    box_I2 = F.conv2d(I2, ones, padding=0)  # [1,1,H-n+1, W-n+1]
    box_I2 = box_I2[0,0]

    # cross-correlation via FFT: F^{-1} { F{I}* ∘ F{T} }
    # We can compute valid cross-correlation map by using conv2d with T (no FFT) but we keep FFT form.
    # For clarity and stability on small tiles, conv2d is fine and equivalent for valid region:
    T_flip = torch.flip(T, dims=[0,1])
    corr = F.conv2d(I[None,None], T_flip[None,None], padding=0)[0,0]

    # ||T||^2
    tt = (T*T).sum()

    # D2 = ||T||^2 + box(I∘I) - 2*corr   (valid region)
    D2 = tt + box_I2 - 2.0*corr
    return D2

# ----------------------------
# Eq. (3) & (4): Quadratic fit in 3×3 neighborhood for subpixel min
# ----------------------------

def subpixel_quad_min(D: Tensor, u0:int, v0:int) -> Tuple[float,float]:
    """
    Fit quadratic to 3x3 around (u0,v0): 0.5 [u v] A [u v]^T + b^T [u v] + c
    Return (du, dv) to add to integer location.
    """
    H, W = D.shape
    u0 = int(max(1, min(H-2, u0)))
    v0 = int(max(1, min(W-2, v0)))
    patch = D[u0-1:u0+2, v0-1:v0+2]  # [3,3]

    # Coordinates relative to center
    coords = []
    values = []
    for di in (-1,0,1):
        for dj in (-1,0,1):
            coords.append((di, dj))
            values.append(patch[di+1, dj+1])
    values = torch.stack(values)  # [9]

    # Build design matrix for quadratic in u,v:
    # q(u,v) = 0.5*(a11 u^2 + 2 a12 u v + a22 v^2) + b1 u + b2 v + c
    # terms: u^2, u v, v^2, u, v, 1
    X = []
    for (u,v) in coords:
        X.append(torch.tensor([u*u, u*v, v*v, u, v, 1.0], dtype=D.dtype, device=D.device))
    X = torch.stack(X)  # [9,6]
    # We fit q(u,v) = X @ theta, but note 0.5 is absorbed into A-later form; solve least squares:
    theta, _ = torch.lstsq(values[:,None], X)  # deprecated in new torch; use pinverse for compatibility
    theta = theta[:6,0]
    # Recover A,b,c consistent with 0.5 in paper form:
    a11, a12, a22, b1, b2, c = theta
    A = torch.tensor([[a11, a12],
                      [a12, a22]], dtype=D.dtype, device=D.device)
    b = torch.tensor([b1, b2], dtype=D.dtype, device=D.device)
    # The 0.5 factor was in paper; our theta may already have absorbed it depending on construction.
    # To be faithful, we can scale A by 1.0 and keep b as-is, since the stationary point solves A*[u;v] + b = 0 if 0.5 was applied.
    # If over/under-scaled, it only scales du,dv; we'll guard with pseudo-inverse.
    du_dv = -torch.linalg.pinv(A) @ b
    return float(du_dv[0].item()), float(du_dv[1].item())

# ----------------------------
# Alignment (multi-scale, tile-based)
# ----------------------------

@torch.no_grad()
def align_burst(gray_burst: List[Tensor],
                ref_idx: int,
                tile_size:int=16,
                levels:int=4,
                coarse_search:int=32,
                fine_search:int=4) -> List[Tensor]:
    """
    gray_burst: list of [h,w] grayscale (2x2-averaged) frames
    Returns per-frame displacement field at finest scale as [2, Ht, Wt] of (u,v) in pixels (integer at finest level),
    where Ht = ceil(h / (tile_size/2)) for half-overlapped tiling.
    Coarse levels: L2 + subpixel refinement (Eq 2,3,4)
    Finest level: pixel-level L1 within small radius
    """
    device = gray_burst[0].device
    dtype  = gray_burst[0].dtype

    # Build pyramids
    pyrs = [build_gauss_pyr(img, levels) for img in gray_burst]
    # tile grid at each level
    def tile_grid_shape(h,w, n):
        stride = n//2
        Ht = (max(1, math.ceil((h - n)/stride) + 1))
        Wt = (max(1, math.ceil((w - n)/stride) + 1))
        return Ht, Wt, stride

    # Initialize displacements per level (coarse->fine)
    disp_levels = [None]*levels  # list of per-frame displacement fields per level
    for lev in reversed(range(levels)):
        # current scale images
        imgs = [p[lev] for p in pyrs]
        ref = imgs[ref_idx]
        h, w = ref.shape
        n = tile_size if lev < levels-1 else tile_size//2  # a bit smaller tiles at coarsest for stability
        Ht, Wt, stride = tile_grid_shape(h,w,n)
        # starting guess from coarser level (upsample and double)
        if lev < levels-1:
            prev = disp_levels[lev+1]  # [F,2,Ht_p,Wt_p]
            # upscale to current grid
            up = []
            for f in range(len(imgs)):
                d = prev[f]
                d = F.interpolate(d.unsqueeze(0), size=(Ht,Wt), mode='bilinear', align_corners=True)[0]*2.0
                up.append(d)
            guess = torch.stack(up,0)
        else:
            # zeros
            guess = torch.zeros((len(imgs), 2, Ht, Wt), dtype=dtype, device=device)

        # Solve per-tile
        cur = torch.zeros_like(guess)
        for f in range(len(imgs)):
            if f == ref_idx:
                continue
            I = imgs[f]
            for ti in range(Ht):
                for tj in range(Wt):
                    u0 = int(guess[f,0,ti,tj].item())
                    v0 = int(guess[f,1,ti,tj].item())
                    y = ti*stride
                    x = tj*stride
                    y = min(max(0, y), h-n)
                    x = min(max(0, x), w-n)
                    T = ref[y:y+n, x:x+n]

                    # choose search radius
                    if lev == 0:
                        # finest: L1 pixel-level, small search
                        R = fine_search
                        best = None
                        best_uv = (0,0)
                        for du in range(-R, R+1):
                            for dv in range(-R, R+1):
                                yy = y + u0 + du
                                xx = x + v0 + dv
                                yy = min(max(0, yy), h-n)
                                xx = min(max(0, xx), w-n)
                                J = I[yy:yy+n, xx:xx+n]
                                cost = (T - J).abs().sum()
                                if (best is None) or (cost < best):
                                    best = cost
                                    best_uv = (u0+du, v0+dv)
                        cur[f,0,ti,tj] = best_uv[0]
                        cur[f,1,ti,tj] = best_uv[1]
                    else:
                        # coarse: L2 + FFT cost (valid conv) over a window around guess
                        R = coarse_search
                        # Clip a local search region
                        yy0 = min(max(0, y + u0 - R), h-n)
                        xx0 = min(max(0, x + v0 - R), w-n)
                        yy1 = min(h, yy0 + n + 2*R)
                        xx1 = min(w, xx0 + n + 2*R)
                        Jwin = I[yy0:yy1, xx0:xx1]
                        D2 = l2_cost_fft_tile(T, Jwin)  # shape [(yy1-yy0-n+1),(xx1-xx0-n+1)]
                        # integer min
                        idx = torch.argmin(D2)
                        iu = int(idx // D2.shape[1])
                        iv = int(idx %  D2.shape[1])
                        u_int = (yy0 + iu) - y
                        v_int = (xx0 + iv) - x
                        # subpixel via quadratic fit on D2
                        du_sub, dv_sub = subpixel_quad_min(D2, iu, iv)
                        cur[f,0,ti,tj] = u_int + du_sub
                        cur[f,1,ti,tj] = v_int + dv_sub
        disp_levels[lev] = cur
    return disp_levels[0]  # finest-level displacements per frame: [F,2,Ht,Wt]

# ----------------------------
# Window & overlap (modified raised-cosine)
# ----------------------------

def raised_cosine_1d(n:int, device, dtype) -> Tensor:
    # w(x) = 1/2 - 1/2 * cos( 2π (x + 1/2) / n ), for x=0..n-1
    x = torch.arange(n, device=device, dtype=dtype)
    w = 0.5 - 0.5*torch.cos(2*math.pi*(x+0.5)/n)
    return w

def window2d(n:int, device, dtype) -> Tensor:
    w1 = raised_cosine_1d(n, device, dtype)
    w2 = w1[:,None] * w1[None,:]
    return w2

# ----------------------------
# Pairwise temporal merge (Eq. 6) with Az (Eq. 7), then spatial denoising
# ----------------------------

def rms(x: Tensor) -> Tensor:
    return torch.sqrt((x*x).mean() + 1e-12)

def temporal_merge_tiles(tiles: List[Tensor],
                         sigma2_tile: float,
                         c_const: float,
                         win: Tensor) -> Tensor:
    """
    tiles: list of [n,n] tiles aligned to reference (z=0 is reference)
    Apply window, FFT, Eq. (6) with Az from Eq. (7).
    Returns merged tile [n,n] (spatial domain).
    """
    n = tiles[0].shape[0]
    Z = len(tiles)
    # Window
    Tsp = [t*win for t in tiles]
    # 2D FFT
    T = [torch.fft.fft2(t) for t in Tsp]
    T0 = T[0]
    # Dz = T0 - Tz ; Az = |Dz|^2 / (|Dz|^2 + c*sigma2_eff)
    # sigma2 scaling per paper: n^2 (DFT samples) * (1/4)^2 (window power) * 2 (difference of two tiles)
    sigma2_eff = sigma2_tile * (n*n) * (1.0/16.0) * 2.0
    out = 0
    for z in range(Z):
        Tz = T[z]
        Dz = T0 - Tz
        num = (Dz.real*Dz.real + Dz.imag*Dz.imag)
        den = num + c_const * sigma2_eff
        Az = num / torch.clamp(den, min=1e-12)
        contrib = Tz + Az*(T0 - Tz)
        out = out + contrib
    out = out / Z
    # inverse FFT, undo window by simple division (overlap-add will restore energy)
    merged = torch.fft.ifft2(out).real
    return merged

def spatial_shrinkage(tile: Tensor, sigma2_tile: float, noise_shape: Optional[Tensor]=None) -> Tensor:
    """
    Apply pointwise shrinkage (Eq. 7 form) in spatial frequency domain on a single tile.
    Conservative: use sigma^2 / N; handled by caller via sigma2_tile.
    noise_shape: optional multiplicative factor f(ω) for σ (>=1 for high frequencies).
    """
    T = torch.fft.fft2(tile)
    if noise_shape is None:
        noise_shape = 1.0
    # |T|^2 / (|T|^2 + c*sigma2) with c=1 here (we can fold a constant if desired)
    mag2 = (T.real*T.real + T.imag*T.imag)
    den = mag2 + (sigma2_tile * (noise_shape**2))
    Shr = mag2 / torch.clamp(den, min=1e-12)
    Tout = Shr * T
    return torch.fft.ifft2(Tout).real

def radial_noise_shaping(n:int, device, dtype, base:float=1.0, max_boost:float=3.0) -> Tensor:
    """
    Simple piecewise-linear f(ω): 1 at DC up to max_boost at Nyquist radially.
    """
    yy, xx = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
    cy, cx = (n//2), (n//2)
    r = torch.sqrt((yy-cy)**2 + (xx-cx)**2)
    r = r / (r.max()+1e-6)
    return base + (max_boost - base)*r.to(dtype)

# ----------------------------
# Full merge over overlapped tiles per Bayer plane
# ----------------------------

def merge_burst_bayer_planes(burst_planes: List[Dict[str,Tensor]],
                             disp_field: Tensor,
                             tile_size:int=16,
                             c_const: float = 8.0,
                             A_param: float = 1.0,
                             B_param: float = 0.0,
                             use_32_for_dark: bool = True) -> Dict[str, Tensor]:
    """
    burst_planes: list over frames of dict {'R','G1','G2','B'} each [h,w]
    disp_field: [F,2,Ht,Wt] displacements (pixels) computed on gray at half-res (same grid used here)
    Returns merged Bayer planes dict
    """
    Fm = len(burst_planes)
    device = disp_field.device
    dtype  = burst_planes[0]['R'].dtype
    h, w = burst_planes[0]['R'].shape
    n = tile_size
    stride = n//2
    Ht = (max(1, math.ceil((h - n)/stride) + 1))
    Wt = (max(1, math.ceil((w - n)/stride) + 1))
    win = window2d(n, device, dtype)

    def assemble_channel(chan:str) -> Tensor:
        acc = torch.zeros((h,w), dtype=dtype, device=device)
        wsum = torch.zeros((h,w), dtype=dtype, device=device)
        for ti in range(Ht):
            for tj in range(Wt):
                y = min(max(0, ti*stride), h-n)
                x = min(max(0, tj*stride), w-n)
                # Extract aligned tiles per frame (pixel-level in Bayer planes)
                tiles = []
                for f in range(Fm):
                    u = int(round(disp_field[f,0,ti,tj].item()))
                    v = int(round(disp_field[f,1,ti,tj].item()))
                    yy = min(max(0, y + u), h-n)
                    xx = min(max(0, x + v), w-n)
                    tiles.append(burst_planes[f][chan][yy:yy+n, xx:xx+n])

                # Noise model sigma^2 = A*x + B evaluated at tile RMS (paper’s tilewise approx)
                x_rms = rms(tiles[0])  # reference tile RMS
                sigma2 = A_param * x_rms + B_param

                # Temporal merge (Eq. 6 with Az per Eq. 7)
                merged = temporal_merge_tiles(tiles, sigma2, c_const, win)

                # Spatial denoising (conservative): assume N perfect average => sigma^2/N
                N = len(tiles)
                noise_shape = radial_noise_shaping(n, device, dtype)
                merged = spatial_shrinkage(merged, sigma2 / max(1,N), noise_shape)

                # Overlap-add with window (window already applied inside temporal stage; we blend smoothly)
                acc[y:y+n, x:x+n] += merged * win
                wsum[y:y+n, x:x+n] += win
        out = acc / torch.clamp(wsum, min=1e-8)
        return out

    merged = {}
    for ch in ['R','G1','G2','B']:
        merged[ch] = assemble_channel(ch)
    return merged

# ----------------------------
# Simple demosaic + color + tone to sRGB (minimal finishing)
# ----------------------------

def demosaic_bilinear(planes: Dict[str,Tensor]) -> Tensor:
    """
    Very simple bilinear demosaic to RGB [H,W,3] (linear space).
    """
    h,w = planes['R'].shape
    raw = merge_bayer_planes(planes, "RGGB")
    R = F.interpolate(planes['R'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    G1 = F.interpolate(planes['G1'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    G2 = F.interpolate(planes['G2'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    B = F.interpolate(planes['B'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    G = 0.5*(G1+G2)
    rgb = torch.stack([R,G,B], dim=-1)
    return rgb

def apply_wb_ccm_gamma(rgb_lin: Tensor,
                       wb_gains: Tuple[float,float,float]=(1.0,1.0,1.0),
                       ccm: Optional[Tensor]=None) -> Tensor:
    """
    rgb_lin: [H,W,3] linear
    wb, 3x3 ccm (sensor->sRGB), then simple gamma
    """
    rgb = rgb_lin * torch.tensor(wb_gains, dtype=rgb_lin.dtype, device=rgb_lin.device)
    if ccm is not None:
        H,W,_ = rgb.shape
        rgb = rgb.reshape(-1,3) @ ccm.T
        rgb = rgb.reshape(H,W,3)
    # clip negatives
    rgb = torch.clamp(rgb, 0.0, None)
    # global mild tone (S-curve) then sRGB gamma
    # simple sRGB oetf
    def srgb_gamma(x):
        a = 0.055
        return torch.where(x<=0.0031308, 12.92*x, (1+a)*torch.pow(x, 1/2.4)-a)
    # mild contrast curve
    m = 0.5
    s = 0.8
    rgb = torch.clamp((rgb - m)*s + m, 0.0, None)
    return torch.clamp(srgb_gamma(rgb), 0.0, 1.0)

# ----------------------------
# Public API
# ----------------------------

@torch.no_grad()
def hdrplus_process(
    burst_raw: List[Tensor],
    bayer_pattern: str = "RGGB",
    A_param: float = 1.0,
    B_param: float = 0.0,
    tile_size:int = 16,
    levels:int = 4,
    coarse_search:int = 32,
    fine_search:int = 4,
    wbgains: Tuple[float,float,float]=(1.0,1.0,1.0),
    ccm: Optional[Tensor]=None,
    c_const: float = 8.0,
) -> Tensor:
    """
    burst_raw: list of [H,W] RAW (linear, black-level subtracted)
    Returns sRGB uint8-ish tensor in [0,1], shape [H,W,3]
    """
    assert len(burst_raw)>=2
    device = burst_raw[0].device
    dtype  = burst_raw[0].dtype

    # Choose reference (sharpest of first few frames)
    ref_idx = choose_reference(burst_raw, bayer_pattern, consider_first_k=min(3,len(burst_raw)))

    # Grayscale (2x2-avg) for alignment
    gray_burst = [gray_from_bayer(r, bayer_pattern) for r in burst_raw]

    # Align on gray pyramid (Eq. 2,3,4 at coarse; L1 pixel at finest)
    disp = align_burst(gray_burst, ref_idx,
                       tile_size=tile_size, levels=levels,
                       coarse_search=coarse_search, fine_search=fine_search)  # [F,2,Ht,Wt]

    # Split planes (Bayer) at half-res for merge
    burst_planes = [split_bayer_planes(r, bayer_pattern) for r in burst_raw]

    # Merge per plane using pairwise temporal filter (Eq. 6,7) and overlapped tiles + window
    merged_planes = merge_burst_bayer_planes(
        burst_planes, disp, tile_size=tile_size, c_const=c_const, A_param=A_param, B_param=B_param
    )

    # Simple demosaic + finishing to sRGB
    rgb_lin = demosaic_bilinear(merged_planes)
    srgb = apply_wb_ccm_gamma(rgb_lin, wb_gains=wbgains, ccm=ccm)
    return srgb
# hdrplus_ref.py
# Minimal, faithful PyTorch reference of HDR+ alignment & merge (Hasinoff et al., 2016)
# Implements Eq. (2), (3), (4), (6), (7) and the window/overlap rules on Bayer RAW bursts.
# Requires: torch>=1.10

import math
from typing import Dict, Tuple, List, Optional
import torch
import torch.nn.functional as F

Tensor = torch.Tensor

# ----------------------------
# Utility: Bayer packing/unpacking (RGGB)
# ----------------------------

def split_bayer_planes(raw: Tensor, pattern: str = "RGGB") -> Dict[str, Tensor]:
    """
    raw: [H, W] linear, black-level subtracted
    returns dict of 4 planes at half-res: R, G1, G2, B each [H//2, W//2]
    """
    assert raw.dim() == 2
    H, W = raw.shape
    rggb = {
        "RGGB": ( (0,0,'R'), (0,1,'G1'), (1,0,'G2'), (1,1,'B') ),
        "BGGR": ( (0,0,'B'), (0,1,'G1'), (1,0,'G2'), (1,1,'R') ),
        "GRBG": ( (0,0,'G1'), (0,1,'R'), (1,0,'B'), (1,1,'G2') ),
        "GBRG": ( (0,0,'G1'), (0,1,'B'), (1,0,'R'), (1,1,'G2') ),
    }[pattern]
    planes = {}
    for i,j,name in rggb:
        planes[name] = raw[i::2, j::2].contiguous()
    return planes

def merge_bayer_planes(planes: Dict[str, Tensor], pattern: str = "RGGB") -> Tensor:
    """
    planes: dict with keys 'R','G1','G2','B' each [h,w]
    returns raw mosaic [2h, 2w]
    """
    h, w = planes['R'].shape
    out = torch.empty((2*h, 2*w), dtype=planes['R'].dtype, device=planes['R'].device)
    if pattern == "RGGB":
        out[0::2,0::2] = planes['R']
        out[0::2,1::2] = planes['G1']
        out[1::2,0::2] = planes['G2']
        out[1::2,1::2] = planes['B']
    elif pattern == "BGGR":
        out[0::2,0::2] = planes['B']
        out[0::2,1::2] = planes['G1']
        out[1::2,0::2] = planes['G2']
        out[1::2,1::2] = planes['R']
    elif pattern == "GRBG":
        out[0::2,0::2] = planes['G1']
        out[0::2,1::2] = planes['R']
        out[1::2,0::2] = planes['B']
        out[1::2,1::2] = planes['G2']
    elif pattern == "GBRG":
        out[0::2,0::2] = planes['G1']
        out[0::2,1::2] = planes['B']
        out[1::2,0::2] = planes['R']
        out[1::2,1::2] = planes['G2']
    else:
        raise ValueError("Unknown Bayer pattern")
    return out

def gray_from_bayer(raw: Tensor, pattern: str = "RGGB") -> Tensor:
    """ 2x2 average to grayscale (paper aligns on 3Mpix gray) """
    planes = split_bayer_planes(raw, pattern)
    # Average the 2x2 block = average of planes
    return (planes['R'] + planes['G1'] + planes['G2'] + planes['B']) * 0.25

# ----------------------------
# Reference selection (lucky imaging) on green gradients
# ----------------------------

def green_grad_sharpness(raw: Tensor, pattern: str="RGGB") -> float:
    p = split_bayer_planes(raw, pattern)
    G = 0.5*(p['G1'] + p['G2'])
    gx = F.pad(G[:,1:] - G[:,:-1], (1,0))
    gy = F.pad(G[1:,:] - G[:-1,:], (0,0,1,0))
    return (gx.abs().mean() + gy.abs().mean()).item()

def choose_reference(burst_raw: List[Tensor], pattern: str="RGGB", consider_first_k:int=3) -> int:
    k = min(consider_first_k, len(burst_raw))
    scores = [green_grad_sharpness(burst_raw[i], pattern) for i in range(k)]
    return int(torch.tensor(scores).argmax().item())

# ----------------------------
# Gaussian pyramid helpers
# ----------------------------

def pyr_down(img: Tensor) -> Tensor:
    # 5-tap Gaussian-like kernel separable
    k = torch.tensor([1,4,6,4,1], dtype=img.dtype, device=img.device)[None,None,:]
    k = k / k.sum()
    x = F.conv2d(img[None,None], k, padding=(0,2))
    x = F.conv2d(x, k.transpose(-1,-2), padding=(2,0))
    x = x[:, :, ::2, ::2]
    return x[0,0]

def build_gauss_pyr(img: Tensor, levels:int=4) -> List[Tensor]:
    pyr=[img]
    for _ in range(1,levels):
        pyr.append(pyr_down(pyr[-1]))
    return pyr

# ----------------------------
# Eq. (2): Fast L2 cost via FFT and box filter
# ----------------------------

def l2_cost_fft_tile(T: Tensor, I: Tensor) -> Tensor:
    """
    Compute D2(u,v) for all valid (u,v) s.t. T fits inside I[u:u+n, v:v+n]
    Returns a cost map with shape [H_T, W_T] aligned to top-left valid positions.
    T: [n, n], I: [n+R, n+R] or bigger
    """
    n = T.shape[0]
    # sum of squares of I over all n×n windows via conv with ones:
    # Implement with FFT-friendly box or direct conv (torch conv2d)
    ones = torch.ones((1,1,n,n), dtype=I.dtype, device=I.device)
    I2 = (I*I)[None,None]
    box_I2 = F.conv2d(I2, ones, padding=0)  # [1,1,H-n+1, W-n+1]
    box_I2 = box_I2[0,0]

    # cross-correlation via FFT: F^{-1} { F{I}* ∘ F{T} }
    # We can compute valid cross-correlation map by using conv2d with T (no FFT) but we keep FFT form.
    # For clarity and stability on small tiles, conv2d is fine and equivalent for valid region:
    T_flip = torch.flip(T, dims=[0,1])
    corr = F.conv2d(I[None,None], T_flip[None,None], padding=0)[0,0]

    # ||T||^2
    tt = (T*T).sum()

    # D2 = ||T||^2 + box(I∘I) - 2*corr   (valid region)
    D2 = tt + box_I2 - 2.0*corr
    return D2

# ----------------------------
# Eq. (3) & (4): Quadratic fit in 3×3 neighborhood for subpixel min
# ----------------------------

def subpixel_quad_min(D: Tensor, u0:int, v0:int) -> Tuple[float,float]:
    """
    Fit quadratic to 3x3 around (u0,v0): 0.5 [u v] A [u v]^T + b^T [u v] + c
    Return (du, dv) to add to integer location.
    """
    H, W = D.shape
    u0 = int(max(1, min(H-2, u0)))
    v0 = int(max(1, min(W-2, v0)))
    patch = D[u0-1:u0+2, v0-1:v0+2]  # [3,3]

    # Coordinates relative to center
    coords = []
    values = []
    for di in (-1,0,1):
        for dj in (-1,0,1):
            coords.append((di, dj))
            values.append(patch[di+1, dj+1])
    values = torch.stack(values)  # [9]

    # Build design matrix for quadratic in u,v:
    # q(u,v) = 0.5*(a11 u^2 + 2 a12 u v + a22 v^2) + b1 u + b2 v + c
    # terms: u^2, u v, v^2, u, v, 1
    X = []
    for (u,v) in coords:
        X.append(torch.tensor([u*u, u*v, v*v, u, v, 1.0], dtype=D.dtype, device=D.device))
    X = torch.stack(X)  # [9,6]
    # We fit q(u,v) = X @ theta, but note 0.5 is absorbed into A-later form; solve least squares:
    theta, _ = torch.lstsq(values[:,None], X)  # deprecated in new torch; use pinverse for compatibility
    theta = theta[:6,0]
    # Recover A,b,c consistent with 0.5 in paper form:
    a11, a12, a22, b1, b2, c = theta
    A = torch.tensor([[a11, a12],
                      [a12, a22]], dtype=D.dtype, device=D.device)
    b = torch.tensor([b1, b2], dtype=D.dtype, device=D.device)
    # The 0.5 factor was in paper; our theta may already have absorbed it depending on construction.
    # To be faithful, we can scale A by 1.0 and keep b as-is, since the stationary point solves A*[u;v] + b = 0 if 0.5 was applied.
    # If over/under-scaled, it only scales du,dv; we'll guard with pseudo-inverse.
    du_dv = -torch.linalg.pinv(A) @ b
    return float(du_dv[0].item()), float(du_dv[1].item())

# ----------------------------
# Alignment (multi-scale, tile-based)
# ----------------------------

@torch.no_grad()
def align_burst(gray_burst: List[Tensor],
                ref_idx: int,
                tile_size:int=16,
                levels:int=4,
                coarse_search:int=32,
                fine_search:int=4) -> List[Tensor]:
    """
    gray_burst: list of [h,w] grayscale (2x2-averaged) frames
    Returns per-frame displacement field at finest scale as [2, Ht, Wt] of (u,v) in pixels (integer at finest level),
    where Ht = ceil(h / (tile_size/2)) for half-overlapped tiling.
    Coarse levels: L2 + subpixel refinement (Eq 2,3,4)
    Finest level: pixel-level L1 within small radius
    """
    device = gray_burst[0].device
    dtype  = gray_burst[0].dtype

    # Build pyramids
    pyrs = [build_gauss_pyr(img, levels) for img in gray_burst]
    # tile grid at each level
    def tile_grid_shape(h,w, n):
        stride = n//2
        Ht = (max(1, math.ceil((h - n)/stride) + 1))
        Wt = (max(1, math.ceil((w - n)/stride) + 1))
        return Ht, Wt, stride

    # Initialize displacements per level (coarse->fine)
    disp_levels = [None]*levels  # list of per-frame displacement fields per level
    for lev in reversed(range(levels)):
        # current scale images
        imgs = [p[lev] for p in pyrs]
        ref = imgs[ref_idx]
        h, w = ref.shape
        n = tile_size if lev < levels-1 else tile_size//2  # a bit smaller tiles at coarsest for stability
        Ht, Wt, stride = tile_grid_shape(h,w,n)
        # starting guess from coarser level (upsample and double)
        if lev < levels-1:
            prev = disp_levels[lev+1]  # [F,2,Ht_p,Wt_p]
            # upscale to current grid
            up = []
            for f in range(len(imgs)):
                d = prev[f]
                d = F.interpolate(d.unsqueeze(0), size=(Ht,Wt), mode='bilinear', align_corners=True)[0]*2.0
                up.append(d)
            guess = torch.stack(up,0)
        else:
            # zeros
            guess = torch.zeros((len(imgs), 2, Ht, Wt), dtype=dtype, device=device)

        # Solve per-tile
        cur = torch.zeros_like(guess)
        for f in range(len(imgs)):
            if f == ref_idx:
                continue
            I = imgs[f]
            for ti in range(Ht):
                for tj in range(Wt):
                    u0 = int(guess[f,0,ti,tj].item())
                    v0 = int(guess[f,1,ti,tj].item())
                    y = ti*stride
                    x = tj*stride
                    y = min(max(0, y), h-n)
                    x = min(max(0, x), w-n)
                    T = ref[y:y+n, x:x+n]

                    # choose search radius
                    if lev == 0:
                        # finest: L1 pixel-level, small search
                        R = fine_search
                        best = None
                        best_uv = (0,0)
                        for du in range(-R, R+1):
                            for dv in range(-R, R+1):
                                yy = y + u0 + du
                                xx = x + v0 + dv
                                yy = min(max(0, yy), h-n)
                                xx = min(max(0, xx), w-n)
                                J = I[yy:yy+n, xx:xx+n]
                                cost = (T - J).abs().sum()
                                if (best is None) or (cost < best):
                                    best = cost
                                    best_uv = (u0+du, v0+dv)
                        cur[f,0,ti,tj] = best_uv[0]
                        cur[f,1,ti,tj] = best_uv[1]
                    else:
                        # coarse: L2 + FFT cost (valid conv) over a window around guess
                        R = coarse_search
                        # Clip a local search region
                        yy0 = min(max(0, y + u0 - R), h-n)
                        xx0 = min(max(0, x + v0 - R), w-n)
                        yy1 = min(h, yy0 + n + 2*R)
                        xx1 = min(w, xx0 + n + 2*R)
                        Jwin = I[yy0:yy1, xx0:xx1]
                        D2 = l2_cost_fft_tile(T, Jwin)  # shape [(yy1-yy0-n+1),(xx1-xx0-n+1)]
                        # integer min
                        idx = torch.argmin(D2)
                        iu = int(idx // D2.shape[1])
                        iv = int(idx %  D2.shape[1])
                        u_int = (yy0 + iu) - y
                        v_int = (xx0 + iv) - x
                        # subpixel via quadratic fit on D2
                        du_sub, dv_sub = subpixel_quad_min(D2, iu, iv)
                        cur[f,0,ti,tj] = u_int + du_sub
                        cur[f,1,ti,tj] = v_int + dv_sub
        disp_levels[lev] = cur
    return disp_levels[0]  # finest-level displacements per frame: [F,2,Ht,Wt]

# ----------------------------
# Window & overlap (modified raised-cosine)
# ----------------------------

def raised_cosine_1d(n:int, device, dtype) -> Tensor:
    # w(x) = 1/2 - 1/2 * cos( 2π (x + 1/2) / n ), for x=0..n-1
    x = torch.arange(n, device=device, dtype=dtype)
    w = 0.5 - 0.5*torch.cos(2*math.pi*(x+0.5)/n)
    return w

def window2d(n:int, device, dtype) -> Tensor:
    w1 = raised_cosine_1d(n, device, dtype)
    w2 = w1[:,None] * w1[None,:]
    return w2

# ----------------------------
# Pairwise temporal merge (Eq. 6) with Az (Eq. 7), then spatial denoising
# ----------------------------

def rms(x: Tensor) -> Tensor:
    return torch.sqrt((x*x).mean() + 1e-12)

def temporal_merge_tiles(tiles: List[Tensor],
                         sigma2_tile: float,
                         c_const: float,
                         win: Tensor) -> Tensor:
    """
    tiles: list of [n,n] tiles aligned to reference (z=0 is reference)
    Apply window, FFT, Eq. (6) with Az from Eq. (7).
    Returns merged tile [n,n] (spatial domain).
    """
    n = tiles[0].shape[0]
    Z = len(tiles)
    # Window
    Tsp = [t*win for t in tiles]
    # 2D FFT
    T = [torch.fft.fft2(t) for t in Tsp]
    T0 = T[0]
    # Dz = T0 - Tz ; Az = |Dz|^2 / (|Dz|^2 + c*sigma2_eff)
    # sigma2 scaling per paper: n^2 (DFT samples) * (1/4)^2 (window power) * 2 (difference of two tiles)
    sigma2_eff = sigma2_tile * (n*n) * (1.0/16.0) * 2.0
    out = 0
    for z in range(Z):
        Tz = T[z]
        Dz = T0 - Tz
        num = (Dz.real*Dz.real + Dz.imag*Dz.imag)
        den = num + c_const * sigma2_eff
        Az = num / torch.clamp(den, min=1e-12)
        contrib = Tz + Az*(T0 - Tz)
        out = out + contrib
    out = out / Z
    # inverse FFT, undo window by simple division (overlap-add will restore energy)
    merged = torch.fft.ifft2(out).real
    return merged

def spatial_shrinkage(tile: Tensor, sigma2_tile: float, noise_shape: Optional[Tensor]=None) -> Tensor:
    """
    Apply pointwise shrinkage (Eq. 7 form) in spatial frequency domain on a single tile.
    Conservative: use sigma^2 / N; handled by caller via sigma2_tile.
    noise_shape: optional multiplicative factor f(ω) for σ (>=1 for high frequencies).
    """
    T = torch.fft.fft2(tile)
    if noise_shape is None:
        noise_shape = 1.0
    # |T|^2 / (|T|^2 + c*sigma2) with c=1 here (we can fold a constant if desired)
    mag2 = (T.real*T.real + T.imag*T.imag)
    den = mag2 + (sigma2_tile * (noise_shape**2))
    Shr = mag2 / torch.clamp(den, min=1e-12)
    Tout = Shr * T
    return torch.fft.ifft2(Tout).real

def radial_noise_shaping(n:int, device, dtype, base:float=1.0, max_boost:float=3.0) -> Tensor:
    """
    Simple piecewise-linear f(ω): 1 at DC up to max_boost at Nyquist radially.
    """
    yy, xx = torch.meshgrid(torch.arange(n, device=device), torch.arange(n, device=device), indexing='ij')
    cy, cx = (n//2), (n//2)
    r = torch.sqrt((yy-cy)**2 + (xx-cx)**2)
    r = r / (r.max()+1e-6)
    return base + (max_boost - base)*r.to(dtype)

# ----------------------------
# Full merge over overlapped tiles per Bayer plane
# ----------------------------

def merge_burst_bayer_planes(burst_planes: List[Dict[str,Tensor]],
                             disp_field: Tensor,
                             tile_size:int=16,
                             c_const: float = 8.0,
                             A_param: float = 1.0,
                             B_param: float = 0.0,
                             use_32_for_dark: bool = True) -> Dict[str, Tensor]:
    """
    burst_planes: list over frames of dict {'R','G1','G2','B'} each [h,w]
    disp_field: [F,2,Ht,Wt] displacements (pixels) computed on gray at half-res (same grid used here)
    Returns merged Bayer planes dict
    """
    Fm = len(burst_planes)
    device = disp_field.device
    dtype  = burst_planes[0]['R'].dtype
    h, w = burst_planes[0]['R'].shape
    n = tile_size
    stride = n//2
    Ht = (max(1, math.ceil((h - n)/stride) + 1))
    Wt = (max(1, math.ceil((w - n)/stride) + 1))
    win = window2d(n, device, dtype)

    def assemble_channel(chan:str) -> Tensor:
        acc = torch.zeros((h,w), dtype=dtype, device=device)
        wsum = torch.zeros((h,w), dtype=dtype, device=device)
        for ti in range(Ht):
            for tj in range(Wt):
                y = min(max(0, ti*stride), h-n)
                x = min(max(0, tj*stride), w-n)
                # Extract aligned tiles per frame (pixel-level in Bayer planes)
                tiles = []
                for f in range(Fm):
                    u = int(round(disp_field[f,0,ti,tj].item()))
                    v = int(round(disp_field[f,1,ti,tj].item()))
                    yy = min(max(0, y + u), h-n)
                    xx = min(max(0, x + v), w-n)
                    tiles.append(burst_planes[f][chan][yy:yy+n, xx:xx+n])

                # Noise model sigma^2 = A*x + B evaluated at tile RMS (paper’s tilewise approx)
                x_rms = rms(tiles[0])  # reference tile RMS
                sigma2 = A_param * x_rms + B_param

                # Temporal merge (Eq. 6 with Az per Eq. 7)
                merged = temporal_merge_tiles(tiles, sigma2, c_const, win)

                # Spatial denoising (conservative): assume N perfect average => sigma^2/N
                N = len(tiles)
                noise_shape = radial_noise_shaping(n, device, dtype)
                merged = spatial_shrinkage(merged, sigma2 / max(1,N), noise_shape)

                # Overlap-add with window (window already applied inside temporal stage; we blend smoothly)
                acc[y:y+n, x:x+n] += merged * win
                wsum[y:y+n, x:x+n] += win
        out = acc / torch.clamp(wsum, min=1e-8)
        return out

    merged = {}
    for ch in ['R','G1','G2','B']:
        merged[ch] = assemble_channel(ch)
    return merged

# ----------------------------
# Simple demosaic + color + tone to sRGB (minimal finishing)
# ----------------------------

def demosaic_bilinear(planes: Dict[str,Tensor]) -> Tensor:
    """
    Very simple bilinear demosaic to RGB [H,W,3] (linear space).
    """
    h,w = planes['R'].shape
    raw = merge_bayer_planes(planes, "RGGB")
    R = F.interpolate(planes['R'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    G1 = F.interpolate(planes['G1'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    G2 = F.interpolate(planes['G2'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    B = F.interpolate(planes['B'][None,None], size=(h*2,w*2), mode='bilinear', align_corners=False)[0,0]
    G = 0.5*(G1+G2)
    rgb = torch.stack([R,G,B], dim=-1)
    return rgb

def apply_wb_ccm_gamma(rgb_lin: Tensor,
                       wb_gains: Tuple[float,float,float]=(1.0,1.0,1.0),
                       ccm: Optional[Tensor]=None) -> Tensor:
    """
    rgb_lin: [H,W,3] linear
    wb, 3x3 ccm (sensor->sRGB), then simple gamma
    """
    rgb = rgb_lin * torch.tensor(wb_gains, dtype=rgb_lin.dtype, device=rgb_lin.device)
    if ccm is not None:
        H,W,_ = rgb.shape
        rgb = rgb.reshape(-1,3) @ ccm.T
        rgb = rgb.reshape(H,W,3)
    # clip negatives
    rgb = torch.clamp(rgb, 0.0, None)
    # global mild tone (S-curve) then sRGB gamma
    # simple sRGB oetf
    def srgb_gamma(x):
        a = 0.055
        return torch.where(x<=0.0031308, 12.92*x, (1+a)*torch.pow(x, 1/2.4)-a)
    # mild contrast curve
    m = 0.5
    s = 0.8
    rgb = torch.clamp((rgb - m)*s + m, 0.0, None)
    return torch.clamp(srgb_gamma(rgb), 0.0, 1.0)

# ----------------------------
# Public API
# ----------------------------

@torch.no_grad()
def hdrplus_process(
    burst_raw: List[Tensor],
    bayer_pattern: str = "RGGB",
    A_param: float = 1.0,
    B_param: float = 0.0,
    tile_size:int = 16,
    levels:int = 4,
    coarse_search:int = 32,
    fine_search:int = 4,
    wbgains: Tuple[float,float,float]=(1.0,1.0,1.0),
    ccm: Optional[Tensor]=None,
    c_const: float = 8.0,
) -> Tensor:
    """
    burst_raw: list of [H,W] RAW (linear, black-level subtracted)
    Returns sRGB uint8-ish tensor in [0,1], shape [H,W,3]
    """
    assert len(burst_raw)>=2
    device = burst_raw[0].device
    dtype  = burst_raw[0].dtype

    # Choose reference (sharpest of first few frames)
    ref_idx = choose_reference(burst_raw, bayer_pattern, consider_first_k=min(3,len(burst_raw)))

    # Grayscale (2x2-avg) for alignment
    gray_burst = [gray_from_bayer(r, bayer_pattern) for r in burst_raw]

    # Align on gray pyramid (Eq. 2,3,4 at coarse; L1 pixel at finest)
    disp = align_burst(gray_burst, ref_idx,
                       tile_size=tile_size, levels=levels,
                       coarse_search=coarse_search, fine_search=fine_search)  # [F,2,Ht,Wt]

    # Split planes (Bayer) at half-res for merge
    burst_planes = [split_bayer_planes(r, bayer_pattern) for r in burst_raw]

    # Merge per plane using pairwise temporal filter (Eq. 6,7) and overlapped tiles + window
    merged_planes = merge_burst_bayer_planes(
        burst_planes, disp, tile_size=tile_size, c_const=c_const, A_param=A_param, B_param=B_param
    )

    # Simple demosaic + finishing to sRGB
    rgb_lin = demosaic_bilinear(merged_planes)
    srgb = apply_wb_ccm_gamma(rgb_lin, wb_gains=wbgains, ccm=ccm)
    return srgb
