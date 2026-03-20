import torch
import torch.nn.functional as F


def _make_base_grid(height: int, width: int, *, device, dtype) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    return torch.stack([xx, yy], dim=0).unsqueeze(0)  # [1, 2, H, W]


def _normalize_grid(grid_xy: torch.Tensor, height: int, width: int) -> torch.Tensor:
    x = 2.0 * grid_xy[:, 0] / max(width - 1, 1) - 1.0
    y = 2.0 * grid_xy[:, 1] / max(height - 1, 1) - 1.0
    return torch.stack([x, y], dim=-1)  # [B, H, W, 2]


def backward_warp(
    image: torch.Tensor,
    flow: torch.Tensor,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = True,
    return_valid_mask: bool = False,
):
    """
    Backward warp with grid_sample.

    Args:
        image: [B, C, H, W] source image.
        flow: [B, 2, H, W] target->source flow in pixel units.
              flow[:, 0] is dx and flow[:, 1] is dy.

    Returns:
        warped: [B, C, H, W]
        valid_mask: [B, 1, H, W] if return_valid_mask=True
    """
    if image.ndim != 4 or flow.ndim != 4:
        raise ValueError("image and flow must both be 4D tensors")
    if image.shape[0] != flow.shape[0] or image.shape[-2:] != flow.shape[-2:]:
        raise ValueError("image and flow must have matching batch and spatial dimensions")
    if flow.shape[1] != 2:
        raise ValueError("flow must have shape [B, 2, H, W]")

    b, _, h, w = image.shape
    base_grid = _make_base_grid(h, w, device=flow.device, dtype=flow.dtype)
    sample_grid = base_grid + flow
    sample_grid_norm = _normalize_grid(sample_grid, h, w)

    warped = F.grid_sample(
        image,
        sample_grid_norm,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )

    if not return_valid_mask:
        return warped

    x = sample_grid[:, 0]
    y = sample_grid[:, 1]
    valid = (x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)
    return warped, valid.unsqueeze(1).to(image.dtype)


def sample_flow(
    flow: torch.Tensor,
    coords_xy: torch.Tensor,
    *,
    padding_mode: str = "zeros",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Sample a flow field at pixel coordinates.

    Args:
        flow: [B, 2, H, W]
        coords_xy: [B, 2, H, W] pixel coordinates
    """
    if flow.shape != coords_xy.shape:
        raise ValueError("flow and coords_xy must have the same shape [B, 2, H, W]")

    _, _, h, w = flow.shape
    grid = _normalize_grid(coords_xy, h, w)
    return F.grid_sample(
        flow,
        grid,
        mode="bilinear",
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


def forward_backward_occlusion_mask(
    flow_ab: torch.Tensor,
    flow_ba: torch.Tensor,
    *,
    alpha: float = 0.01,
    beta: float = 0.5,
    return_error: bool = False,
):
    """
    Occlusion mask from forward-backward consistency.

    flow_ab maps pixels in A to pixels in B.
    flow_ba maps pixels in B to pixels in A.

    Returns:
        occluded_a: [B, 1, H, W] float mask, 1 means occluded in B.
        error: [B, 1, H, W] if return_error=True
    """
    if flow_ab.shape != flow_ba.shape:
        raise ValueError("flow_ab and flow_ba must have the same shape [B, 2, H, W]")

    _, _, h, w = flow_ab.shape
    base_grid = _make_base_grid(h, w, device=flow_ab.device, dtype=flow_ab.dtype)
    coords_in_b = base_grid + flow_ab
    flow_ba_at_ab = sample_flow(flow_ba, coords_in_b)

    residual = flow_ab + flow_ba_at_ab
    residual_sq = (residual * residual).sum(dim=1, keepdim=True)
    mag_sq = (flow_ab * flow_ab).sum(dim=1, keepdim=True) + (
        flow_ba_at_ab * flow_ba_at_ab
    ).sum(dim=1, keepdim=True)

    inside = (
        (coords_in_b[:, 0] >= 0)
        & (coords_in_b[:, 0] <= w - 1)
        & (coords_in_b[:, 1] >= 0)
        & (coords_in_b[:, 1] <= h - 1)
    ).unsqueeze(1)

    occluded = (~inside) | (residual_sq > alpha * mag_sq + beta)
    occluded = occluded.to(flow_ab.dtype)

    if return_error:
        return occluded, residual_sq.sqrt()
    return occluded
