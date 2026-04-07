"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
------------------------------------------------------------------------------

Encoder wrapper for dense segmentation using the V-JEPA 2.1 backbone.

The V-JEPA 2.1 ViT natively exposes intermediate-layer outputs via its
`out_layers` constructor argument.  Passing the model's hierarchical layer
indices (e.g. [5, 11, 17, 23] for ViT-Large/depth=24) makes `forward()`
return a list of four normed feature maps instead of a single final token
tensor.  This wrapper:

  1. Loads the frozen V-JEPA 2.1 encoder with those intermediate hooks.
  2. Wraps it in `DenseEncoderWrapper`, which accepts an image tensor
     [B, C, H, W] and returns per-pixel logits [B, num_classes, H, W]
     using a lightweight FPN + bilinear upsampling decoder.

Hierarchical layer indices per encoder depth:
  depth=12  (ViT-Base)     : [2, 5, 8, 11]
  depth=24  (ViT-Large)    : [5, 11, 17, 23]
  depth=40  (ViT-Giant)    : [9, 19, 29, 39]
  depth=48  (ViT-Gigantic) : [11, 23, 37, 47]
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

import app.vjepa_2_1.models.vision_transformer as vit

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Maps encoder depth to the pre-defined hierarchical layer indices.
_HIER_LAYERS = {
    12: [2, 5, 8, 11],
    24: [5, 11, 17, 23],
    40: [9, 19, 29, 39],
    48: [11, 23, 37, 47],
}


def init_module(
    resolution: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    # frames_per_clip is accepted but unused: segmentation operates on images
    frames_per_clip: int = 1,
    **kwargs,
):
    logger.info(f"Loading V-JEPA 2.1 pretrained model from {checkpoint!r}")
    ckpt = torch.load(checkpoint, map_location="cpu")

    enc_kwargs = model_kwargs["encoder"]
    model_name = enc_kwargs.get("model_name", "vit_large")
    ckpt_key = enc_kwargs.get("checkpoint_key", "ema_encoder")
    patch_size = enc_kwargs.get("patch_size", 16)

    # Determine out_layers from depth or explicit config
    depth = enc_kwargs.get("depth", None)
    out_layers = enc_kwargs.get("out_layers", None)
    if out_layers is None:
        # Infer from model name if depth not given
        if depth is None:
            depth = {"vit_base": 12, "vit_large": 24, "vit_giant_xformers": 40,
                     "vit_gigantic_xformers": 48}.get(model_name, 24)
        out_layers = _HIER_LAYERS[depth]
    logger.info(f"Using out_layers={out_layers} for dense feature extraction")

    # Build encoder — out_layers activates multi-scale output mode
    build_kwargs = dict(
        img_size=(resolution, resolution),
        num_frames=1,           # image mode
        patch_size=patch_size,
        tubelet_size=2,
        use_rope=enc_kwargs.get("use_rope", True),
        img_temporal_dim_size=enc_kwargs.get("img_temporal_dim_size", 1),
        interpolate_rope=enc_kwargs.get("interpolate_rope", True),
        out_layers=out_layers,
        use_sdpa=enc_kwargs.get("use_sdpa", True),
        uniform_power=enc_kwargs.get("uniform_power", False),
        modality_embedding=enc_kwargs.get("modality_embedding", True),
    )
    encoder = vit.__dict__[model_name](**build_kwargs)

    # Load pretrained weights
    sd = ckpt[ckpt_key]
    sd = {k.replace("module.", "").replace("backbone.", ""): v for k, v in sd.items()}
    missing, unexpected = encoder.load_state_dict(sd, strict=False)
    if missing:
        logger.info(f"Missing keys ({len(missing)}): {missing[:5]} ...")
    if unexpected:
        logger.info(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]} ...")
    del ckpt

    return DenseEncoderWrapper(
        encoder=encoder,
        embed_dim=encoder.embed_dim,
        patch_size=patch_size,
        resolution=resolution,
        num_levels=len(out_layers),
        **wrapper_kwargs,
    )


# ---------------------------------------------------------------------------
# Decoder components
# ---------------------------------------------------------------------------

class LateralBlock(nn.Module):
    """1×1 projection + 3×3 refinement, operating on 2-D spatial grids."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)          # token-space projection
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, tokens: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, embed_dim]  (N = H*W spatial tokens)
            H, W:   spatial grid dimensions
        Returns:
            [B, out_dim, H, W]
        """
        x = self.proj(tokens)                           # [B, N, out_dim]
        x = x.reshape(x.size(0), H, W, -1).permute(0, 3, 1, 2)  # [B, out_dim, H, W]
        return self.refine(x)


class FPNDecoder(nn.Module):
    """
    Lightweight Feature Pyramid Network that fuses 4 hierarchical feature
    maps from the V-JEPA 2.1 encoder into a single dense representation,
    then upsamples to pixel resolution.
    """

    def __init__(self, embed_dim: int, fpn_dim: int, num_levels: int,
                 num_classes: int, patch_size: int):
        super().__init__()
        self.patch_size = patch_size

        # Lateral projections: one per encoder level
        self.laterals = nn.ModuleList(
            [LateralBlock(embed_dim, fpn_dim) for _ in range(num_levels)]
        )

        # Top-down merging convolutions (for num_levels > 1)
        self.merges = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
                for _ in range(num_levels - 1)
            ]
        )

        # Final segmentation head: upsample patch-grid → pixel space
        self.seg_head = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=patch_size, mode="bilinear", align_corners=False),
            nn.Conv2d(fpn_dim // 2, num_classes, 1),
        )

    def forward(self, features: list, H: int, W: int) -> torch.Tensor:
        """
        Args:
            features: list of [B, N, embed_dim] tensors, coarse → fine
            H, W:     spatial grid size (num_patches along each axis)
        Returns:
            [B, num_classes, H*patch_size, W*patch_size]
        """
        # Project each level into FPN feature maps
        maps = [lat(f, H, W) for lat, f in zip(self.laterals, features)]

        # Top-down merging: start from the finest (last) level
        out = maps[-1]
        for i in range(len(maps) - 2, -1, -1):
            coarse = F.interpolate(maps[i], size=out.shape[-2:],
                                   mode="bilinear", align_corners=False)
            out = self.merges[i](out + coarse)

        return self.seg_head(out)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class DenseEncoderWrapper(nn.Module):
    """
    Wraps the frozen V-JEPA 2.1 encoder together with a trainable FPN
    segmentation decoder.

    forward() accepts an image tensor [B, C, H, W] and returns per-pixel
    class logits [B, num_classes, H, W].

    Only `laterals`, `merges`, and `seg_head` (inside FPNDecoder) have
    requires_grad=True; the encoder is kept frozen by models.py.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        patch_size: int,
        resolution: int,
        num_classes: int = 150,     # ADE20K default; override via config
        fpn_dim: int = 256,
        num_levels: int = 4,
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.H_patches = resolution // patch_size
        self.W_patches = resolution // patch_size

        self.decoder = FPNDecoder(
            embed_dim=embed_dim,
            fpn_dim=fpn_dim,
            num_levels=num_levels,
            num_classes=num_classes,
            patch_size=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]  — normalized image at training resolution
        Returns:
            logits: [B, num_classes, H, W]
        """
        # V-JEPA 2.1 encoder with out_layers returns a list of feature maps.
        # Input is 4-D; the encoder detects image mode via img_temporal_dim_size.
        features = self.encoder(x)      # list of num_levels × [B, N, D]

        return self.decoder(features, self.H_patches, self.W_patches)
