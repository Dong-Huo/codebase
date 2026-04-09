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
     using a lightweight FPN decoder.

Upsampling
----------
Two modes are available (controlled by `use_anyup` in wrapper_kwargs):

  use_anyup=False (default)
      Bilinear upsampling inside the seg_head (patch_size × scale factor).

  use_anyup=True
      AnyUp (wimmerth/anyup) — a learned feature upsampler that takes the
      original image as a guide.  The FPN output at patch-grid resolution is
      passed to AnyUp together with the input image; AnyUp produces features
      at full image resolution which are then projected to class logits.
      Install: pip install anyup
               (optionally: pip install natten for faster windowed attention)

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
        if depth is None:
            depth = {"vit_base": 12, "vit_large": 24, "vit_giant_xformers": 40,
                     "vit_gigantic_xformers": 48}.get(model_name, 24)
        out_layers = _HIER_LAYERS[depth]
    logger.info(f"Using out_layers={out_layers} for dense feature extraction")

    # num_frames must match the pretraining value (default 64) so that
    # PatchEmbed3D is built with the right 3-D conv shape [C, 3, tubelet, 16, 16]
    # and the checkpoint loads without shape mismatch.
    #
    # For image inputs the encoder does NOT use patch_embed — it detects
    # shape[2] == img_temporal_dim_size (=1) and routes through patch_embed_img
    # instead (tubelet_size=1).  So num_frames here only affects model
    # construction / weight loading, not inference behaviour.
    num_frames = enc_kwargs.get("num_frames", 64)
    tubelet_size = enc_kwargs.get("tubelet_size", 2)

    # Build encoder — out_layers activates multi-scale output mode
    build_kwargs = dict(
        img_size=(resolution, resolution),
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
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
    """
    1×1 channel projection + 3×3 spatial refinement.

    Accepts pre-upsampled spatial feature maps [B, in_dim, H, W] — the
    token→spatial reshape has already been done (and upsampling applied)
    before this block is called.
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel_size=1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, in_dim, H, W]  — already at target spatial resolution
        Returns:
            [B, out_dim, H, W]
        """
        return self.refine(self.proj(x))


class FPNDecoder(nn.Module):
    """
    Lightweight FPN that fuses hierarchical feature maps already upsampled to
    full image resolution, producing a single dense feature map for classification.

    Input:  list of num_levels × [B, embed_dim, H, W]  (all at image resolution)
    Output: [B, fpn_dim//2, H, W]
    """

    def __init__(self, embed_dim: int, fpn_dim: int, num_levels: int):
        super().__init__()

        self.laterals = nn.ModuleList(
            [LateralBlock(embed_dim, fpn_dim) for _ in range(num_levels)]
        )

        # Top-down merging convolutions
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

        self.out_conv = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_dim // 2),
            nn.ReLU(inplace=True),
        )
        self.out_dim = fpn_dim // 2

    def forward(self, features: list) -> torch.Tensor:
        """
        Args:
            features: list of [B, embed_dim, H, W], coarse → fine,
                      all already at the same (full image) resolution
        Returns:
            [B, fpn_dim//2, H, W]
        """
        maps = [lat(f) for lat, f in zip(self.laterals, features)]

        out = maps[-1]
        for i in range(len(maps) - 2, -1, -1):
            out = self.merges[i](out + maps[i])

        return self.out_conv(out)


# ---------------------------------------------------------------------------
# Main wrapper
# ---------------------------------------------------------------------------

class DenseEncoderWrapper(nn.Module):
    """
    Wraps the frozen V-JEPA 2.1 encoder together with a trainable FPN
    segmentation decoder.

    Upsampling pipeline
    -------------------
    use_anyup=False (default):
        FPN output → bilinear ×patch_size → 1×1 conv → logits

    use_anyup=True:
        FPN output (patch grid) + original image →
            AnyUp (guided, learned) → 1×1 conv → logits

    Only decoder parameters (FPNDecoder + classifier) have requires_grad=True.
    The encoder is frozen by models.py after construction.
    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        patch_size: int,
        resolution: int,
        num_classes: int = 150,
        fpn_dim: int = 256,
        num_levels: int = 4,
        use_anyup: bool = False,
        anyup_use_natten: bool = False,     # set True if NATTEN is installed
    ):
        super().__init__()
        self.encoder = encoder
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.H_patches = resolution // patch_size
        self.W_patches = resolution // patch_size
        self.use_anyup = use_anyup

        self.decoder = FPNDecoder(
            embed_dim=embed_dim,
            fpn_dim=fpn_dim,
            num_levels=num_levels,
        )
        feat_dim = self.decoder.out_dim     # fpn_dim // 2

        if use_anyup:
            # AnyUp: guided learned upsampler (wimmerth/anyup).
            # Loaded from PyTorch Hub — weights are downloaded automatically.
            # AnyUp is feature-agnostic: no retraining needed.
            logger.info("Loading AnyUp from PyTorch Hub (wimmerth/anyup)...")
            self.anyup = torch.hub.load(
                "wimmerth/anyup",
                "anyup_multi_backbone",
                use_natten=anyup_use_natten,
                pretrained=True,
            )
            # AnyUp is a fixed pretrained model — freeze its weights.
            for p in self.anyup.parameters():
                p.requires_grad = False
            self.anyup.eval()
            logger.info("AnyUp loaded and frozen.")
        else:
            self.anyup = None
            # Bilinear upsample: simple scale factor
            self.upsample = nn.Upsample(
                scale_factor=patch_size, mode="bilinear", align_corners=False
            )

        # 1×1 classifier: maps FPN features → class logits at pixel resolution
        self.classifier = nn.Conv2d(feat_dim, num_classes, kernel_size=1)

    def _tokens_to_spatial(self, tokens: torch.Tensor) -> torch.Tensor:
        """Reshape [B, N, D] encoder tokens → [B, D, H_patch, W_patch]."""
        B, N, D = tokens.shape
        return tokens.reshape(B, self.H_patches, self.W_patches, D).permute(0, 3, 1, 2)

    def _upsample_level(self, x: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        """
        Upsample one encoder feature map from patch-grid to image resolution.

        Args:
            x:     [B, D, H_patch, W_patch]
            image: [B, 3, H, W]  — original normalized image (AnyUp guide)
        Returns:
            [B, D, H, W]
        """
        if self.use_anyup:
            with torch.no_grad():
                return self.anyup(image, x)         # guided learned upsample
        else:
            return self.upsample(x)                 # bilinear ×patch_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W]  — ImageNet-normalized image at training resolution
        Returns:
            logits: [B, num_classes, H, W]
        """
        # 1. Encode → list of num_levels × [B, N, embed_dim]
        token_features = self.encoder(x)

        # 2. Reshape each level to [B, D, H_patch, W_patch] and upsample to
        #    full image resolution using AnyUp (guided) or bilinear.
        hr_features = [
            self._upsample_level(self._tokens_to_spatial(tok), x)
            for tok in token_features
        ]   # list of num_levels × [B, D, H, W]

        # 3. FPN fusion at full resolution → [B, fpn_dim//2, H, W]
        fused = self.decoder(hr_features)

        # 4. Classify
        return self.classifier(fused)               # [B, num_classes, H, W]
