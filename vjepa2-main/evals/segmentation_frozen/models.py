# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


def init_module(
    module_name: str,
    device,
    resolution: int,
    checkpoint: str,
    model_kwargs: dict,
    wrapper_kwargs: dict,
    frames_per_clip: int = 1,
):
    """
    Build the (frozen encoder + trainable decoder) model and move to device.

    The module at `module_name` must expose:
        init_module(resolution, checkpoint, model_kwargs, wrapper_kwargs)
        -> nn.Module  with forward(x: [B,C,H,W]) -> [B, num_classes, H, W]

    The encoder backbone is frozen here; only the decoder parameters remain
    trainable so they can be optimized during the probe-training phase.
    """
    model = (
        importlib.import_module(module_name)
        .init_module(
            resolution=resolution,
            checkpoint=checkpoint,
            model_kwargs=model_kwargs,
            wrapper_kwargs=wrapper_kwargs,
            frames_per_clip=frames_per_clip,
        )
        .to(device)
    )

    # Freeze the encoder backbone; decoder (laterals + seg_head) stays trainable
    for p in model.encoder.parameters():
        p.requires_grad = False
    model.encoder.eval()

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.encoder.parameters())
    logger.info(f"Encoder frozen  : {n_frozen:,} parameters")
    logger.info(f"Decoder trainable: {n_trainable:,} parameters")

    return model
