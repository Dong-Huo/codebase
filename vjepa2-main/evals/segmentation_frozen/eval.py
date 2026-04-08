# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Semantic segmentation probe on top of a frozen V-JEPA 2.1 encoder.

Training pipeline
-----------------
  1. Load frozen V-JEPA 2.1 encoder + randomly-initialized FPN decoder.
  2. For each epoch, iterate over (image, mask) pairs.
  3. Forward: encoder(image) -> multi-scale tokens -> decoder -> logits [B, C, H, W].
  4. Loss: pixel-wise cross-entropy (unlabeled pixels are ignored via ignore_index).
  5. Metrics: mean Intersection-over-Union (mIoU) over all classes.

Entry point
-----------
  python -m evals.main --fname configs/eval/vjepa2_1/segmentation.yaml --devices cuda:0
"""

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from evals.segmentation_frozen.dataset import make_segmentation_dataloader
from evals.segmentation_frozen.models import init_module
from evals.segmentation_frozen.visualize import get_palette, save_seg_visualizations
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.logging import AverageMeter, CSVLogger, get_logger

logger = get_logger("segmentation_frozen")

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


# ---------------------------------------------------------------------------
# mIoU metric
# ---------------------------------------------------------------------------

class SegmentationMetrics:
    """Accumulates per-class IoU across batches."""

    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.intersection = torch.zeros(self.num_classes, dtype=torch.long)
        self.union = torch.zeros(self.num_classes, dtype=torch.long)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        preds:   [B, num_classes, H, W] logits
        targets: [B, H, W] integer class labels
        """
        pred_labels = preds.argmax(dim=1)               # [B, H, W]
        valid = targets != self.ignore_index
        pred_labels = pred_labels[valid]
        targets = targets[valid]

        for c in range(self.num_classes):
            pred_c = pred_labels == c
            tgt_c = targets == c
            self.intersection[c] += (pred_c & tgt_c).sum().cpu()
            self.union[c] += (pred_c | tgt_c).sum().cpu()

    def miou(self) -> float:
        iou = self.intersection.float() / (self.union.float() + 1e-6)
        valid_cls = self.union > 0
        return iou[valid_cls].mean().item() * 100.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args_eval, resume_preempt=False):

    # -----------------------------------------------------------------------
    # Parse config
    # -----------------------------------------------------------------------
    pretrain_folder = args_eval.get("folder", None)
    resume_checkpoint = args_eval.get("resume_checkpoint", False) or resume_preempt
    eval_tag = args_eval.get("tag", None)
    num_workers = args_eval.get("num_workers", 8)
    val_only = args_eval.get("val_only", False)

    args_pretrain = args_eval["model_kwargs"]
    checkpoint = args_pretrain["checkpoint"]
    module_name = args_pretrain["module_name"]
    pretrain_kwargs = args_pretrain["pretrain_kwargs"]
    wrapper_kwargs = args_pretrain.get("wrapper_kwargs", {})

    args_exp = args_eval["experiment"]

    args_data = args_exp["data"]
    data_root = args_data["root"]
    dataset_name = args_data.get("dataset_name", "generic")
    train_split = args_data.get("train_split", None)   # required for generic
    val_split = args_data.get("val_split", None)       # required for generic
    num_classes = args_data["num_classes"]
    resolution = args_data.get("resolution", 384)
    ignore_index = args_data.get("ignore_index", 255)

    args_opt = args_exp["optimization"]
    batch_size = args_opt.get("batch_size", 16)
    num_epochs = args_opt.get("num_epochs", 50)
    base_lr = args_opt.get("lr", 1e-3)
    weight_decay = args_opt.get("weight_decay", 1e-4)
    warmup_epochs = args_opt.get("warmup_epochs", 5)
    use_bfloat16 = args_opt.get("use_bfloat16", True)
    # -----------------------------------------------------------------------

    # evals/main.py already spawns one process per GPU and calls
    # init_distributed() before reaching here — do not call set_start_method
    # or init_distributed() again; just read the already-initialised state.
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    device = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)

    # Output paths
    out_dir = os.path.join(pretrain_folder, "segmentation_frozen")
    if eval_tag is not None:
        out_dir = os.path.join(out_dir, eval_tag)
    os.makedirs(out_dir, exist_ok=True)
    log_file = os.path.join(out_dir, f"log_r{rank}.csv")
    latest_path = os.path.join(out_dir, "latest.pt")

    if rank == 0:
        csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%.4f", "train_loss"),
            ("%.2f", "val_miou"),
        )

    # -----------------------------------------------------------------------
    # Visualization palette  (built once, reused every epoch)
    # -----------------------------------------------------------------------
    vis_dir = os.path.join(out_dir, "vis")
    palette = get_palette(dataset_name, num_classes)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = init_module(
        module_name=module_name,
        device=device,
        resolution=resolution,
        checkpoint=checkpoint,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=dict(**wrapper_kwargs),
    )
    # Only wrap with DDP when the process group was actually initialised.
    # Falls back to plain nn.Module for single-GPU / CPU runs.
    if dist.is_initialized():
        model = DistributedDataParallel(model, find_unused_parameters=False)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    train_loader, train_sampler = make_segmentation_dataloader(
        dataset_name=dataset_name,
        root=data_root,
        split_file=train_split,
        resolution=resolution,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True,
        num_workers=num_workers,
        ignore_index=ignore_index,
        drop_last=True,
    )
    val_loader, _ = make_segmentation_dataloader(
        dataset_name=dataset_name,
        root=data_root,
        split_file=val_split,
        resolution=resolution,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False,
        num_workers=num_workers,
        ignore_index=ignore_index,
        drop_last=False,
    )
    ipe = len(train_loader)
    logger.info(f"Iterations per epoch: {ipe}")

    # -----------------------------------------------------------------------
    # Optimizer  (only decoder parameters are trainable)
    # -----------------------------------------------------------------------
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=base_lr, weight_decay=weight_decay
    )
    scheduler = _build_cosine_scheduler(
        optimizer, warmup_epochs=warmup_epochs, total_epochs=num_epochs,
        iterations_per_epoch=ipe,
    )
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

    # -----------------------------------------------------------------------
    # Resume
    # -----------------------------------------------------------------------
    start_epoch = 0
    if resume_checkpoint and os.path.exists(latest_path):
        start_epoch = _load_checkpoint(latest_path, model, optimizer, scaler, device)
        for _ in range(start_epoch * ipe):
            scheduler.step()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    for epoch in range(start_epoch, num_epochs):
        train_sampler.set_epoch(epoch)

        if val_only:
            train_loss = -1.0
        else:
            train_loss = _run_epoch(
                model=model, loader=train_loader, optimizer=optimizer,
                scheduler=scheduler, scaler=scaler, device=device,
                num_classes=num_classes, ignore_index=ignore_index,
                training=True, use_bfloat16=use_bfloat16,
            )

        val_miou = _run_epoch(
            model=model, loader=val_loader, optimizer=optimizer,
            scheduler=scheduler, scaler=scaler, device=device,
            num_classes=num_classes, ignore_index=ignore_index,
            training=False, use_bfloat16=use_bfloat16,
            vis_dir=vis_dir if rank == 0 else None,
            palette=palette, epoch=epoch + 1,
        )

        logger.info(
            "[%4d] train_loss=%.4f  val_mIoU=%.2f%%"
            % (epoch + 1, train_loss, val_miou)
        )
        if rank == 0:
            csv_logger.log(epoch + 1, train_loss, val_miou)
            # Save only the decoder — the encoder is frozen and is always
            # reloaded from the pretrained backbone checkpoint.
            torch.save(
                {
                    "decoder": _get_decoder(model).state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict() if scaler else None,
                    "epoch": epoch + 1,
                },
                latest_path,
            )

        if val_only:
            return


# ---------------------------------------------------------------------------
# Epoch runner
# ---------------------------------------------------------------------------

def _get_decoder(model):
    """Return the decoder module regardless of DDP wrapping."""
    m = model.module if hasattr(model, "module") else model
    return m.decoder


def _run_epoch(
    model, loader, optimizer, scheduler, scaler, device,
    num_classes, ignore_index, training, use_bfloat16,
    vis_dir=None, palette=None, epoch=None,
):
    # Set decoder to the correct mode; encoder stays in eval() always
    # (it is frozen — train mode would enable dropout / update BN stats).
    _get_decoder(model).train(mode=training)
    m = model.module if hasattr(model, "module") else model
    m.encoder.eval()

    loss_meter = AverageMeter()
    metrics = SegmentationMetrics(num_classes=num_classes, ignore_index=ignore_index)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

    for itr, (images, masks) in enumerate(loader):
        if training:
            scheduler.step()

        images = images.to(device, non_blocking=True)   # [B, 3, H, W]
        masks = masks.to(device, non_blocking=True)     # [B, H, W]

        amp_dtype = torch.bfloat16 if use_bfloat16 else torch.float32
        with torch.cuda.amp.autocast(enabled=use_bfloat16, dtype=amp_dtype):
            with torch.set_grad_enabled(training):
                logits = model(images)                  # [B, C, H, W]
                # Resize logits to match mask resolution (in case of rounding)
                if logits.shape[-2:] != masks.shape[-2:]:
                    logits = F.interpolate(
                        logits, size=masks.shape[-2:],
                        mode="bilinear", align_corners=False,
                    )
                loss = criterion(logits, masks)

        if training:
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        with torch.no_grad():
            loss_meter.update(loss.item())
            metrics.update(logits.detach(), masks)

        # Save visualizations on the first batch of each validation epoch
        if not training and itr == 0 and vis_dir is not None:
            save_seg_visualizations(
                out_dir=vis_dir,
                epoch=epoch,
                images=images,
                gt_masks=masks,
                pred_logits=logits.detach(),
                palette=palette,
                ignore_index=ignore_index,
            )

        if itr % 1 == 0:
            logger.info(
                "[%5d] loss=%.4f  [mem=%.0fMB]"
                % (itr, loss_meter.avg,
                   torch.cuda.max_memory_allocated() / 1024**2)
            )

    if training:
        return loss_meter.avg
    else:
        return metrics.miou()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                             iterations_per_epoch):
    warmup_iters = warmup_epochs * iterations_per_epoch
    total_iters = total_epochs * iterations_per_epoch

    def lr_lambda(step):
        if step < warmup_iters:
            return step / max(1, warmup_iters)
        progress = (step - warmup_iters) / max(1, total_iters - warmup_iters)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _load_checkpoint(path, model, optimizer, scaler, device):
    ckpt = robust_checkpoint_loader(path, map_location=device)
    _get_decoder(model).load_state_dict(ckpt["decoder"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    epoch = ckpt.get("epoch", 0)
    logger.info(f"Resumed from epoch {epoch}")
    return epoch
