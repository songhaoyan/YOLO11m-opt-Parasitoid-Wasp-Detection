"""Wise-IoU v3 reconstruction for recent Ultralytics YOLO11 BboxLoss.

This is a *chosen* implementation (alpha=1.9, delta=3, EMA=0.01), not recovered
original parameters. The class is at module scope so Torch checkpoints can pickle
and reload it when this project is importable.
"""
from __future__ import annotations
import torch
from ultralytics.utils.loss import BboxLoss
from ultralytics.utils.tal import bbox2dist


def wiou_v3_loss(pred: torch.Tensor, target: torch.Tensor,
                 running: torch.Tensor, alpha: float = 1.9,
                 delta: float = 3.0, eps: float = 1e-7):
    """Per-box Wise-IoU-v3 on xyxy boxes, and detached per-box IoU loss."""
    x1, y1, x2, y2 = pred.unbind(-1)
    a1, b1, a2, b2 = target.unbind(-1)
    inter = ((torch.minimum(x2, a2) - torch.maximum(x1, a1)).clamp(min=0)
             * (torch.minimum(y2, b2) - torch.maximum(y1, b1)).clamp(min=0))
    area_pred = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area_gt = (a2 - a1).clamp(min=0) * (b2 - b1).clamp(min=0)
    iou = inter / (area_pred + area_gt - inter + eps)
    iou_loss = 1.0 - iou
    cw = (torch.maximum(x2, a2) - torch.minimum(x1, a1)).detach()
    ch = (torch.maximum(y2, b2) - torch.minimum(y1, b1)).detach()
    dist2 = ((x1 + x2 - a1 - a2) * 0.5).square() + ((y1 + y2 - b1 - b2) * 0.5).square()
    distance_factor = torch.exp(dist2 / (cw.square() + ch.square() + eps))
    beta = iou_loss.detach() / running.clamp(min=eps)
    gain = beta / (delta * torch.pow(torch.as_tensor(alpha, device=pred.device, dtype=pred.dtype), beta - delta))
    return gain.detach() * distance_factor * iou_loss, iou_loss.detach()


class WIoUv3BboxLoss(BboxLoss):
    """Ultralytics drop-in criterion: WIoU-v3 BBR and unchanged DFL."""
    alpha: float = 1.9
    delta: float = 3.0
    momentum: float = 0.01

    def __init__(self, reg_max: int = 16):
        super().__init__(reg_max)
        self.register_buffer("mean_iou_loss", torch.tensor(1.0))

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask, imgsz, stride):
        fg_idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[fg_idx].sum(-1, keepdim=True)
        wiou, iou_loss = wiou_v3_loss(
            pred_bboxes[fg_idx].float(), target_bboxes[fg_idx].float(),
            self.mean_iou_loss, self.alpha, self.delta
        )
        if self.training and iou_loss.numel():
            with torch.no_grad():
                self.mean_iou_loss.lerp_(iou_loss.mean().to(self.mean_iou_loss.device), self.momentum)
        loss_iou = (wiou.unsqueeze(-1) * weight).sum() / target_scores_sum
        if not self.dfl_loss:
            raise NotImplementedError("This adapter requires reg_max>1 (standard YOLO11).")
        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
        ld = self.dfl_loss(pred_dist[fg_idx].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_idx])
        return loss_iou, (ld * weight).sum() / target_scores_sum
