"""
utils.py
========
Shared helpers for the 2D U-Net prostate segmentation project

Currently contains:
- ``to_probs``          : logits -> soft probabilities (handles binary & multi-class)
- ``dice_score``        : per-class + macro + macro-over-present Dice
- ``iou_score``         : per-class + macro + macro-over-present IoU
- ``dice_loss``         : soft Dice training loss, averaged over present classes only

Design notes
------------
- All functions accept ``from_logits=True`` (default) and internally apply
  ``softmax`` (multi-class) or ``sigmoid`` (binary). This keeps the model
  free of output activation and lets the loss choose the right composition.
- A class that is absent from the target produces a trivially high
  Dice/IoU (because of the smoothing term). This is fine for *reporting*
  (absent classes are correctly recognized as "not to be predicted"), but
  must be excluded from the *loss* — otherwise slices with no prostate
  contribute no penalty and slices with prostate are diluted across
  multiple absent classes. ``dice_loss`` therefore averages over
  present classes only, and ``dice_score`` exposes both ``macro``
  (over all classes) and ``macro_present`` (over present classes only).
- ``smooth=1.0`` is used by default. This is the standard choice for
  binary medical segmentation and prevents NaN when both prediction and
  target are all-zero for a class (they then yield Dice = IoU = 1.0,
  which is the accepted convention).
"""

from __future__ import annotations

import torch


def to_probs(pred: torch.Tensor, from_logits: bool = True) -> torch.Tensor:
    """
    Convert raw logits to soft probabilities.

    - ``pred.shape[1] == 1``: binary segmentation, uses ``sigmoid``.
    - ``pred.shape[1] > 1`` : multi-class segmentation, uses ``softmax(dim=1)``.

    If ``from_logits=False``, the input is returned unchanged (assumed
    already to be probabilities).
    """
    if not from_logits:
        return pred
    if pred.shape[1] == 1:
        return torch.sigmoid(pred)
    return torch.softmax(pred, dim=1)


def _aggregate(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Shared per-channel aggregation step.

    Returns
    -------
    inter      : (C,)  sum over N, H, W of (prob * target) per channel
    pred_sum   : (C,)  sum of probabilities per channel
    target_sum : (C,)  sum of target values per channel

    Raises ``ValueError`` if shapes do not match. Caller must supply a
    one-hot (or soft) target of the same shape as ``pred``.
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have identical shape, "
            f"got pred={tuple(pred.shape)}, target={tuple(target.shape)}"
        )
    probs = to_probs(pred, from_logits=from_logits)
    dims = (0, 2, 3)
    inter = (probs * target).sum(dim=dims)
    pred_sum = probs.sum(dim=dims)
    target_sum = target.sum(dim=dims)
    return inter, pred_sum, target_sum


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    smooth: float = 1.0,
) -> dict:
    """
    Soft Dice per channel plus aggregate summaries.

    Parameters
    ----------
    pred   : (N, C, H, W) logits or probabilities.
    target : (N, C, H, W) one-hot (or soft) target of the same shape.
             For binary segmentation, ``C=1``.
    from_logits : apply sigmoid/softmax before scoring.
    smooth      : numerical stabilizer (default 1.0).

    Returns
    -------
    dict with keys:
      per_class      : list[float]  length C — Dice for each channel
      macro          : float        mean over all C channels
      macro_present  : float        mean over channels with target_sum > 0
                                     (0.0 if no channel is present)
      target_pixels  : list[int]    length C — total target pixels per class
      pred_pixels    : list[int]    length C — total predicted probability mass
    """
    inter, pred_sum, target_sum = _aggregate(pred, target, from_logits=from_logits)
    union = pred_sum + target_sum
    dice = (2.0 * inter + smooth) / (union + smooth)

    present = target_sum > 0
    macro_present = (
        float(dice[present].mean().item()) if present.any() else 0.0
    )

    return {
        "per_class": dice.tolist(),
        "macro": float(dice.mean().item()),
        "macro_present": macro_present,
        "target_pixels": target_sum.long().tolist(),
        "pred_pixels": pred_sum.long().tolist(),
    }


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    smooth: float = 1.0,
) -> dict:
    """
    Soft IoU (Jaccard) per channel plus aggregate summaries.

    Same API and return keys as :func:`dice_score`. The exact
    relationship is ``IoU = Dice / (2 - Dice)`` for binary masks; both
    are reported separately because the spec requires both.
    """
    inter, pred_sum, target_sum = _aggregate(pred, target, from_logits=from_logits)
    union = pred_sum + target_sum - inter
    iou = (inter + smooth) / (union + smooth)

    present = target_sum > 0
    macro_present = (
        float(iou[present].mean().item()) if present.any() else 0.0
    )

    return {
        "per_class": iou.tolist(),
        "macro": float(iou.mean().item()),
        "macro_present": macro_present,
        "target_pixels": target_sum.long().tolist(),
        "pred_pixels": pred_sum.long().tolist(),
    }


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    from_logits: bool = True,
    smooth: float = 1.0,
    class_weights: torch.Tensor | None = None,
    ignore_absent: bool = True,
) -> torch.Tensor:
    """
    Soft Dice loss = 1 - Dice, averaged over present classes.

    Parameters
    ----------
    pred, target, from_logits : as in :func:`dice_score`.
    smooth        : numerical stabilizer (default 1.0).
    class_weights : optional (C,) tensor of per-class weights
                    (e.g. ``[1.0, 0.5, 1.0, 2.0, 3.0, 5.0]`` to up-weight
                    the rarer rectum and prostate). Weights are applied
                    *after* the "1 - Dice" step.
    ignore_absent : if True (default) channels with ``target_sum == 0``
                    are excluded from the mean. This is important for
                    the HipMRI data, where most coronal slices contain
                    no prostate at all — without it, those slices would
                    dilute the prostate contribution to near zero.

    Returns
    -------
    scalar ``torch.Tensor`` (0-dim, differentiable w.r.t. ``pred``).
    """
    inter, pred_sum, target_sum = _aggregate(pred, target, from_logits=from_logits)
    union = pred_sum + target_sum
    dice = (2.0 * inter + smooth) / (union + smooth)
    loss_per_class = 1.0 - dice

    if class_weights is not None:
        if class_weights.shape != loss_per_class.shape:
            raise ValueError(
                f"class_weights must have shape {tuple(loss_per_class.shape)}, "
                f"got {tuple(class_weights.shape)}"
            )
        loss_per_class = loss_per_class * class_weights.to(loss_per_class.device)

    if not ignore_absent:
        return loss_per_class.mean()

    present = (target_sum > 0).to(loss_per_class.dtype)
    denom = present.sum()
    if denom.item() == 0:
        # No class is present — every pixel is "ignore" (should not happen
        # in practice with a background class, but guard defensively).
        return loss_per_class.sum() * 0.0
    return (loss_per_class * present).sum() / denom
