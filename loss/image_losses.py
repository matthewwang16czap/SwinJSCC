from typing import List, Optional
import torch
import lpips
from torchmetrics.image import (
    StructuralSimilarityIndexMeasure as SSIM,
    MultiScaleStructuralSimilarityIndexMeasure as MS_SSIM,
)


def mse_to_psnr(mse: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    return 10.0 * torch.log10(data_range**2 / mse.clamp(min=1e-8))


@torch.jit.script
def masked_mse(
    X: torch.Tensor,
    Y: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Returns per-image MSE. Shape: (B,)

    Args:
        X, Y:       Input tensors (B, C, H, W).
        mask:       Optional float mask (B, H, W) or (B, 1, H, W) in [0, 1].
        data_range: The value range of the output (e.g. 1.0 or 255.0).
        normalized: If True, inputs are already in [0, data_range].
                    If False, inputs are in [-1, 1] and are remapped.
    """
    if not normalized:
        X = (X + 1.0) / 2.0 * data_range
        Y = (Y + 1.0) / 2.0 * data_range
    diff = (X - Y) ** 2
    if mask is None:
        return diff.mean(dim=(1, 2, 3))
    if mask.dim() == 3:
        mask = mask.unsqueeze(1)  # (B, 1, H, W)
    num = (diff * mask).sum(dim=(1, 2, 3))
    den = mask.sum(dim=(1, 2, 3)).clamp_min(1.0)
    return num / den


@torch.jit.ignore
def masked_lpips(
    X: torch.Tensor,
    Y: torch.Tensor,
    model: torch.nn.Module,
    mask: Optional[torch.Tensor] = None,
    data_range: float = 1.0,
    normalized: bool = False,
) -> torch.Tensor:
    """
    Returns per-image LPIPS. Shape: (B,)

    Args:
        X, Y:       Input tensors (B, C, H, W).
        model:      LPIPS model instance.
        mask:       Optional float mask (B, H, W) or (B, 1, H, W) in [0, 1].
        data_range: The value range of the inputs (e.g. 1.0 or 255.0).
        normalized: If True, inputs are in [0, data_range] and are remapped
                    to [-1, 1] for LPIPS. If False, inputs are already [-1, 1].
    """
    if normalized:
        X = (X / data_range) * 2.0 - 1.0
        Y = (Y / data_range) * 2.0 - 1.0
    # normalized=False: inputs are already in [-1, 1], nothing to do.
    score = model(X, Y).view(X.size(0))  # (B,)
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # (B, 1, H, W)
        # Weight each image's score by its mask coverage instead of
        # zeroing pixels, which would corrupt the perceptual features.
        mask_weight = mask.mean(dim=(1, 2, 3))  # (B,)
        score = score * mask_weight
    return score


class LPIPSScore(torch.nn.Module):
    """
    Returns:
        loss:       Scalar LPIPS for backprop (mean over batch).
        lpips_val:  Per-image LPIPS, detached. Shape: (B,)
    """

    def __init__(self, normalized: bool = False, data_range: float = 1.0):
        super().__init__()
        self.normalized = normalized
        self.data_range = data_range
        self.lpips_model = lpips.LPIPS(net="alex")

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        lpips_val = masked_lpips(
            X,
            Y,
            self.lpips_model,
            mask,
            data_range=self.data_range,
            normalized=self.normalized,
        )
        return lpips_val


class ImageLoss(torch.nn.Module):
    """
    output: tuple of G tensors, each (B, 3, H, W)
    target: (B, 3, H, W)
    L_total = Σ_i  w_i * mean(masked_mse(output_i, target))
    """

    def __init__(
        self,
        data_range: float = 1.0,
        normalized: bool = True,
    ):
        super().__init__()
        self.data_range = data_range
        self.normalized = normalized
        # metrics
        self.lpips = LPIPSScore(normalized=normalized, data_range=data_range)
        self.ssim = SSIM(data_range=data_range)
        self.msssim = MS_SSIM(data_range=data_range)

    def forward(
        self,
        output: tuple,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        ralative_importance: Optional[List[float]] = None,
    ):
        losses = torch.stack(
            [
                masked_mse(
                    output_i, target, mask, self.data_range, self.normalized
                ).mean()
                for output_i in output
            ]
        )
        rel_importance = (
            torch.ones_like(losses)
            if ralative_importance is None
            else torch.tensor(
                ralative_importance,
                dtype=losses.dtype,
                device=losses.device,
            )
        )
        loss_sum = (rel_importance * losses).sum()
        # summary
        metrics = {}
        metrics["psnr"] = [
            mse_to_psnr(loss, self.data_range).detach() for loss in losses
        ]
        metrics["lpips"] = [
            self.lpips(output_i, target, mask).mean().detach() for output_i in output
        ]
        metrics["ssim"] = [
            self.ssim(target, output_i).mean().detach() for output_i in output
        ]
        metrics["msssim"] = [
            self.msssim(target, output_i).mean().detach() for output_i in output
        ]
        return loss_sum, metrics
