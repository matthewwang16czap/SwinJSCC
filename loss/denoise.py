import torch


def masked_mse_loss(pred, target, mask):
    """
    pred:  [B, N, C] restored feature
    target:       [B, N, C] ground-truth feature
    mask:          [B, N, C] 1 for active entries, 0 for masked
    """
    diff = (pred - target) ** 2
    diff = diff * mask
    loss = diff.sum() / mask.sum().clamp(min=1)  # avoid div by 0
    return loss


def masked_orthogonal_loss(pred, target, mask):
    error = (target - pred) ** 2
    M = (target**2) * error  # weight error by true feature energy
    M = M * mask
    loss = M.sum() / mask.sum().clamp(min=1)
    return loss


def denoise_loss_fn(pred, target, mask, alpha=1.0, beta=1.0):
    mse = masked_mse_loss(pred, target, mask)
    orth = masked_orthogonal_loss(pred, target, mask)
    denoise_loss = alpha * mse + beta * orth
    return denoise_loss, mse, orth
