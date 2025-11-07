import torch
import torch.nn.functional as F


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


# def masked_orthogonal_loss(pred, target, mask):
#     error = (target - pred) ** 2
#     M = (pred**2) * error
#     M = M * mask
#     loss = M.sum() / mask.sum().clamp(min=1)
#     return loss


def masked_orthogonal_loss(restored_feature, noise, pred_noise, mask, alpha=0.8):
    """
    Compute orthogonality loss to encourage the restored feature 
    to be orthogonal to both true and predicted noise components.

    restored_feature : [B, N, C]  -- model output (predicted clean feature)
    noise            : [B, N, C]  -- ground-truth noise (noisy_feature - feature)
    pred_noise       : [B, N, C]  -- predicted noise (noisy_feature - restored_feature)
    mask             : [B, N, C]  -- binary mask (1 = valid, 0 = ignore)
    alpha            : float      -- weight between true_noise and pred_noise terms
    """

    # --- 1. Apply mask to focus on valid regions only ---
    restored_feature = restored_feature * mask
    noise = noise * mask
    pred_noise = pred_noise * mask

    # --- 2. Subtract mean (zero-center each batch sample along spatial dim N) ---
    #     → removes global bias so cosine correlation measures shape, not mean offset
    restored_feature = restored_feature - restored_feature.mean(dim=1, keepdim=True)
    noise = noise - noise.mean(dim=1, keepdim=True)
    pred_noise = pred_noise - pred_noise.mean(dim=1, keepdim=True)

    # --- 3. Normalize along channel dimension ---
    #     → turns each vector into a unit vector, so dot product ≈ cosine similarity
    restored_feature = F.normalize(restored_feature, dim=2)
    noise = F.normalize(noise, dim=2)
    pred_noise = F.normalize(pred_noise, dim=2)

    # --- 4. Compute cosine correlation with true noise ---
    #     → measures alignment between restored feature and actual noise
    corr_true = (restored_feature * noise).sum(dim=2)  # [B, N]
    loss_true = (corr_true ** 2).mean()                # squared → penalize correlation magnitude

    # --- 5. Compute cosine correlation with predicted noise ---
    #     → ensures model’s internal “noise direction” also becomes orthogonal
    corr_pred = (restored_feature * pred_noise).sum(dim=2)  # [B, N]
    loss_pred = (corr_pred ** 2).mean()

    # --- 6. Blend the two orthogonality losses ---
    #     α controls the balance:
    #     α = 1 → only true noise; α = 0 → only predicted noise
    loss = alpha * loss_true + (1 - alpha) * loss_pred

    # --- 7. Return the scalar orthogonality loss ---
    return loss

