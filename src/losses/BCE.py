import torch
from torch.nn import BCEWithLogitsLoss

def compute_bce_loss(scores, labels):
    """
    Compute BCE loss for chunk ranking, ignoring padded chunks.

    Args:
        scores: [batch, num_chunks] - raw logits from model
        labels: [batch, num_chunks] - binary labels (1=positive, 0=negative or padding)

    Returns:
        loss: scalar tensor
    """
    # Create mask for valid chunks (non-padding)
    # Padding chunks have both label=0 and score~0, so we need to identify real negatives
    # Real chunks have either label=1 OR non-zero scores after forward pass
    valid_mask = (labels == 1) | (scores.abs() > 1e-6)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True)

    # Use BCEWithLogitsLoss (combines sigmoid + BCE for numerical stability)
    criterion = BCEWithLogitsLoss(reduction='none')
    loss_per_chunk = criterion(scores, labels)

    # Only compute loss on valid (non-padded) chunks
    loss = (loss_per_chunk * valid_mask).sum() / valid_mask.sum()

    return loss
