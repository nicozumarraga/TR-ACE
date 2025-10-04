import torch
import torch.nn.functional as F

def compute_infonce_loss(scores, labels, temperature=0.07):
    """
    Compute InfoNCE (Normalized Temperature-scaled Cross Entropy) loss for chunk ranking.

    InfoNCE maximizes similarity between query and positive chunks while minimizing
    similarity with negative chunks using a contrastive objective.

    Args:
        scores: [batch, num_chunks] - similarity scores (dot product or cosine)
        labels: [batch, num_chunks] - binary labels (1=positive, 0=negative or padding)
        temperature: Temperature scaling factor (default: 0.07)

    Returns:
        loss: scalar tensor
    """
    batch_size, num_chunks = scores.shape

    # Create mask for valid chunks (non-padding)
    valid_mask = (labels == 1) | (scores.abs() > 1e-6)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)

    # Scale scores by temperature
    scores = scores / temperature

    # For each sample in batch, compute InfoNCE loss
    losses = []

    for i in range(batch_size):
        sample_scores = scores[i]  # [num_chunks]
        sample_labels = labels[i]  # [num_chunks]
        sample_valid = valid_mask[i]  # [num_chunks]

        # Get positive and negative indices
        positive_mask = (sample_labels == 1) & sample_valid
        negative_mask = (sample_labels == 0) & sample_valid

        # Skip if no positives or no negatives
        if not positive_mask.any() or not negative_mask.any():
            continue

        # Get scores for positives and negatives
        pos_scores = sample_scores[positive_mask]  # [num_positives]
        neg_scores = sample_scores[negative_mask]  # [num_negatives]

        # For each positive, compute loss against all negatives
        # InfoNCE: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        for pos_score in pos_scores:
            # Numerator: exp(positive score)
            numerator = torch.exp(pos_score)

            # Denominator: exp(positive) + sum(exp(negatives))
            denominator = numerator + torch.exp(neg_scores).sum()

            # -log(numerator / denominator)
            loss = -torch.log(numerator / denominator)
            losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)

    return torch.stack(losses).mean()


def compute_infonce_loss_efficient(scores, labels, temperature=0.07):
    """
    Efficient vectorized InfoNCE loss (alternative implementation).

    Uses logsumexp for numerical stability and vectorized operations.

    Args:
        scores: [batch, num_chunks] - similarity scores
        labels: [batch, num_chunks] - binary labels (1=positive, 0=negative)
        temperature: Temperature scaling factor

    Returns:
        loss: scalar tensor
    """
    batch_size, num_chunks = scores.shape

    # Create mask for valid chunks
    valid_mask = (labels == 1) | (scores.abs() > 1e-6)

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)

    # Scale by temperature
    scores = scores / temperature

    # Mask out invalid chunks with very negative values
    masked_scores = scores.clone()
    masked_scores[~valid_mask] = -1e9

    # Create positive and negative masks
    positive_mask = (labels == 1) & valid_mask
    negative_mask = (labels == 0) & valid_mask

    losses = []

    for i in range(batch_size):
        pos_mask_i = positive_mask[i]
        neg_mask_i = negative_mask[i]

        if not pos_mask_i.any() or not neg_mask_i.any():
            continue

        pos_scores_i = masked_scores[i][pos_mask_i]
        neg_scores_i = masked_scores[i][neg_mask_i]

        # For each positive: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        # = -pos + log(exp(pos) + sum(exp(neg)))
        # = -pos + logsumexp([pos, neg_1, neg_2, ...])

        for pos_score in pos_scores_i:
            # Stack positive with all negatives
            all_scores = torch.cat([pos_score.unsqueeze(0), neg_scores_i])

            # InfoNCE: -log(softmax(pos))
            # = -pos + logsumexp(all_scores)
            loss = -pos_score + torch.logsumexp(all_scores, dim=0)
            losses.append(loss)

    if len(losses) == 0:
        return torch.tensor(0.0, requires_grad=True, device=scores.device)

    return torch.stack(losses).mean()
