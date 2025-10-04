import torch

def compute_ranking_metrics(scores, labels):
    """
    Compute ranking metrics for chunk retrieval (optimized for small batches).

    Args:
        scores: [batch, num_chunks] - relevance scores
        labels: [batch, num_chunks] - binary labels (1=positive, 0=negative/padding)

    Returns:
        dict with metrics
    """
    batch_size = scores.size(0)

    # Create mask for valid chunks (ignore padding)
    valid_mask = (labels == 1) | (scores.abs() > 1e-6)

    metrics = {}

    # 1. Ranking Metrics
    # Precision@1: Is the top-scored chunk positive?
    top_indices = scores.argmax(dim=1)  # [batch]
    top_is_positive = labels[torch.arange(batch_size), top_indices]
    metrics['precision@1'] = top_is_positive.float().mean().item()

    # Mean Reciprocal Rank (MRR)
    reciprocal_ranks = []
    for i in range(batch_size):
        # Get valid chunks for this sample
        valid = valid_mask[i]
        if not valid.any():
            continue

        # Sort by scores (descending)
        sorted_indices = scores[i][valid].argsort(descending=True)
        valid_labels = labels[i][valid]

        # Find rank of first positive
        for rank, idx in enumerate(sorted_indices, start=1):
            if valid_labels[idx] == 1:
                reciprocal_ranks.append(1.0 / rank)
                break

    metrics['MRR'] = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # NDCG@10 (handles variable number of positives per sample)
    ndcg_scores = []
    for i in range(batch_size):
        valid = valid_mask[i]
        if not valid.any():
            continue

        # Sort by scores (descending)
        sorted_indices = scores[i][valid].argsort(descending=True)
        valid_labels = labels[i][valid]

        # Get top-k predictions (k=10 or fewer if not enough chunks)
        k = min(10, valid.sum().item())
        top_k_indices = sorted_indices[:k]
        top_k_labels = valid_labels[top_k_indices]

        # DCG: sum of (2^rel - 1) / log2(rank + 1)
        dcg = 0.0
        for rank, label in enumerate(top_k_labels, start=1):
            dcg += (2**label.item() - 1) / torch.log2(torch.tensor(rank + 1.0)).item()

        # IDCG: DCG of perfect ranking
        num_positives = (valid_labels == 1).sum().item()
        if num_positives == 0:
            continue

        ideal_labels = torch.cat([
            torch.ones(min(num_positives, k)),
            torch.zeros(max(0, k - num_positives))
        ])
        idcg = 0.0
        for rank, label in enumerate(ideal_labels, start=1):
            idcg += (2**label.item() - 1) / torch.log2(torch.tensor(rank + 1.0)).item()

        ndcg_scores.append(dcg / idcg if idcg > 0 else 0.0)

    metrics['NDCG@10'] = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0

    # 2. Score-based Metrics
    positive_mask = (labels == 1) & valid_mask
    negative_mask = (labels == 0) & valid_mask

    if positive_mask.any():
        metrics['avg_positive_score'] = scores[positive_mask].mean().item()
    else:
        metrics['avg_positive_score'] = 0.0

    if negative_mask.any():
        metrics['avg_negative_score'] = scores[negative_mask].mean().item()
    else:
        metrics['avg_negative_score'] = 0.0

    # Score margin (higher is better)
    metrics['score_margin'] = metrics['avg_positive_score'] - metrics['avg_negative_score']

    return metrics
