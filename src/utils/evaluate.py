"""
Full-tender retrieval evaluation.

Evaluates model performance on complete tenders (not sampled batches).
Computes Recall@K metrics to measure retrieval quality.
"""
import torch
import logging
from typing import List, Dict
from tqdm import tqdm


def evaluate_full_tender_retrieval(
    model,
    tenders: List[Dict],
    tokenizer,
    device: str = "cpu",
    max_len: int = 256,
    k_values: List[int] = [1, 3, 5, 10],
    logger: logging.Logger = None,
    negative_multiplier: int = 10
) -> Dict[str, float]:
    """
    Evaluate retrieval performance on complete tenders.

    For each tender:
    1. Score sampled chunks (all positives + negative_multiplier * negatives)
    2. Rank chunks by score
    3. Check how many positives appear in top-K

    Args:
        model: LearnableQueryRanker instance
        tenders: List of tender dicts with 'chunks' and 'positives'
        tokenizer: HuggingFace tokenizer
        device: "cpu" or "cuda"
        max_len: max sequence length
        k_values: List of K values for Recall@K
        logger: Optional logger
        negative_multiplier: Number of negatives per positive to sample (e.g., 10x)

    Returns:
        Dict with Recall@K metrics
    """
    import random

    if logger is None:
        logger = logging.getLogger(__name__)

    model.eval()

    # Track recall for each tender
    recall_at_k = {k: [] for k in k_values}

    with torch.no_grad():
        for tender in tqdm(tenders, desc="Full-Tender Eval", leave=False):
            # Get all chunks
            chunks = tender.get("chunks", [])
            positives = set(tender.get("positives", []))

            if len(chunks) == 0 or len(positives) == 0:
                continue

            # Extract chunk texts and IDs
            chunk_texts = [c["text"] for c in chunks]
            chunk_ids = [c["metadata"]["chunk_id"] for c in chunks]

            # Filter positives that actually exist in chunks
            valid_positives = [pid for pid in positives if pid in chunk_ids]
            if len(valid_positives) == 0:
                continue

            # Sample chunks: all positives + N * negatives
            positive_indices = [i for i, cid in enumerate(chunk_ids) if cid in valid_positives]
            negative_indices = [i for i, cid in enumerate(chunk_ids) if cid not in valid_positives]

            # Sample negatives
            num_negatives_to_sample = min(
                len(valid_positives) * negative_multiplier,
                len(negative_indices)
            )
            sampled_negative_indices = random.sample(negative_indices, num_negatives_to_sample) if num_negatives_to_sample > 0 else []

            # Combine positives and sampled negatives
            sampled_indices = positive_indices + sampled_negative_indices
            sampled_chunk_texts = [chunk_texts[i] for i in sampled_indices]
            sampled_chunk_ids = [chunk_ids[i] for i in sampled_indices]

            # Tokenize sampled chunks
            encodings = tokenizer(
                sampled_chunk_texts,
                truncation=True,
                padding="max_length",
                max_length=max_len,
                return_tensors="pt"
            )

            chunk_input_ids = encodings["input_ids"].to(device)
            chunk_attention_mask = encodings["attention_mask"].to(device)

            # Score sampled chunks
            # Model expects [batch, num_chunks, seq_len]
            # We have [num_chunks, seq_len], so add batch dimension
            scores = model(
                chunk_input_ids.unsqueeze(0),
                chunk_attention_mask.unsqueeze(0)
            )  # [1, num_chunks]

            scores = scores.squeeze(0)  # [num_chunks]

            # Rank chunks by score (descending)
            sorted_indices = scores.argsort(descending=True).cpu().numpy()

            # For each K, compute recall
            for k in k_values:
                top_k_indices = sorted_indices[:k]
                top_k_chunk_ids = [sampled_chunk_ids[i] for i in top_k_indices]

                # How many positives are in top-K?
                positives_in_top_k = len([cid for cid in top_k_chunk_ids if cid in valid_positives])

                # Recall = (positives retrieved) / (total positives)
                recall = positives_in_top_k / len(valid_positives)
                recall_at_k[k].append(recall)

    # Average recall across all tenders
    avg_recall = {}
    for k in k_values:
        if len(recall_at_k[k]) > 0:
            avg_recall[f"Recall@{k}"] = sum(recall_at_k[k]) / len(recall_at_k[k])
        else:
            avg_recall[f"Recall@{k}"] = 0.0

    return avg_recall


if __name__ == "__main__":
    """
    Standalone evaluation script.
    Usage: python evaluate.py
    """
    import json
    from pathlib import Path
    from transformers import AutoTokenizer
    from src.models.chunk_encoder import LearnableQueryRanker
    from src.utils.logger import setup_logger

    # Configuration
    CATEGORY = "5. CRITERIOS DE ADJUDICACIÓN"
    CATEGORY_SLUG = f"section_{CATEGORY.split('.')[0]}"
    MODEL_PATH = f"ranker_{CATEGORY_SLUG}.pt"
    PREPROC_PATH = f"data/1.pre-processed/{CATEGORY_SLUG}"
    MODEL_NAME = "distilbert-base-uncased"
    DEVICE = "cpu"
    MAX_LENGTH = 256

    # Setup
    logger = setup_logger("evaluation", level=logging.INFO)

    # Load tenders
    tender_files = list(Path(PREPROC_PATH).glob("*.json"))
    logger.info(f"Loading {len(tender_files)} tenders from {PREPROC_PATH}")
    tenders = [json.load(open(f, "r", encoding="utf-8")) for f in tender_files]

    # Filter valid tenders
    valid_tenders = []
    for t in tenders:
        if len(t.get("positives", [])) == 0:
            continue
        if not isinstance(t.get("chunks"), list):
            continue
        chunk_ids = [c["metadata"]["chunk_id"] for c in t["chunks"]]
        matching_positives = [pid for pid in t["positives"] if pid in chunk_ids]
        if len(matching_positives) > 0:
            valid_tenders.append(t)

    logger.info(f"Found {len(valid_tenders)} valid tenders with positives")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = LearnableQueryRanker(
        initial_query="Find award criteria sections",
        tokenizer=tokenizer
    )

    if Path(MODEL_PATH).exists():
        logger.info(f"Loading model from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    else:
        logger.warning(f"Model not found at {MODEL_PATH}, using untrained model")

    model.to(DEVICE)

    # Evaluate
    metrics = evaluate_full_tender_retrieval(
        model=model,
        tenders=valid_tenders,
        tokenizer=tokenizer,
        device=DEVICE,
        max_len=MAX_LENGTH,
        k_values=[1, 3, 5, 10, 20],
        logger=logger
    )

    # Report
    logger.info("=" * 50)
    logger.info("Full-Tender Retrieval Evaluation")
    logger.info("=" * 50)
    for metric_name, value in metrics.items():
        logger.info(f"  {metric_name}: {value:.4f}")
