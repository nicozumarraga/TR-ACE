import csv
from pathlib import Path
from typing import Dict, Any


def write_metrics_to_csv(
    metrics_file: str,
    config: Dict[str, Any],
    epoch: int,
    train_loss: float,
    val_metrics: Dict[str, float] = None
):
    """
    Write training/validation metrics to CSV file.

    Args:
        metrics_file: Path to CSV file
        config: Training configuration dict
        epoch: Current epoch number
        train_loss: Training loss for this epoch
        val_metrics: Dictionary of validation metrics (optional)
    """
    metrics_path = Path(metrics_file)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if file exists to determine if we need to write header
    file_exists = metrics_path.exists()

    # Prepare row data
    row = {
        'epoch': epoch,
        'model_type': config.get('model_type'),
        'freeze_encoder': config.get('freeze_encoder'),
        'freeze_query': config.get('freeze_query'),
        'train_loss': train_loss,
    }

    # Add validation metrics if provided
    if val_metrics:
        row.update({
            'val_loss': val_metrics.get('val_loss'),
            'precision@1': val_metrics.get('precision@1'),
            'MRR': val_metrics.get('MRR'),
            'NDCG@10': val_metrics.get('NDCG@10'),
            'score_margin': val_metrics.get('score_margin'),
            'Recall@1': val_metrics.get('Recall@1'),
            'Recall@3': val_metrics.get('Recall@3'),
            'Recall@5': val_metrics.get('Recall@5'),
            'Recall@10': val_metrics.get('Recall@10'),
            'Recall@20': val_metrics.get('Recall@20'),
        })

    # Write to CSV
    with open(metrics_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())

        # Write header if file is new
        if not file_exists:
            writer.writeheader()

        writer.writerow(row)
