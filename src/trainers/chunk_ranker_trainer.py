import torch
import logging
from pathlib import Path
from tqdm import tqdm
from src.utils.metrics import compute_ranking_metrics
from src.utils.evaluate import evaluate_full_tender_retrieval
from src.utils.metrics_writer import write_metrics_to_csv

class ChunkRankerTrainer:
    def __init__(self, model, train_loader, loss_fn, optimizer, device="cpu", val_loader=None, logger=None,
                 val_tenders=None, tokenizer=None, max_len=256, metrics_file=None, config_dict=None):
        """
        Args:
            model: LearnableQueryRanker instance
            train_loader: DataLoader with TenderDataset
            loss_fn: Loss function (e.g., compute_bce_loss)
            optimizer: torch optimizer
            device: "cpu" or "cuda"
            val_loader: Optional validation DataLoader
            logger: logging.Logger instance
            val_tenders: Optional list of tender dicts for full-tender evaluation
            tokenizer: Optional tokenizer for full-tender evaluation
            max_len: Max sequence length for full-tender evaluation
            metrics_file: Optional path to save metrics CSV
            config_dict: Optional config dictionary for CSV metadata
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        self.val_tenders = val_tenders
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.metrics_file = metrics_file
        self.config_dict = config_dict or {}

    def train_epoch(self):
        """Train for one epoch and return average loss."""
        # Check if there are trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if trainable_params == 0:
            self.logger.warning("No trainable parameters detected. Skipping training epoch.")
            return 0.0

        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            # Move batch to device
            chunk_input_ids = batch["chunk_input_ids"].to(self.device)
            chunk_attention_mask = batch["chunk_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            scores = self.model(chunk_input_ids, chunk_attention_mask)

            # Compute loss
            loss = self.loss_fn(scores, labels)
            self.logger.debug(f"Batch loss: {loss.item():.4f}, scores range: [{scores.min():.2f}, {scores.max():.2f}]")

            # Backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss

    def evaluate(self):
        """Evaluate model on validation set."""
        if self.val_loader is None:
            self.logger.warning("No validation loader provided, skipping evaluation")
            return {}

        self.model.eval()
        total_loss = 0
        all_metrics = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                # Move batch to device
                chunk_input_ids = batch["chunk_input_ids"].to(self.device)
                chunk_attention_mask = batch["chunk_attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                scores = self.model(chunk_input_ids, chunk_attention_mask)

                # Compute loss
                loss = self.loss_fn(scores, labels)
                total_loss += loss.item()

                # Compute metrics
                metrics = compute_ranking_metrics(scores, labels)
                all_metrics.append(metrics)

        # Average metrics across batches
        avg_metrics = {}
        if all_metrics:
            metric_keys = all_metrics[0].keys()
            for key in metric_keys:
                avg_metrics[key] = sum(m[key] for m in all_metrics) / len(all_metrics)

        avg_metrics['val_loss'] = total_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0

        return avg_metrics

    def train(self, num_epochs, checkpoint_dir="checkpoints"):
        """
        Train for multiple epochs and save checkpoints.

        Args:
            num_epochs: Number of training epochs
            checkpoint_dir: Directory to save model checkpoints
        """
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        for epoch in range(num_epochs):
            # Train
            avg_loss = self.train_epoch()
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}")

            # Evaluate
            if self.val_loader:
                val_metrics = self.evaluate()
                self.logger.info(f"  Val Loss: {val_metrics.get('val_loss', 0):.4f}")
                self.logger.info(f"  Precision@1: {val_metrics.get('precision@1', 0):.4f}")
                self.logger.info(f"  MRR: {val_metrics.get('MRR', 0):.4f}")
                self.logger.info(f"  NDCG@10: {val_metrics.get('NDCG@10', 0):.4f}")
                self.logger.info(f"  Score Margin: {val_metrics.get('score_margin', 0):.4f}")

                # Full-tender retrieval evaluation
                if self.val_tenders and self.tokenizer:
                    retrieval_metrics = evaluate_full_tender_retrieval(
                        model=self.model,
                        tenders=self.val_tenders,
                        tokenizer=self.tokenizer,
                        device=self.device,
                        max_len=self.max_len,
                        k_values=[1, 3, 5, 10, 20],
                        logger=self.logger
                    )
                    self.logger.info(f"  [Full-Tender] Recall@1: {retrieval_metrics.get('Recall@1', 0):.4f}")
                    self.logger.info(f"  [Full-Tender] Recall@3: {retrieval_metrics.get('Recall@3', 0):.4f}")
                    self.logger.info(f"  [Full-Tender] Recall@5: {retrieval_metrics.get('Recall@5', 0):.4f}")
                    self.logger.info(f"  [Full-Tender] Recall@10: {retrieval_metrics.get('Recall@10', 0):.4f}")
                    self.logger.info(f"  [Full-Tender] Recall@20: {retrieval_metrics.get('Recall@20', 0):.4f}")
                    val_metrics.update(retrieval_metrics)

            # Write metrics to CSV if file provided
            if self.metrics_file:
                write_metrics_to_csv(
                    metrics_file=self.metrics_file,
                    config=self.config_dict,
                    epoch=epoch + 1,
                    train_loss=avg_loss,
                    val_metrics=val_metrics if self.val_loader else None
                )

            # Save checkpoint every epoch
            checkpoint_file = checkpoint_path / f"ranker_epoch_{epoch+1}.pt"
            checkpoint_data = {
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_loss': avg_loss,
            }
            if self.val_loader:
                checkpoint_data['val_metrics'] = val_metrics

            torch.save(checkpoint_data, checkpoint_file)

        self.logger.info(f"Training complete. Checkpoints saved to {checkpoint_path}")

    def save_model(self, path):
        """Save only the model state dict."""
        torch.save(self.model.state_dict(), path)
        self.logger.info(f"Model saved to {path}")
