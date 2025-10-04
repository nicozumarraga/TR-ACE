import os
import json
import yaml
import torch
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.data.dataset import TenderDataset, collate_fn
from src.data.preprocessor import TenderPreprocessor
from src.models.factory import create_model, get_loss_function, get_model_config
from src.trainers.chunk_ranker_trainer import ChunkRankerTrainer
from src.utils.logger import setup_logger
from src.utils.metrics_writer import write_metrics_to_csv


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model selection
    model_type: str = "ranker"  # "ranker" (DistilBERT) or "embedder" (all-MiniLM)
    model_name: str = None  # Override default model for the type (optional)

    # Model training flags
    freeze_encoder: bool = True  # Freeze encoder weights
    freeze_query: bool = True   # Freeze query embedding

    # Data configuration
    category: str = "5. CRITERIOS DE ADJUDICACIÓN"
    initial_query: str = "Cuales son los criterios de adjudicación?"

    # Training hyperparameters
    batch_size: int = 2
    num_epochs: int = 1
    learning_rate: float = None  # Will use default from model_type if not specified
    max_length: int = 256
    device: str = "cpu"

    # Dataset configuration
    val_split: float = 0.2
    num_positives_per_tender: int = 2
    num_negatives_per_positive: int = 4

    # Logging
    debug: bool = True

    def __post_init__(self):
        """Set defaults based on model_type if not specified."""
        model_config = get_model_config(self.model_type)

        if self.model_name is None:
            self.model_name = model_config["model_name"]

        if self.learning_rate is None:
            self.learning_rate = model_config["learning_rate"]

    @property
    def category_slug(self) -> str:
        """Extract section number from category (e.g., '5. CRITERIOS...' -> 'section_5')."""
        return f"section_{self.category.split('.')[0]}"

    @property
    def preproc_path(self) -> str:
        """Path to preprocessed data for this category."""
        return f"data/1.pre-processed/{self.category_slug}"

    @property
    def checkpoint_dir(self) -> str:
        """Path to checkpoint directory for this category and model type."""
        return f"checkpoints/{self.model_type}_{self.category_slug}"


def load_config_from_yaml(yaml_path: str) -> TrainingConfig:
    """Load training configuration from YAML file."""
    with open(yaml_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return TrainingConfig(**config_dict)


def main():
    parser = argparse.ArgumentParser(description="Train tender ranking/embedding model")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--log-file", type=str, help="Path to save training logs")
    parser.add_argument("--metrics-file", type=str, help="Path to save metrics CSV")
    args = parser.parse_args()

    # Load config from YAML if provided, otherwise use defaults
    if args.config:
        config = load_config_from_yaml(args.config)
    else:
        config = TrainingConfig()

    # Setup logger
    log_level = logging.DEBUG if config.debug else logging.INFO
    logger = setup_logger("training", level=log_level, log_file=args.log_file)

    # Run preprocessor if needed
    preproc_path = Path(config.preproc_path)
    if not preproc_path.exists() or len(list(preproc_path.glob("*.json"))) == 0:
        logger.info(f"Preprocessing data for category: {config.category}")
        preprocessor = TenderPreprocessor(config.category)
        preprocessor.process_all_tenders()
        logger.info(f"Preprocessing complete. Output at: {config.preproc_path}")
    else:
        logger.info(f"Using existing preprocessed data at: {config.preproc_path}")

    # Load preprocessed tenders
    tender_files = list(preproc_path.glob("*.json"))
    logger.info(f"Found {len(tender_files)} pre-processed tenders")

    tenders = [json.load(open(f, "r", encoding="utf-8")) for f in tender_files]

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    # Create dataset
    dataset = TenderDataset(
        tenders,
        tokenizer,
        max_len=config.max_length,
        num_positives_per_tender=config.num_positives_per_tender,
        num_negatives_per_positive=config.num_negatives_per_positive
    )

    # Split into train and validation
    train_dataset, val_dataset = dataset.train_val_split(val_ratio=config.val_split)
    logger.info(f"Train set: {len(train_dataset)} tenders, Val set: {len(val_dataset)} tenders")

    # Extract validation tenders for full-tender evaluation
    val_tenders = val_dataset.tender_jsons

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Initialize model with warm start
    model = create_model(
        model_type=config.model_type,
        model_name=config.model_name,
        initial_query=config.initial_query,
        tokenizer=tokenizer,
        freeze_encoder=config.freeze_encoder,
        freeze_query=config.freeze_query
    )

    # Get loss function for model type
    loss_fn = get_loss_function(config.model_type)

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # Create trainer
    trainer = ChunkRankerTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=config.device,
        logger=logger,
        val_tenders=val_tenders,
        tokenizer=tokenizer,
        max_len=config.max_length,
        metrics_file=args.metrics_file,
        config_dict=config.__dict__
    )

    # Train
    logger.info(f"Starting Training - Model: {config.model_type}, Loss: {get_model_config(config.model_type)['loss_name']}")
    trainer.train(num_epochs=config.num_epochs, checkpoint_dir=config.checkpoint_dir)

    # Save final model with category and type name
    model_path = os.path.join(config.checkpoint_dir, f"{config.model_type}_{config.category_slug}.pt")
    trainer.save_model(model_path)
    logger.info(f"Training complete for category: {config.category}, model: {config.model_type}")


if __name__ == "__main__":
    main()
