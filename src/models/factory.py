"""
Model factory for creating rankers and embedders.

Provides a unified interface for model selection to keep training code DRY.
"""

from src.models.chunk_encoder import LearnableQueryRanker
from src.models.adaptive_embedder import AdaptiveEmbedder
from src.losses.BCE import compute_bce_loss
from src.losses.contrastive import compute_infonce_loss_efficient


def create_model(model_type, model_name=None, initial_query=None, tokenizer=None,
                 freeze_encoder=False, freeze_query=False):
    """
    Factory function to create a model based on type.

    Args:
        model_type: "ranker" or "embedder"
        model_name: Model identifier (e.g., "distilbert-base-uncased" or "all-MiniLM-L6-v2")
        initial_query: Initial query text for warm-start
        tokenizer: Tokenizer for the model
        freeze_encoder: Whether to freeze encoder weights
        freeze_query: Whether to freeze query embedding

    Returns:
        model: Instance of LearnableQueryRanker or AdaptiveEmbedder
    """
    if model_type == "ranker":
        default_model = "distilbert-base-uncased"
        model_name = model_name or default_model
        return LearnableQueryRanker(
            model_name=model_name,
            initial_query=initial_query,
            tokenizer=tokenizer,
            freeze_encoder=freeze_encoder,
            freeze_query=freeze_query
        )

    elif model_type == "embedder":
        default_model = "sentence-transformers/all-MiniLM-L6-v2"
        model_name = model_name or default_model
        return AdaptiveEmbedder(
            model_name=model_name,
            initial_query=initial_query,
            tokenizer=tokenizer,
            freeze_encoder=freeze_encoder,
            freeze_query=freeze_query
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'ranker' or 'embedder'")


def get_loss_function(model_type):
    """
    Get the appropriate loss function for a model type.

    Args:
        model_type: "ranker" or "embedder"

    Returns:
        loss_fn: Loss function
    """
    if model_type == "ranker":
        return compute_bce_loss

    elif model_type == "embedder":
        return compute_infonce_loss_efficient

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose 'ranker' or 'embedder'")


def get_model_config(model_type):
    """
    Get default configuration for a model type.

    Args:
        model_type: "ranker" or "embedder"

    Returns:
        dict with recommended hyperparameters
    """
    if model_type == "ranker":
        return {
            "model_name": "distilbert-base-uncased",
            "learning_rate": 1e-4,
            "loss_name": "BCE"
        }

    elif model_type == "embedder":
        return {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "learning_rate": 1e-3,  # Higher LR since only training query embedding
            "loss_name": "InfoNCE"
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}")
