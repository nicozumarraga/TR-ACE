import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class AdaptiveEmbedder(nn.Module):
    """
    Embedding model with configurable freezing options.

    Uses transformers models (e.g., sentence-transformers/all-MiniLM-L6-v2) with optional
    freezing of encoder and/or query embedding for flexible fine-tuning strategies.
    """

    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", initial_query=None, tokenizer=None,
                 freeze_encoder=True, freeze_query=False):
        super().__init__()

        # Load model
        self.encoder = AutoModel.from_pretrained(model_name)

        # Freeze encoder if requested (default: True for embedder)
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Get embedding dimension from the model config
        self.hidden_size = self.encoder.config.hidden_size

        # Learnable query embedding
        if initial_query and tokenizer:
            # Warm start: encode initial query text to initialize embedding
            with torch.no_grad():
                # Tokenize and encode the initial query
                encoded = tokenizer(initial_query, return_tensors="pt", padding=True, truncation=True, max_length=256)
                outputs = self.encoder(**encoded)

                # Mean pooling
                pooled = self.mean_pool(outputs.last_hidden_state, encoded['attention_mask'])
                self.query_embedding = nn.Parameter(pooled, requires_grad=not freeze_query)
        else:
            # Cold start: random initialization
            self.query_embedding = nn.Parameter(torch.randn(1, self.hidden_size), requires_grad=not freeze_query)

    def mean_pool(self, token_embeddings, attention_mask):
        """
        Mean pooling over token dimension, ignoring padding tokens.

        Args:
            token_embeddings: [batch_size, seq_len, hidden_size]
            attention_mask: [batch_size, seq_len]

        Returns:
            pooled: [batch_size, hidden_size]
        """
        # Sum of attention mask per sequence
        mask_sum = attention_mask.sum(1, keepdim=True)  # [batch_size, 1]

        # Avoid division by zero
        mask_sum = torch.clamp(mask_sum, min=1e-9)

        # Weighted sum and normalize
        return (token_embeddings * attention_mask.unsqueeze(-1)).sum(1) / mask_sum

    def encode_chunks(self, input_ids, attention_mask):
        """
        Encode chunks using transformer model.

        Args:
            input_ids: [batch_size, num_chunks, seq_len]
            attention_mask: [batch_size, num_chunks, seq_len]

        Returns:
            chunk_embeddings: [batch_size, num_chunks, hidden_size]
        """
        batch_size, num_chunks, seq_len = input_ids.shape

        # Reshape to [batch_size * num_chunks, seq_len]
        input_ids_flat = input_ids.view(batch_size * num_chunks, seq_len)
        attention_mask_flat = attention_mask.view(batch_size * num_chunks, seq_len)

        # Encode (gradients depend on whether encoder is frozen)
        outputs = self.encoder(input_ids_flat, attention_mask_flat)

        # Apply mean pooling to get sentence embeddings
        embeddings = self.mean_pool(outputs.last_hidden_state, attention_mask_flat)

        # Reshape back to [batch_size, num_chunks, hidden_size]
        chunk_embeddings = embeddings.view(batch_size, num_chunks, self.hidden_size)

        return chunk_embeddings

    def forward(self, chunk_input_ids, chunk_attention_mask):
        """
        Compute similarity scores between learnable query and all chunks.

        Args:
            chunk_input_ids: [batch_size, num_chunks, seq_len]
            chunk_attention_mask: [batch_size, num_chunks, seq_len]

        Returns:
            scores: [batch_size, num_chunks] - cosine similarity scores
        """
        batch_size = chunk_input_ids.size(0)

        # Encode chunks → [batch_size, num_chunks, hidden_size]
        chunk_embeddings = self.encode_chunks(chunk_input_ids, chunk_attention_mask)

        # Normalize embeddings for cosine similarity
        chunk_embeddings_norm = torch.nn.functional.normalize(chunk_embeddings, p=2, dim=2)
        query_norm = torch.nn.functional.normalize(self.query_embedding, p=2, dim=1)

        # Expand query to batch size
        query = query_norm.expand(batch_size, -1)  # [batch_size, hidden_size]

        # Compute cosine similarity scores
        # query: [batch_size, 1, hidden_size]
        # chunks: [batch_size, num_chunks, hidden_size] → transpose → [batch_size, hidden_size, num_chunks]
        # result: [batch_size, 1, num_chunks] → squeeze → [batch_size, num_chunks]
        scores = torch.bmm(
            query.unsqueeze(1),
            chunk_embeddings_norm.transpose(1, 2)
        ).squeeze(1)

        return scores
