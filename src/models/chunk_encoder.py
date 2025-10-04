import json
import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer, AutoModel

class LearnableQueryRanker(nn.Module):
    def __init__(self, model_name="distilbert-base-uncased", initial_query=None, tokenizer=None,
                 freeze_encoder=False, freeze_query=False):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size

        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Learnable query embedding
        if initial_query and tokenizer:
            # Warm start: encode initial query text to initialize embedding
            with torch.no_grad():
                encoded = tokenizer(initial_query, return_tensors="pt", padding=True, truncation=True, max_length=256)
                outputs = self.encoder(**encoded)
                # Mean pool the initial query
                pooled = (outputs.last_hidden_state * encoded['attention_mask'].unsqueeze(-1)).sum(1) / encoded['attention_mask'].sum(1).unsqueeze(-1)
                self.query_embedding = nn.Parameter(pooled, requires_grad=not freeze_query)
        else:
            # Cold start: random initialization
            self.query_embedding = nn.Parameter(torch.randn(1, self.hidden_size), requires_grad=not freeze_query)

    def mean_pool(self, token_embeddings, attention_mask):
        """
        Mean pooling over token dimension, ignoring padding tokens.

        token_embeddings: [batch_size, seq_len, hidden_size]
        attention_mask: [batch_size, seq_len]

        Returns: [batch_size, hidden_size]
        """
        # Sum of attention mask per sequence
        mask_sum = attention_mask.sum(1, keepdim=True)  # [batch_size, 1]

        # Avoid division by zero for fully padded sequences
        mask_sum = torch.clamp(mask_sum, min=1e-9)

        # Weighted sum and normalize
        return (token_embeddings * attention_mask.unsqueeze(-1)).sum(1) / mask_sum

    def encode_chunks(self, input_ids, attention_mask):
        """
        Encode chunks using DistilBERT + mean pooling.

        input_ids: [batch_size, num_chunks, seq_len]
        attention_mask: [batch_size, num_chunks, seq_len]

        Returns: [batch_size, num_chunks, hidden_size]
        """
        batch_size, num_chunks, seq_len = input_ids.shape

        # Reshape to [batch_size * num_chunks, seq_len]
        input_ids_flat = input_ids.view(batch_size * num_chunks, seq_len)
        attention_mask_flat = attention_mask.view(batch_size * num_chunks, seq_len)

        # Pass through encoder
        outputs = self.encoder(input_ids_flat, attention_mask_flat)

        # Apply mean pooling
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask_flat)

        # Reshape back to [batch_size, num_chunks, hidden_size]
        return pooled.view(batch_size, num_chunks, self.hidden_size)

    def forward(self, chunk_input_ids, chunk_attention_mask):
        """
        Compute scores between learnable query and all chunks.

        Returns: scores [batch_size, num_chunks]
        """
        batch_size = chunk_input_ids.size(0)

        # Encode chunks → [batch_size, num_chunks, hidden_size]
        chunk_embeddings = self.encode_chunks(chunk_input_ids, chunk_attention_mask)

        # Expand query_embedding to batch_size
        # self.query_embedding is [1, hidden_size] → [batch_size, hidden_size]
        query = self.query_embedding.expand(batch_size, -1)

        # Compute dot product scores
        # query: [batch_size, hidden_size] → [batch_size, 1, hidden_size]
        # chunks: [batch_size, num_chunks, hidden_size] → transpose last 2 dims → [batch_size, hidden_size, num_chunks]
        # bmm: [batch_size, 1, hidden_size] @ [batch_size, hidden_size, num_chunks] → [batch_size, 1, num_chunks]
        # squeeze: [batch_size, num_chunks]
        scores = torch.bmm(query.unsqueeze(1), chunk_embeddings.transpose(1, 2)).squeeze(1)

        return scores
