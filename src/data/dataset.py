"""
Creates a torch dataset and dataloader for traning the embeddings.
"""
import torch
from torch.utils.data import Dataset

class TenderDataset(Dataset):
    def __init__(self, tender_jsons, tokenizer, max_len=256, num_positives_per_tender=1, num_negatives_per_positive=4):
        """
        tender_jsons : list of dicts like
            { "chunks": [{"text":..., "metadata": {"chunk_id": ...}}, ...],
              "positives": ["chunk_id_1", "chunk_id_2"] }
        tokenizer    : HuggingFace tokenizer
        max_len      : max sequence length for tokenization
        num_positives_per_tender : number of positive chunks to sample per tender
        num_negatives_per_positive : number of negative chunks to sample per positive

        Note: No per-tender context/query. Model uses a learnable query embedding.
        """
        original_count = len(tender_jsons)

        # Filter tenders: must have positives AND at least one matching chunk_id
        valid_tenders = []
        for t in tender_jsons:
            if len(t.get("positives", [])) == 0:
                continue

            # Skip if chunks is not a list (malformed data)
            if not isinstance(t.get("chunks"), list):
                continue

            # Check if any positive actually matches a chunk_id
            chunk_ids = [c["metadata"]["chunk_id"] for c in t["chunks"]]
            matching_positives = [pid for pid in t["positives"] if pid in chunk_ids]

            if len(matching_positives) > 0:
                valid_tenders.append(t)

        self.tender_jsons = valid_tenders
        skipped = original_count - len(self.tender_jsons)

        if skipped > 0:
            print(f"Skipped {skipped} tenders with no matching positive chunks")

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.num_positives_per_tender = num_positives_per_tender
        self.num_negatives_per_positive = num_negatives_per_positive

    def __len__(self):
        return len(self.tender_jsons)

    def train_val_split(self, val_ratio=0.2, seed=42):
        """
        Split dataset into train and validation sets.

        Args:
            val_ratio: Fraction of data for validation
            seed: Random seed for reproducibility

        Returns:
            (train_dataset, val_dataset): Two TenderDataset instances
        """
        import random
        random.seed(seed)

        indices = list(range(len(self.tender_jsons)))
        random.shuffle(indices)

        val_size = int(len(indices) * val_ratio)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        # Create new datasets
        train_dataset = TenderDataset.__new__(TenderDataset)
        train_dataset.tender_jsons = [self.tender_jsons[i] for i in train_indices]
        train_dataset.tokenizer = self.tokenizer
        train_dataset.max_len = self.max_len
        train_dataset.num_positives_per_tender = self.num_positives_per_tender
        train_dataset.num_negatives_per_positive = self.num_negatives_per_positive

        val_dataset = TenderDataset.__new__(TenderDataset)
        val_dataset.tender_jsons = [self.tender_jsons[i] for i in val_indices]
        val_dataset.tokenizer = self.tokenizer
        val_dataset.max_len = self.max_len
        val_dataset.num_positives_per_tender = self.num_positives_per_tender
        val_dataset.num_negatives_per_positive = self.num_negatives_per_positive

        return train_dataset, val_dataset

    def __getitem__(self, idx):
        import random
        tender = self.tender_jsons[idx]

        # all chunk texts
        chunk_texts = [c["text"] for c in tender["chunks"]]
        chunk_ids   = [c["metadata"]["chunk_id"] for c in tender["chunks"]]

        # Find positive and negative indices
        positive_indices = [i for i, cid in enumerate(chunk_ids) if cid in tender["positives"]]
        negative_indices = [i for i, cid in enumerate(chunk_ids) if cid not in tender["positives"]]

        # Sample positives and negatives based on hyperparameters
        num_positives_to_sample = min(self.num_positives_per_tender, len(positive_indices))
        sampled_positives = random.sample(positive_indices, num_positives_to_sample)

        total_negatives = num_positives_to_sample * self.num_negatives_per_positive
        num_negatives_to_sample = min(total_negatives, len(negative_indices))
        sampled_negatives = random.sample(negative_indices, num_negatives_to_sample) if num_negatives_to_sample > 0 else []

        # Combine and shuffle
        sampled_indices = sampled_positives + sampled_negatives
        random.shuffle(sampled_indices)

        # Get sampled chunks
        sampled_texts = [chunk_texts[i] for i in sampled_indices]
        sampled_labels = [1 if chunk_ids[i] in tender["positives"] else 0 for i in sampled_indices]

        # tokenize chunks
        encodings = self.tokenizer(
            sampled_texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "chunk_input_ids": encodings["input_ids"],         # [num_sampled_chunks, max_len]
            "chunk_attention_mask": encodings["attention_mask"], # [num_sampled_chunks, max_len]
            "labels": torch.tensor(sampled_labels, dtype=torch.float), # [num_sampled_chunks]
        }

def collate_fn(batch):
    """
    Collate function for batching tenders with variable number of chunks.
    Pads all tenders to have the same number of chunks (max in batch).
    """
    max_chunks = max(len(item["labels"]) for item in batch)

    chunk_input_ids = []
    chunk_attention_masks = []
    labels = []

    for item in batch:
        num_chunks = len(item["labels"])
        pad_len = max_chunks - num_chunks

        # Pad chunk dimension with zeros
        chunk_input_ids.append(
            torch.cat([
                item["chunk_input_ids"],
                torch.zeros(pad_len, item["chunk_input_ids"].size(1), dtype=torch.long)
            ])
        )
        chunk_attention_masks.append(
            torch.cat([
                item["chunk_attention_mask"],
                torch.zeros(pad_len, item["chunk_attention_mask"].size(1), dtype=torch.long)
            ])
        )
        labels.append(
            torch.cat([item["labels"], torch.zeros(pad_len)])
        )

    return {
        "chunk_input_ids": torch.stack(chunk_input_ids),             # [B, max_chunks, max_len]
        "chunk_attention_mask": torch.stack(chunk_attention_masks),  # [B, max_chunks, max_len]
        "labels": torch.stack(labels),                               # [B, max_chunks]
    }
