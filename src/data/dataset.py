import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer
from typing import Dict, Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class SummarizationDataset(Dataset):
    """
    PyTorch Dataset for text summarization.
    
    Loads the RussianNLP/Mixed-Summarization-Dataset from HuggingFace
    and prepares it for training masked diffusion models.
    """
    
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        split: str = "train",
        max_source_length: int = 512,
        max_target_length: int = 128,
        dataset_name: str = "RussianNLP/Mixed-Summarization-Dataset",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            tokenizer: Tokenizer for encoding text
            split: Dataset split ("train" or "test")
            max_source_length: Maximum source sequence length
            max_target_length: Maximum target sequence length
            dataset_name: HuggingFace dataset name
            cache_dir: Optional cache directory for dataset
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        logger.info(f"Loading dataset {dataset_name} split={split}")
        
        # Load dataset from HuggingFace
        self.dataset = load_dataset(
            dataset_name,
            split=split,
            cache_dir=cache_dir,
        )
        
        logger.info(f"Loaded {len(self.dataset)} examples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example.
        
        Returns:
            Dictionary with:
                - input_ids: Source token IDs
                - attention_mask: Source attention mask
                - labels: Target token IDs
                - labels_attention_mask: Target attention mask
        """
        item = self.dataset[idx]
        
        # Get source and target text
        source_text = item["text"]
        target_text = item["summary"]
        
        # Tokenize source
        source_encoding = self.tokenizer(
            source_text,
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize target
        target_encoding = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        
        return {
            "input_ids": source_encoding["input_ids"].squeeze(0),
            "attention_mask": source_encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
            "labels_attention_mask": target_encoding["attention_mask"].squeeze(0),
        }


class SummarizationCollator:
    """
    Collator for batching summarization examples.
    """
    
    def __init__(self, pad_token_id: int = 0):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Collate batch of examples."""
        return {
            "input_ids": torch.stack([x["input_ids"] for x in batch]),
            "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
            "labels": torch.stack([x["labels"] for x in batch]),
            "labels_attention_mask": torch.stack([x["labels_attention_mask"] for x in batch]),
        }


def create_dataloaders(
    tokenizer: AutoTokenizer,
    batch_size: int = 8,
    max_source_length: int = 512,
    max_target_length: int = 128,
    num_workers: int = 4,
    dataset_name: str = "RussianNLP/Mixed-Summarization-Dataset",
    cache_dir: Optional[str] = None,
    train_subset_size: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders.
    
    Args:
        tokenizer: Tokenizer for encoding text
        batch_size: Batch size
        max_source_length: Maximum source sequence length
        max_target_length: Maximum target sequence length
        num_workers: Number of dataloader workers
        dataset_name: HuggingFace dataset name
        cache_dir: Optional cache directory
        train_subset_size: Optional size limit for training data (for debugging)
    
    Returns:
        Tuple of (train_dataloader, test_dataloader)
    """
    # Create datasets
    train_dataset = SummarizationDataset(
        tokenizer=tokenizer,
        split="train",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    
    test_dataset = SummarizationDataset(
        tokenizer=tokenizer,
        split="test",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
        dataset_name=dataset_name,
        cache_dir=cache_dir,
    )
    
    # Optionally subset training data
    if train_subset_size is not None and train_subset_size < len(train_dataset):
        logger.info(f"Using subset of {train_subset_size} training examples")
        train_dataset.dataset = train_dataset.dataset.select(range(train_subset_size))
    
    # Create collator
    collator = SummarizationCollator(pad_token_id=tokenizer.pad_token_id)
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collator,
        pin_memory=True,
    )
    
    logger.info(f"Train dataloader: {len(train_dataloader)} batches")
    logger.info(f"Test dataloader: {len(test_dataloader)} batches")
    
    return train_dataloader, test_dataloader


def get_sample_batch(
    tokenizer: AutoTokenizer,
    batch_size: int = 4,
    max_source_length: int = 512,
    max_target_length: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Get a sample batch for testing.
    
    Useful for debugging and model verification.
    """
    dataset = SummarizationDataset(
        tokenizer=tokenizer,
        split="test",
        max_source_length=max_source_length,
        max_target_length=max_target_length,
    )
    
    collator = SummarizationCollator(pad_token_id=tokenizer.pad_token_id)
    
    batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    return collator(batch)
