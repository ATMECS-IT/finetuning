import os
import json
import logging
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class PlainTextDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        with open(data_path, 'r') as f:
            self.texts = [line.strip() for line in f if line.strip()]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        tokens['labels'] = tokens['input_ids'].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}

class MCQDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                prompt = f"Question: {obj['question']}\nAnswer:"
                answer = obj["options"][obj["answer"]]
                self.samples.append((prompt, answer))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, answer = self.samples[idx]
        full_text = prompt + " " + answer
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        tokens['labels'] = tokens['input_ids'].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}

class InstructionDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        self.samples = []
        with open(data_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                prompt = f"Instruction: {obj['instruction']}\nResponse:"
                response = obj['response']
                self.samples.append((prompt, response))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, response = self.samples[idx]
        full_text = prompt + " " + response
        tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )
        tokens['labels'] = tokens['input_ids'].clone()
        return {k: v.squeeze(0) for k, v in tokens.items()}


class SubsetDataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.column_names = None
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Train, val, and test ratios must sum to 1.0"
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    logger.info(f"Splitting dataset: train={train_size}, val={val_size}, test={test_size}")
    
    import torch
    torch.manual_seed(seed)
    indices = torch.randperm(total_size).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_dataset = SubsetDataset(dataset, train_indices)
    val_dataset = SubsetDataset(dataset, val_indices)
    test_dataset = SubsetDataset(dataset, test_indices)
    
    return train_dataset, val_dataset, test_dataset


def load_dataset_from_config(config, tokenizer, logger):
    data_path = config["training_data"]["path"]
    dataset_type = config["training_data"].get("dataset_type", "plain_text").lower()
    max_length = int(config.get("max_length", 1024))
    if dataset_type == "plain_text":
        full_dataset = PlainTextDataset(data_path, tokenizer, max_length)
    elif dataset_type == "mcq_jsonl":
        full_dataset = MCQDataset(data_path, tokenizer, max_length)
    elif dataset_type == "instruction_jsonl":
        full_dataset = InstructionDataset(data_path, tokenizer, max_length)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    logger.info(f"Loaded {len(full_dataset)} samples from {data_path}")
    val_path = config["training_data"].get("val_path")
    test_path = config["training_data"].get("test_path")
    datasets = {}
    if val_path or test_path:
        logger.info("Using separate validation/test datasets")
        datasets['train'] = full_dataset
        if val_path:
            if dataset_type == "plain_text":
                datasets['val'] = PlainTextDataset(val_path, tokenizer, max_length)
            elif dataset_type == "mcq_jsonl":
                datasets['val'] = MCQDataset(val_path, tokenizer, max_length)
            elif dataset_type == "instruction_jsonl":
                datasets['val'] = InstructionDataset(val_path, tokenizer, max_length)
            logger.info(f"Loaded {len(datasets['val'])} validation samples from {val_path}")
        
        if test_path:
            if dataset_type == "plain_text":
                datasets['test'] = PlainTextDataset(test_path, tokenizer, max_length)
            elif dataset_type == "mcq_jsonl":
                datasets['test'] = MCQDataset(test_path, tokenizer, max_length)
            elif dataset_type == "instruction_jsonl":
                datasets['test'] = InstructionDataset(test_path, tokenizer, max_length)
            logger.info(f"Loaded {len(datasets['test'])} test samples from {test_path}")
    
    else:
        logger.info("Splitting dataset into train/val/test")
        train_ratio = float(config["training_data"].get("train_ratio", 0.8))
        val_ratio = float(config["training_data"].get("val_ratio", 0.1))
        test_ratio = float(config["training_data"].get("test_ratio", 0.1))
        seed = int(config["training_data"].get("split_seed", 42))
        
        train_dataset, val_dataset, test_dataset = split_dataset(
            full_dataset, train_ratio, val_ratio, test_ratio, seed
        )
        
        datasets['train'] = train_dataset
        datasets['val'] = val_dataset if val_ratio > 0 else None
        datasets['test'] = test_dataset if test_ratio > 0 else None
    
    return datasets
