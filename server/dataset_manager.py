import os
import json
from torch.utils.data import Dataset, DataLoader

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
                prompt = f"Question: {obj['question']}" #\nOptions:\n"
                # for key, val in obj["options"].items():
                #     prompt += f"{key}. {val}\n"
                prompt += "Answer:"
                answer = obj["answer"]
                self.samples.append((prompt, answer))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt, answer = self.samples[idx]
        full_text = prompt + "\nAnswer: " + answer
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


def load_dataset_from_config(config, tokenizer):
    data_path = config["training_data"]["path"]
    dataset_type = config["training_data"].get("dataset_type", "plain_text").lower()
    max_length = config["fine_tuning"].get("max_length", 512)
    batch_size = config["fine_tuning"].get("batch_size", 8)
    print("###BtachSize",batch_size)
    batch_size = int(config["fine_tuning"].get("batch_size", 8))


    if dataset_type == "plain_text":
        dataset = PlainTextDataset(data_path, tokenizer, max_length)
    elif dataset_type == "mcq_jsonl":
        dataset = MCQDataset(data_path, tokenizer, max_length)
    elif dataset_type == "instruction_jsonl":
        dataset = InstructionDataset(data_path, tokenizer, max_length)
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
