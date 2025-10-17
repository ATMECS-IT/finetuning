import json
import logging
import os
from abc import ABC, abstractmethod
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional, Type
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FormatConfig:
    input_fields: List[str]
    output_fields: List[str]
    format_template: str
    name: str

class BaseDatasetFormatter(ABC):    
    def __init__(self, config: FormatConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def format_sample(self, item: Dict) -> Tuple[str, str]:
        pass
    
    def extract_field(self, item: Dict, field_list: List[str], default: str = '') -> str:
        for field in field_list:
            if field in item and item[field]:
                value = item[field]
                if isinstance(value, (list, dict)):
                    return str(value)
                return str(value).strip()
        return default

class InstructionFormatter(BaseDatasetFormatter):    
    def format_sample(self, item: Dict) -> Tuple[str, str]:
        instruction = self.extract_field(item, ['instruction'])
        input_text = self.extract_field(item, ['input', 'context'])
        output_text = self.extract_field(item, self.config.output_fields)
        if input_text:
            formatted_input = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
        else:
            formatted_input = f"### Instruction:\n{instruction}\n\n### Response:"
        return formatted_input, output_text

class QAFormatter(BaseDatasetFormatter):
    def format_sample(self, item: Dict) -> Tuple[str, str]:
        question = self.extract_field(item, self.config.input_fields)
        answer = self.extract_field(item, self.config.output_fields)
        context = item.get('context', item.get('passage', ''))
        if context:
            formatted_input = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            formatted_input = f"Question: {question}\n\nAnswer:"
        return formatted_input, answer

class GenericTextFormatter(BaseDatasetFormatter):
    def format_sample(self, item: Dict) -> Tuple[str, str]:
        input_text = self.extract_field(item, self.config.input_fields)
        output_text = self.extract_field(item, self.config.output_fields)
        return input_text, output_text

class ConversationFormatter(BaseDatasetFormatter):
    def format_sample(self, item: Dict) -> Tuple[str, str]:
        messages = item.get('messages', [])
        if not messages:
            return "", ""
        conversation_parts = []
        assistant_response = ""
        for msg in messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            if role == 'system':
                conversation_parts.append(f"System: {content}")
            elif role == 'user':
                conversation_parts.append(f"User: {content}")
            elif role == 'assistant':
                assistant_response = content
                break  # Stop at first assistant response
        formatted_input = "\n\n".join(conversation_parts) + "\n\nAssistant:"
        return formatted_input, assistant_response


class DatasetFormatterFactory:    
    _registry: Dict[str, Type[BaseDatasetFormatter]] = {}
    _configs: Dict[str, FormatConfig] = {}
    
    @classmethod
    def register_formatter(cls, name: str, formatter_class: Type[BaseDatasetFormatter], 
                          config: FormatConfig):
        cls._registry[name] = formatter_class
        cls._configs[name] = config
        logger.info(f"Registered formatter: {name}")
    
    @classmethod
    def create_formatter(cls, dataset_type: str):
        if dataset_type not in cls._registry:
            raise ValueError(f"Unknown dataset type: {dataset_type}. "
                           f"Available: {list(cls._registry.keys())}")
        
        formatter_class = cls._registry[dataset_type]
        config = cls._configs[dataset_type]
        return formatter_class(config)
    
    @classmethod
    def list_formatters(cls) -> List[str]:
        """List all registered formatters"""
        return list(cls._registry.keys())

class GenericLLMDataset(Dataset):
    def __init__(self, data: List[Dict], tokenizer, config: Dict, 
                 formatter: Optional[BaseDatasetFormatter] = None):
        self.data = data
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = int(config.get('max_length', 1024))
        if formatter is None:
            dataset_type = config['training_data'].get('dataset_type', 'text')
            self.formatter = DatasetFormatterFactory.create_formatter(dataset_type)
        else:
            self.formatter = formatter
        
        logger.info(f"Initialized dataset with {len(data)} samples using "
                   f"{self.formatter.__class__.__name__}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        try:
            input_text, output_text = self.formatter.format_sample(item)
            full_text = f"{input_text}\n{output_text}".strip()
            encodings = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            
            return {
                'input_ids': encodings['input_ids'].squeeze(),
                'attention_mask': encodings['attention_mask'].squeeze(),
                'labels': encodings['input_ids'].squeeze()
            }
        
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            empty_encodings = self.tokenizer(
                "",
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            return {
                'input_ids': empty_encodings['input_ids'].squeeze(),
                'attention_mask': empty_encodings['attention_mask'].squeeze(),
                'labels': empty_encodings['input_ids'].squeeze()
            }


class SubsetDataset(Dataset):
    def __init__(self, dataset: GenericLLMDataset, indices: List[int]):
        self.dataset = dataset
        self.indices = indices
        self.original_data = [dataset.data[i] for i in indices]
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]
    
    def get_original_item(self, idx):
        return self.original_data[idx]

def register_default_formatters():        
    DatasetFormatterFactory.register_formatter(
        'instruction',
        InstructionFormatter,
        FormatConfig(
            name='instruction',
            input_fields=['instruction', 'input'],
            output_fields=['output', 'response', 'completion'],
            format_template='### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}'
        )
    )
    
    DatasetFormatterFactory.register_formatter(
        'qa',
        QAFormatter,
        FormatConfig(
            name='qa',
            input_fields=['question', 'query', 'q'],
            output_fields=['answer', 'response', 'a'],
            format_template='Question: {input}\n\nAnswer: {output}'
        )
    )
    
    DatasetFormatterFactory.register_formatter(
        'text',
        GenericTextFormatter,
        FormatConfig(
            name='text',
            input_fields=['text', 'prompt', 'input'],
            output_fields=['completion', 'response', 'output'],
            format_template='{input}\n\n{output}'
        )
    )
    
    DatasetFormatterFactory.register_formatter(
        'conversation',
        ConversationFormatter,
        FormatConfig(
            name='conversation',
            input_fields=['messages'],
            output_fields=[],
            format_template='conversation'
        )
    )

register_default_formatters()


class DataLoader(ABC):
    @abstractmethod
    def load(self, file_path: str) -> List[Dict]:
        pass


class JSONLDataLoader(DataLoader):
    def load(self, file_path: str) -> List[Dict]:
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        return data


class JSONDataLoader(DataLoader):
    def load(self, file_path: str) -> List[Dict]:
        with open(file_path, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)
            if isinstance(loaded_data, list):
                return loaded_data
            elif isinstance(loaded_data, dict) and 'data' in loaded_data and isinstance(loaded_data['data'], list):
                return loaded_data['data']
            else:
                return [loaded_data]
        return []

class CSVDataLoader(DataLoader):
    def load(self, file_path: str) -> List[Dict]:
        import csv
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data


class DatasetManager:
    def __init__(self, config: Dict, tokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def load_data_from_file(self, file_path: str, file_format: str = 'jsonl') -> List[Dict]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        if file_format == 'auto':
            ext = os.path.splitext(file_path)[-1].lower()
            file_format = ext.lstrip('.')
        extension_mapper = {
        'jsonl': JSONLDataLoader(),
        'json': JSONDataLoader(),
        'csv': CSVDataLoader()
        }
        loader = extension_mapper.get(file_format)
        data = loader.load(file_path)
        self.logger.info(f"Loaded {len(data)} samples from {file_path}")
        return data

    def create_metrics_dir(self,timestamp: str) -> str:
        output_dir = self.config['fine_tuning'].get('output_dir', './results')
        project_name = self.config['project'].get('project_name', 'default_project')
        model_name = self.config['model']['model_name'].replace('/', '_')
        metrics_dir = os.path.join(output_dir, project_name, model_name, timestamp)
        os.makedirs(metrics_dir, exist_ok=True)
        return metrics_dir
    
    def save_test_split(self, data: List[Dict], indices: List[int], metrics_dir: str) -> str:
        test_split_path = os.path.join(metrics_dir, 'test_split.jsonl')
        self.logger.info(f"Saving test split to {test_split_path}")
        with open(test_split_path, 'w', encoding='utf-8') as f:
            for idx in indices:
                json.dump(data[idx], f, ensure_ascii=False)
                f.write('\n')
        self.logger.info(f"✓ Saved test split ({len(indices)} samples) to {test_split_path}")
        return test_split_path
    
    def load_and_split(self, timestamp: str):
        training_config = self.config['training_data']
        data_path = training_config['path']
        file_format = training_config.get('data_format', 'jsonl')
        val_path = training_config.get('val_path')
        test_path = training_config.get('test_path')
        data = self.load_data_from_file(data_path, file_format)
        full_dataset = GenericLLMDataset(data, self.tokenizer, self.config)
        if test_path and os.path.exists(test_path):
            return self._split_with_explicit_test(data, full_dataset, val_path, test_path, file_format)
        else:
            return self._auto_split(data, full_dataset, timestamp)
    
    def _split_with_explicit_test(self, data, full_dataset, val_path, test_path, file_format):
        test_data = self.load_data_from_file(test_path, file_format)
        test_dataset = GenericLLMDataset(test_data, self.tokenizer, self.config)
        if val_path and os.path.exists(val_path):
            val_data = self.load_data_from_file(val_path, file_format)
            val_dataset = GenericLLMDataset(val_data, self.tokenizer, self.config)
            train_dataset = full_dataset
        else:
            train_size = int(0.9 * len(data))
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(data)))
            train_dataset = SubsetDataset(full_dataset, train_indices)
            val_dataset = SubsetDataset(full_dataset, val_indices)
        return train_dataset, val_dataset, test_dataset
    
    def _auto_split(self, data, full_dataset,timestamp):
        self.logger.info("Auto-splitting dataset into train/val/test")
        training_config = self.config['training_data']
        train_ratio, val_ratio, test_ratio = float(training_config.get('train_ratio', 0.8)), float(training_config.get('val_ratio', 0.1)), float(training_config.get('test_ratio', 0.1))        
        total = train_ratio + val_ratio + test_ratio
        train_ratio, val_ratio, test_ratio = train_ratio/total, val_ratio/total, test_ratio/total
        train_val_indices, test_indices = train_test_split(
            list(range(len(data))),
            test_size=test_ratio,
            random_state=42
        )
        val_adjusted = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            train_val_indices,
            test_size=val_adjusted,
            random_state=42
        )
        self.logger.info(f"Split sizes - train: {len(train_indices)}, "
                        f"val: {len(val_indices)}, test: {len(test_indices)}")
        train_dataset = SubsetDataset(full_dataset, train_indices)
        val_dataset = SubsetDataset(full_dataset, val_indices)
        test_dataset = SubsetDataset(full_dataset, test_indices)
        metrics_dir = self.create_metrics_dir(timestamp)
        self.save_test_split(data, test_indices, metrics_dir)
        return train_dataset, val_dataset, test_dataset

