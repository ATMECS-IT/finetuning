# LLM Fine-Tuning Framework

A production-ready, modular framework for fine-tuning Large Language Models with support for multiple PEFT techniques, comprehensive metrics tracking, MongoDB integration, and automated model comparison.

---

## Quick Start

### Basic Training

python main.py --data_path ./data/train.jsonl --dataset_type qa --technique lora --device cuda --epochs 3 --batch_size 4

### With MongoDB and Model Comparison

python main.py --data_path ./data/train.jsonl --dataset_type instruction_jsonl --technique lora --prompts_file ./prompts.txt --device cuda --config default.env

---

## Complete Parameter Reference

### Model Configuration

--model_name: Base model to fine-tune  
Default: gpt2  
Examples: gpt2, distilgpt2, meta-llama/Llama-2-7b-hf, Writer/palmyra-small

--tokenizer: Tokenizer name or path  
Default: Same as model_name

--hf_token: HuggingFace authentication token  
Required for: Llama models, gated models  
Get from: https://huggingface.co/settings/tokens

---

### Data Configuration

--data_path: Path to training data file (required)  
Formats: .jsonl, .txt

--dataset_type: Format of training data (required)  
Options: plain_text, , instruction_jsonl

--val_path: Separate validation data file (optional)

--test_path: Separate test data file (optional)

--train_ratio: Training data percentage (default: 0.8)

--val_ratio: Validation data percentage (default: 0.1)

--test_ratio: Test data percentage (default: 0.1)

--split_seed: Random seed for reproducible splitting (default: 42)

---

### Fine-Tuning Method

--technique: Fine-tuning technique  
Options: lora, qlora, dora, adapter

- lora: Low-Rank Adaptation (efficient and fast)
- qlora: Quantized LoRA (memory-efficient for large models)
- dora: Weight-Decomposed LoRA (improved performance)
- adapter: Load existing adapter weights

---

### Training Configuration

--epochs: Number of training epochs (default: 3)

--batch_size: Batch size per device (default: 4)

--grad_acc_steps: Gradient accumulation steps (default: 4)

--learning_rate: Learning rate (default: 3e-4)

--device: Training device  
Options: cpu, cuda, mps, auto

--eval_steps: Evaluate every N steps (default: 100)

--save_path: Output directory (default: ./results/output)

---

### LoRA-Specific Parameters

--lora_r: Rank of LoRA matrices (default: 8)  
Range: 4-128

--lora_alpha: LoRA scaling factor (default: 16)

--lora_dropout: Dropout for LoRA layers (default: 0.1)

--lora_target_modules: Target modules for LoRA (default: auto)

---

### Model Comparison

--prompts_file: Path to prompts file for comparison  
Format: One prompt per line or double-newline separated  
Example: ./prompts.txt

Automatically generates comparison report with base vs fine-tuned model responses.

---

### MongoDB Configuration

MONGODB_URI: MongoDB connection string  
Default: mongodb://127.0.0.1:27017/

MONGODB_DB_NAME: Database name  
Default: llm_finetuning

MONGODB_ENABLED: Enable/disable MongoDB (default: true)

MongoDB automatically stores:
- training_runs: Run metadata with project/model/timestamp
- step_metrics: Step-wise training metrics
- epoch_metrics: Epoch-wise evaluation metrics
- final_metrics: Complete training summary

All documents are indexed on project_name, model_name, timestamp, and run_id for fast querying.

---

## Configuration File Usage

Create reusable .env configuration files:

# Project Info
PROJECT_NAME=medical_qa_project

# Model
MODEL_NAME=Writer/palmyra-small
HUGGINGFACE_TOKEN=your_token_here

# Data
TRAIN_DATA_PATH=./data/train.jsonl
DATASET_TYPE=text

# Fine-Tuning
TECHNIQUE=lora
LORA_R=16
LORA_ALPHA=32
EPOCHS=5
BATCH_SIZE=8
LEARNING_RATE=2e-4
DEVICE=cuda

# Output
OUTPUT_DIR=./results/my_model

# Model Comparison
PROMPTS_FILE=./prompts.txt
MAX_NEW_TOKENS=100

# MongoDB
MONGODB_ENABLED=true
MONGODB_URI=mongodb://127.0.0.1:27017/
MONGODB_DB_NAME=llm_finetuning

Run with:
python main.py --config my_config.env

CLI arguments override config file values.

---

## Example Commands

### 1. Quick CPU Test

python main.py --model_name gpt2 --data_path ./data/sample.jsonl --dataset_type instruction_jsonl --technique lora --epochs 1 --batch_size 1 --device cpu

### 2. Production GPU Training

python main.py --model_name gpt2 --data_path ./data/train.jsonl --dataset_type  --technique lora --lora_r 16 --lora_alpha 32 --epochs 5 --batch_size 8 --learning_rate 2e-4 --device cuda --eval_steps 50

### 3. Large Model with QLoRA

python main.py --model_name meta-llama/Llama-2-7b-hf --hf_token your_token --data_path ./data/train.jsonl --dataset_type instruction_jsonl --technique qlora --lora_r 64 --epochs 3 --batch_size 4 --grad_acc_steps 4 --device cuda

### 4. Medical QA with Model Comparison

python main.py --model_name gpt2 --data_path ./data/medical_qa.jsonl --dataset_type  --technique lora --lora_r 32 --epochs 10 --batch_size 8 --device cuda --prompts_file ./medical_prompts.txt

### 5. Using Config File

python main.py --config production.env --hf_token your_token

---

## Output Structure

Training generates timestamped directories with comprehensive metrics:

results/
└── project_name/
    └── model_name/
        └── YYYYMMDD_HHMMSS/
            ├── step_metrics.json       # Per-step metrics
            ├── epoch_metrics.json      # Per-epoch metrics
            ├── final_metrics.json      # Training summary
            ├── model_comparison.json   # Base vs fine-tuned (if prompts provided)
            └── model/                  # Fine-tuned adapter weights

### Metrics Files

step_metrics.json: Step-level metrics

[
  {
    "step": 1,
    "loss": 3.297,
    "learning_rate": 0.0003,
    "grad_norm": 1.14,
    "timestamp": "2025-10-13T23:32:01"
  }
]

epoch_metrics.json: Epoch-level evaluation

[
  {
    "epoch": 1,
    "train_loss": 2.15,
    "eval_loss": 1.85,
    "eval_accuracy": 0.75,
    "eval_f1": 0.72,
    "eval_precision": 0.74,
    "eval_recall": 0.73,
    "eval_bleu": 0.45,
    "eval_rougeL": 0.62,
    "epoch_time": 180.5
  }
]

final_metrics.json: Complete training summary

{
  "training_summary": {
    "total_steps": 20,
    "total_epochs": 4,
    "total_training_time": 1734.35,
    "final_train_loss": 1.19,
    "best_eval_loss": 1.17,
    "best_eval_accuracy": 0.85,
    "best_eval_f1": 0.82
  },
  "configuration": {
    "model": "Writer/palmyra-small",
    "technique": "lora",
    "learning_rate": 0.0003,
    "epochs": 4
  }
}

model_comparison.json: Model output comparison

[
  {
    "prompt_id": 1,
    "prompt": "What is diabetes?",
    "base_model_response": "Diabetes is...",
    "finetuned_model_response": "Diabetes mellitus is...",
    "response_length_diff": 45
  }
]

---

## Troubleshooting

### Out of Memory
- Reduce batch_size to 1-2
- Increase grad_acc_steps to maintain effective batch size
- Use qlora instead of lora
- Lower lora_r value

### Blank Model Comparison Outputs
- Increase MAX_NEW_TOKENS in config
- Add min_new_tokens=20 to generation config
- Check if adapter loaded correctly
- Verify model path in comparison call

### MongoDB Connection Issues
- Ensure MongoDB is running: mongod --dbpath /path/to/db
- Check MONGODB_ENABLED=true in config
- Verify connection string in MONGODB_URI

---