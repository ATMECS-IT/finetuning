# finetuning# Complete Parameter Reference

## Model Configuration

**--model_name:** Specifies the base model to fine-tune. Can be a HuggingFace model name or local path.  
**Default:** `gpt2`  
**Examples:** `gpt2`, `distilgpt2`, `meta-llama/Llama-2-7b-hf`, `Writer/palmyra-small`

**--tokenizer:** Tokenizer name or path if different from model name.  
**Default:** Same as `model_name`  
Use when model and tokenizer differ.

**--hf_token:** HuggingFace authentication token for gated models.  
Required for: Llama models, restricted models.  
Get from: [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## Data Configuration

**--data_path:** Path to training data file (**required**).  
**Formats:** `.jsonl`, `.txt`  
**Example:** `./data/train.jsonl`

**--dataset_type:** Format of the training data (**required**).  
**Options:** `plain_text`, `mcq_jsonl`, `instruction_jsonl`

- `plain_text`: One sample per line  
- `mcq_jsonl`: Multiple choice questions with answers  
- `instruction_jsonl`: Instruction-response pairs

**--val_path:** Separate validation data file (optional).  
Use when you have pre-split validation data.

**--test_path:** Separate test data file (optional).  
Use for final model evaluation.

**--train_ratio:** Percentage of data for training when splitting single file.  
**Default:** `0.8`  
**Range:** `0.0` to `1.0`

**--val_ratio:** Percentage of data for validation when splitting single file.  
**Default:** `0.1`

**--test_ratio:** Percentage of data for testing when splitting single file.  
**Default:** `0.1`

**--split_seed:** Random seed for reproducible data splitting.  
**Default:** `42`  
Use same seed for consistent splits across runs.

---

## Fine-Tuning Method

**--technique:** Fine-tuning technique to use.  
**Options:** `lora`, `qlora`, `dora`, `adapter`

- `lora`: Low-Rank Adaptation, efficient and fast  
- `qlora`: Quantized LoRA, memory efficient for large models  
- `dora`: Weight-Decomposed LoRA, improved performance  
- `adapter`: Load existing adapter weights  

---

## Training Configuration

**--epochs:** Number of complete passes through training data.  
**Default:** `3`  
**Range:** `1` to `20`  
More epochs may improve performance but risk overfitting.

**--batch_size:** Number of samples processed before updating weights.  
**Default:** `4`  
**Range:** `1` to `32`

**--grad_acc_steps:** Gradient accumulation steps for effective larger batches.  
**Default:** `4`  
Effective batch size = `batch_size × grad_acc_steps`.

**--learning_rate:** Step size for weight updates.  
**Default:** `3e-4`  
**Range:** `1e-5` to `5e-4`

**--device:** Hardware device for training.  
**Options:** `cpu`, `cuda`, `mps`, `auto`

- `cpu`: Use CPU (slowest but works everywhere)  
- `cuda`: Use NVIDIA GPU (fastest)  
- `mps`: Use Apple Silicon GPU (not recommended)  
- `auto`: Automatically detect best available device  

**--eval_steps:** Evaluate model every N steps.  
**Default:** `100`  
Only used when validation data provided.

**--save_path:** Directory to save fine-tuned model.  
**Default:** `./results/output`  
Creates timestamped subdirectories automatically.

---

## LoRA-Specific Parameters

**--lora_r:** Rank of LoRA adaptation matrices.  
**Default:** `8`  
**Range:** `4` to `128`  
Higher values capture more complexity but use more memory.  
Recommended: `8–16` for small models, `32–64` for large models.

**--lora_alpha:** Scaling factor for LoRA updates.  
**Default:** `16`  
Typically set to 2× `lora_r`.

**--lora_dropout:** Dropout rate for LoRA layers.  
**Default:** `0.1`  
**Range:** `0.0` to `0.3`

**--lora_target_modules:** Specific model layers to apply LoRA.  
**Default:** `auto` (automatically detected)  
Manual example: `q_proj,v_proj,k_proj`

---

## Model Comparison

**--prompts_file:** Path to text file with test prompts.  
**Default:** `./prompts.txt`  
Format: Separate prompts with blank lines.  
Automatically compares base and fine-tuned model outputs.

---

## Configuration File Usage

Create `.env` files for reusable configurations:

```text
PROJECT_NAME=my_project
MODEL_NAME=gpt2
TRAIN_DATA_PATH=./data/train.jsonl
DATASET_TYPE=mcq_jsonl
TECHNIQUE=lora
EPOCHS=5
BATCH_SIZE=8
LEARNING_RATE=2e-4
DEVICE=cuda
LORA_R=16
LORA_ALPHA=32
OUTPUT_DIR=./results/my_model
Run with:

bash
Copy code
python main.py --config my_config.env
CLI arguments override config file values.

Example Command Combinations
Example 1: Quick CPU Test Run
Fast testing on small dataset with minimal resources.

bash
Copy code
python main.py \
  --model_name gpt2 \
  --data_path ./data/sample.jsonl \
  --dataset_type instruction_jsonl \
  --technique lora \
  --epochs 1 \
  --batch_size 1 \
  --device cpu \
  --train_ratio 1.0 \
  --val_ratio 0.0
Use when testing framework setup, debugging, or quick iterations.

Example 2: Standard GPU Training with Validation
Production training with evaluation metrics.

bash
Copy code
python main.py \
  --model_name gpt2 \
  --data_path ./data/train.jsonl \
  --dataset_type mcq_jsonl \
  --train_ratio 0.8 \
  --val_ratio 0.2 \
  --technique lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --epochs 5 \
  --batch_size 8 \
  --learning_rate 2e-4 \
  --device cuda \
  --eval_steps 50
Use when training with comprehensive metrics and validation.

Example 3: Memory-Efficient Large Model Training
Training 7B+ parameter models with limited VRAM.

bash
Copy code
python main.py \
  --model_name meta-llama/Llama-2-7b-hf \
  --hf_token your_token_here \
  --data_path ./data/train.jsonl \
  --dataset_type instruction_jsonl \
  --technique qlora \
  --lora_r 64 \
  --lora_alpha 16 \
  --epochs 3 \
  --batch_size 4 \
  --grad_acc_steps 4 \
  --device cuda \
  --train_ratio 0.85 \
  --val_ratio 0.15 \
  --eval_steps 100
Use when fine-tuning large models with 8GB–16GB GPU memory.

Example 4: Medical Question Answering with Prompts
Domain-specific fine-tuning with comparison testing.

bash
Copy code
python main.py \
  --model_name gpt2 \
  --data_path ./data/medical_qa.jsonl \
  --dataset_type mcq_jsonl \
  --train_ratio 0.7 \
  --val_ratio 0.2 \
  --test_ratio 0.1 \
  --technique lora \
  --lora_r 32 \
  --lora_alpha 64 \
  --epochs 10 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --device cuda \
  --eval_steps 25 \
  --prompts_file ./medical_prompts.txt
Use when creating specialized models for specific domains.

Example 5: Instruction Tuning with Separate Files
Training with pre-split datasets.

bash
Copy code
python main.py \
  --model_name distilgpt2 \
  --data_path ./data/train_instructions.jsonl \
  --val_path ./data/val_instructions.jsonl \
  --test_path ./data/test_instructions.jsonl \
  --dataset_type instruction_jsonl \
  --technique lora \
  --epochs 3 \
  --batch_size 16 \
  --device cuda \
  --eval_steps 200
Use when you have pre-split train/val/test datasets.

Example 6: DoRA Advanced Fine-Tuning
Using DoRA technique for improved performance.

bash
Copy code
python main.py \
  --model_name gpt2 \
  --data_path ./data/train.jsonl \
  --dataset_type instruction_jsonl \
  --technique dora \
  --lora_r 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --epochs 5 \
  --batch_size 4 \
  --grad_acc_steps 8 \
  --learning_rate 5e-5 \
  --device cuda \
  --train_ratio 0.9 \
  --val_ratio 0.1 \
  --eval_steps 50
Use when you want potentially better performance than standard LoRA.

Example 7: Using Configuration File
Simplifying complex setups with config files.

Create production.env:

text
Copy code
MODEL_NAME=meta-llama/Llama-2-13b-hf
TRAIN_DATA_PATH=./data/large_dataset.jsonl
DATASET_TYPE=instruction_jsonl
TECHNIQUE=qlora
LORA_R=64
LORA_ALPHA=128
EPOCHS=3
BATCH_SIZE=8
GRAD_ACC_STEPS=2
LEARNING_RATE=1e-4
DEVICE=cuda
EVAL_STEPS=100
Run with:

bash
Copy code
python main.py --config production.env --hf_token your_token
Use when managing multiple configurations or collaborating with teams.

## Output Interpretation

Training generates these files in timestamped directories:

- **step_metrics.json:** Loss, learning rate, gradient norm per logging step  
- **epoch_metrics.json:** Train loss, validation loss, accuracy, F1, precision, recall per epoch  
- **final_metrics.json:** Complete training summary with best scores and configuration  
- **model_comparison.json:** Base vs fine-tuned model responses (if prompts provided)  
- **model/:** Directory containing fine-tuned adapter weights  

All metrics are automatically saved to:
results/project_name/model_name/YYYYMMDD_HHMMSS/

