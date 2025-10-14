import os
from dotenv import dotenv_values

def load_config_from_env(env_file_path):
    if not os.path.exists(env_file_path):
        raise FileNotFoundError(f"Config file not found: {env_file_path}")
    
    config_raw = dotenv_values(env_file_path)
    
    config = {
        "project": {
            "project_name": config_raw.get("PROJECT_NAME", "default_project"),
        },
        "huggingface": {
            "token": config_raw.get("HUGGINGFACE_TOKEN", ""),
        },
        "model": {
            "model_name": config_raw.get("MODEL_NAME", "gpt2"),
            "tokenizer": config_raw.get("TOKENIZER", "gpt2"),
            "cache_dir": config_raw.get("CACHE_DIR", "./models"),
            "use_auth_token": config_raw.get("USE_AUTH_TOKEN", "False").lower() == "true",
        },
        "training_data": {
            "path": config_raw.get("TRAIN_DATA_PATH", "./data/train.jsonl"),
            "dataset_type": config_raw.get("DATASET_TYPE", "instruction_jsonl"),
            "data_format": config_raw.get("DATA_FORMAT", "jsonl"),
        },
        "tokenizer": {
            "pad_token": config_raw.get("PAD_TOKEN", "<|endoftext|>"),
        },
        "fine_tuning": {
            "technique": config_raw.get("TECHNIQUE", "lora"),
            "output_dir": config_raw.get("OUTPUT_DIR", "./results/output"),
            "epochs": config_raw.get("EPOCHS", "3"),
            "batch_size": config_raw.get("BATCH_SIZE", "4"),
            "learning_rate": config_raw.get("LEARNING_RATE", "3e-4"),
            "gradient_accumulation_steps": config_raw.get("GRAD_ACC_STEPS", "4"),
            "device": config_raw.get("DEVICE", "auto"),
            "save_strategy": config_raw.get("SAVE_STRATEGY", "epoch"),
            "resume_checkpoint": config_raw.get("RESUME_CHECKPOINT", ""),
        },
        "lora_r": config_raw.get("LORA_R", "8"),
        "lora_alpha": config_raw.get("LORA_ALPHA", "16"),
        "lora_dropout": config_raw.get("LORA_DROPOUT", "0.1"),
        "lora_target_modules": config_raw.get("LORA_TARGET_MODULES", "auto"),
        "lora_bias": config_raw.get("LORA_BIAS", "none"),
        "max_length": config_raw.get("MAX_LENGTH", "512"),
        "log_every_steps": config_raw.get("LOG_INTERVAL", "10"),
        "metrics": config_raw.get("METRICS", "loss,accuracy"),
        "metrics_path": config_raw.get("METRICS_PATH", "./results/metrics.json"),
        "report_path": config_raw.get("REPORT_PATH", "./results/report.json"),
        
        "load_in_4bit": config_raw.get("LOAD_IN_4BIT", "False").lower() == "true",
        "bnb_4bit_compute_dtype": config_raw.get("BNB_4BIT_COMPUTE_DTYPE", "float16"),
        "bnb_4bit_use_double_quant": config_raw.get("BNB_4BIT_USE_DOUBLE_QUANT", "False").lower() == "true",
        "bnb_4bit_quant_type": config_raw.get("BNB_4BIT_QUANT_TYPE", "nf4"),
    }
    config["database"] = {
        "mongodb_uri": config_raw.get("MONGODB_URI", "mongodb://127.0.0.1:27017/"),
        "mongodb_db_name": config_raw.get("MONGODB_DB_NAME", "llm_finetuning"),
        "mongodb_enabled": config_raw.get("MONGODB_ENABLED", "true")
    }
    config['comparison'] = {
        'prompts_file': os.getenv('PROMPTS_FILE', './prompts.txt'),
        'max_new_tokens': int(os.getenv('MAX_NEW_TOKENS', 100))
    }
    return config


def merge_configs(base_config, override_config):
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        elif value is not None:  
            merged[key] = value
    
    return merged
