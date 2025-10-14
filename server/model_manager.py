import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_from_config(config):
    model_name = config["model"]["model_name"]
    tokenizer_name = config["model"]["tokenizer"]
    cache_dir=config["model"].get("cache_dir", None)
    token = config["huggingface"]["token"] if config["model"].get("use_auth_token") else None

    if "llama" in tokenizer_name.lower():
        from transformers import LlamaTokenizer
        tokenizer = LlamaTokenizer.from_pretrained(
            tokenizer_name,
            token=token,
            cache_dir=cache_dir,
        )
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=token,
            cache_dir=cache_dir,
        )
    pad_token = config.get("pad_token", None)
    if pad_token:
        if pad_token == "<eos>":
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': pad_token})
            tokenizer.pad_token = pad_token
    elif tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model with quantization if specified
    model_kwargs = {
        "cache_dir": cache_dir,
        "use_auth_token": token,
    }
    
    # Handle quantization for QLoRA
    if config.get("load_in_4bit", False):
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(torch, config.get("bnb_4bit_compute_dtype", "float16")),
            bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", False),
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    return model, tokenizer