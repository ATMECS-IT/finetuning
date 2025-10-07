import logging
import torch
from transformers import Trainer, TrainingArguments
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
)

logger = logging.getLogger(__name__)

def find_target_modules(model):
    module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            names = name.split('.')
            module_names.add(names[-1])
    target_modules = [name for name in module_names if any(
        keyword in name.lower() for keyword in ['q_proj', 'v_proj', 'k_proj', 'o_proj','query', 'value', 'key', 'dense', 'c_attn', 'c_proj']
    )]
    
    if not target_modules:
        target_modules = list(module_names)[:2]  # fallback to first 2 linear layers
    
    logger.info(f"Auto-detected target modules: {target_modules}")
    return target_modules


class BaseFinetuneMethod:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config

    def prepare_model(self):
        return self.model

    def train(self, dataloader):
        device_preference = self.config["fine_tuning"].get("device", "auto").lower()
        
        use_fp16 = False
        use_bf16 = False
        use_cpu = False
        
        if device_preference == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                use_fp16 = True
                logger.info("Auto-detected CUDA GPU, using fp16 precision")
            elif torch.backends.mps.is_available():
                device = "mps"
                logger.info("Auto-detected Apple Silicon (MPS), using float32 precision")
            else:
                device = "cpu"
                use_cpu = True
                logger.info("No GPU detected, using CPU")
        
        elif device_preference == "cuda":
            if torch.cuda.is_available():
                device = "cuda"
                use_fp16 = True
                logger.info("Using CUDA GPU with fp16 precision")
            else:
                raise ValueError("CUDA requested but not available")
        
        elif device_preference == "mps":
            if torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using Apple Silicon (MPS) with float32 precision")
            else:
                raise ValueError("MPS requested but not available")
        
        elif device_preference == "cpu":
            device = "cpu"
            use_cpu = True
            logger.info("Using CPU for training")
        
        else:
            raise ValueError(f"Invalid device: {device_preference}. Choose from: auto, cuda, mps, cpu")
        
        training_args = TrainingArguments(
            output_dir=self.config["fine_tuning"]["output_dir"],
            num_train_epochs=int(self.config["fine_tuning"].get("epochs", 3)),
            per_device_train_batch_size=int(self.config["fine_tuning"].get("batch_size", 8)),
            gradient_accumulation_steps=int(self.config["fine_tuning"].get("gradient_accumulation_steps", 1)),
            learning_rate=float(self.config["fine_tuning"].get("learning_rate", 2e-5)),
            save_strategy=self.config["fine_tuning"].get("save_strategy", "epoch"),
            logging_steps=int(self.config.get("log_every_steps", 10)),
            report_to="tensorboard" if self.config.get("log_every_steps") else "none",
            fp16=use_fp16,
            bf16=use_bf16,
            use_cpu=use_cpu,
            resume_from_checkpoint=self.config["fine_tuning"].get("resume_checkpoint"),
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataloader.dataset,
            tokenizer=self.tokenizer,
        )
        
        trainer.train()
        trainer.save_model(self.config["fine_tuning"]["output_dir"])
        self.tokenizer.save_pretrained(self.config["fine_tuning"]["output_dir"])
        logger.info(f"Model and tokenizer saved at {self.config['fine_tuning']['output_dir']}")
        return self.model, self.tokenizer


class LoRAMethod(BaseFinetuneMethod):
    def prepare_model(self):
        target_modules_str = self.config.get("lora_target_modules", "")
        if target_modules_str and target_modules_str != "auto":
            target_modules = target_modules_str.split(",")
        else:
            target_modules = find_target_modules(self.model)
        
        lora_config = LoraConfig(
            r=int(self.config.get("lora_r", 8)),
            lora_alpha=int(self.config.get("lora_alpha", 32)),
            lora_dropout=float(self.config.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias=self.config.get("lora_bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info("LoRA adapter applied to model")
        return self.model


class QLoRAMethod(BaseFinetuneMethod):
    def prepare_model(self):
        target_modules_str = self.config.get("lora_target_modules", "")
        if target_modules_str and target_modules_str != "auto":
            target_modules = target_modules_str.split(",")
        else:
            target_modules = find_target_modules(self.model)
        
        lora_config = LoraConfig(
            r=int(self.config.get("lora_r", 8)),
            lora_alpha=int(self.config.get("lora_alpha", 32)),
            lora_dropout=float(self.config.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias=self.config.get("lora_bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info("QLoRA adapter applied to quantized model")
        return self.model


class DoRAMethod(BaseFinetuneMethod):
    def prepare_model(self):
        target_modules_str = self.config.get("lora_target_modules", "")
        if target_modules_str and target_modules_str != "auto":
            target_modules = target_modules_str.split(",")
        else:
            target_modules = find_target_modules(self.model)
        
        lora_config = LoraConfig(
            r=int(self.config.get("lora_r", 8)),
            lora_alpha=int(self.config.get("lora_alpha", 32)),
            lora_dropout=float(self.config.get("lora_dropout", 0.05)),
            target_modules=target_modules,
            bias=self.config.get("lora_bias", "none"),
            use_dora=True,  # Enable DoRA
            task_type=TaskType.CAUSAL_LM,
        )
        self.model = get_peft_model(self.model, lora_config)
        logger.info("DoRA adapter applied to model")
        return self.model


class AdapterMethod(BaseFinetuneMethod):
    def prepare_model(self):
        adapter_path = self.config["fine_tuning"].get("adapter_path")
        if not adapter_path:
            raise ValueError("adapter_path must be specified in config for adapter loading")
        
        logger.info(f"Loading adapter from {adapter_path}")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        return self.model



FINETUNING_METHODS = {
    "lora": LoRAMethod,
    "qlora": QLoRAMethod,
    "dora": DoRAMethod,
    "adapter": AdapterMethod,
}


def finetune_model(model, tokenizer, dataloader, config):
    technique = config["fine_tuning"].get("technique", "lora").lower()
    
    MethodClass = FINETUNING_METHODS.get(technique)
    if MethodClass is None:
        raise ValueError(
            f"Unsupported fine-tuning technique: {technique}. "
            f"Available methods: {list(FINETUNING_METHODS.keys())}"
        )
    
    finetuner = MethodClass(model, tokenizer, config)
    model = finetuner.prepare_model()
    return finetuner.train(dataloader)
