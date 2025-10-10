import logging
import torch
import os
from transformers import Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from metrics_manager import MetricsCallback, compute_metrics
from trl import SFTTrainer, SFTConfig
from transformers import Trainer, TrainingArguments

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
    
    target_modules = [name for name in target_modules if 'lm_head' not in name.lower() 
                     and 'embed' not in name.lower()]
    if not target_modules:
        logger.warning("Auto-detection failed. Using GPT-2 default modules: ['c_attn', 'c_proj']")
        target_modules = ['c_attn', 'c_proj']
    
    logger.info(f"Auto-detected target modules: {target_modules}")
    return target_modules


class BaseFinetuneMethod:
    def __init__(self, model, tokenizer, config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.peft_config = None 

    def prepare_model(self):
        return self.model

    def train(self, train_dataset, eval_dataset=None):
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

        metrics_callback = MetricsCallback(self.config)
        timestamped_output_dir = os.path.join(
            metrics_callback.metrics_dir,
            "model"
        )
        os.makedirs(timestamped_output_dir, exist_ok=True)
        technique = self.config["fine_tuning"].get("technique", "lora").lower()
        
        # SFT Configuration
        if eval_dataset:
            eval_strategy = "steps"
            eval_steps = int(self.config.get("eval_steps", 100))
            save_strategy = "steps"
            save_steps = eval_steps
            load_best = True
        else:
            eval_strategy = "no"
            eval_steps = None
            save_strategy = self.config["fine_tuning"].get("save_strategy", "epoch")
            save_steps = None
            load_best = False
        
        # sft_config = SFTConfig(
        #     output_dir=self.config["fine_tuning"]["output_dir"],
        #     num_train_epochs=int(self.config["fine_tuning"].get("epochs", 3)),
        #     per_device_train_batch_size=int(self.config["fine_tuning"].get("batch_size", 4)),
        #     gradient_accumulation_steps=int(self.config["fine_tuning"].get("grad_acc_steps", 4)),
        #     learning_rate=float(self.config["fine_tuning"].get("learning_rate", 3e-4)),
        #     logging_steps=int(self.config.get("log_interval", 10)),
        #     save_strategy=save_strategy,
        #     save_steps=save_steps,
        #     eval_strategy=eval_strategy,
        #     eval_steps=eval_steps,
        #     load_best_model_at_end=load_best,
        #     max_length=int(self.config.get("max_length", 512)),
        #     packing=self.config.get("packing", False),
        #     dataset_text_field=None,  # Will be handled automatically by SFTTrainer
        #     fp16=use_fp16,
        #     bf16=use_bf16,
        # )
        
        # # Initialize trainer with optional PEFT config
        # trainer = SFTTrainer(
        #     model=self.model,
        #     args=sft_config,
        #     train_dataset=train_dataset,  # Pass dataset directly, not dataloader.dataset
        #     eval_dataset=eval_dataset if eval_dataset else None,
        #     processing_class=self.tokenizer,
        #     peft_config=self.peft_config,
        #     callbacks=[metrics_callback],
        # )
        
        # # Train
        # trainer.train()
        training_args = TrainingArguments(
            output_dir=timestamped_output_dir,
            num_train_epochs=int(self.config["fine_tuning"].get("epochs", 3)),
            per_device_train_batch_size=int(self.config["fine_tuning"].get("batch_size", 4)),
            per_device_eval_batch_size=int(self.config["fine_tuning"].get("batch_size", 4)),
            gradient_accumulation_steps=int(self.config["fine_tuning"].get("grad_acc_steps", 4)),
            learning_rate=float(self.config["fine_tuning"].get("learning_rate", 3e-4)),
            logging_steps=int(self.config.get("log_interval", 10)),
            save_strategy=save_strategy,
            save_steps=save_steps,
            eval_strategy=eval_strategy,
            eval_steps=eval_steps,
            load_best_model_at_end=load_best,
            fp16=use_fp16,
            bf16=use_bf16,
            report_to="none",  # Disable wandb/tensorboard if not configured
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset if eval_dataset else None,
            compute_metrics=compute_metrics,  
            callbacks=[metrics_callback],
        )
        trainer.train()
        trainer.save_model(timestamped_output_dir) 
        self.tokenizer.save_pretrained(timestamped_output_dir)
        logger.info(f"Model and tokenizer saved at {timestamped_output_dir}")
        return self.model, self.tokenizer, metrics_callback.metrics_dir


class LoRAMethod(BaseFinetuneMethod):
    def prepare_model(self):
        target_modules_str = self.config.get("lora_target_modules", "")
        if target_modules_str and target_modules_str != "auto":
            target_modules = target_modules_str.split(",")
        else:
            target_modules = find_target_modules(self.model)
        
        self.peft_config = LoraConfig(
            r=int(self.config.get("lora_r", 8)),
            lora_alpha=int(self.config.get("lora_alpha", 16)),
            lora_dropout=float(self.config.get("lora_dropout", 0.1)),
            target_modules=target_modules,
            bias=self.config.get("lora_bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        logger.info("LoRA adapter applied to model")
        return self.model


class QLoRAMethod(BaseFinetuneMethod):
    def prepare_model(self):
        target_modules_str = self.config.get("lora_target_modules", "")
        if target_modules_str and target_modules_str != "auto":
            target_modules = target_modules_str.split(",")
        else:
            target_modules = find_target_modules(self.model)
        
        self.peft_config = LoraConfig(
            r=int(self.config.get("lora_r", 8)),
            lora_alpha=int(self.config.get("lora_alpha", 16)),
            lora_dropout=float(self.config.get("lora_dropout", 0.1)),
            target_modules=target_modules,
            bias=self.config.get("lora_bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info("QLoRA configuration prepared for quantized model")
        return self.model


class DoRAMethod(BaseFinetuneMethod):
    def prepare_model(self):
        target_modules_str = self.config.get("lora_target_modules", "")
        if target_modules_str and target_modules_str != "auto":
            target_modules = target_modules_str.split(",")
        else:
            target_modules = find_target_modules(self.model)
        
        self.peft_config = LoraConfig(
            r=int(self.config.get("lora_r", 8)),
            lora_alpha=int(self.config.get("lora_alpha", 16)),
            lora_dropout=float(self.config.get("lora_dropout", 0.1)),
            target_modules=target_modules,
            bias=self.config.get("lora_bias", "none"),
            use_dora=True,
            task_type=TaskType.CAUSAL_LM,
        )
        
        logger.info("DoRA configuration prepared")
        return self.model


class AdapterMethod(BaseFinetuneMethod):
    def prepare_model(self):
        adapter_path = self.config["fine_tuning"].get("adapter_path")
        if not adapter_path:
            raise ValueError("adapter_path must be specified in config for adapter loading")
        
        logger.info(f"Loading adapter from {adapter_path}")
        from peft import PeftModel
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.peft_config = None
        return self.model


FINETUNING_METHODS = {
    "lora": LoRAMethod,
    "qlora": QLoRAMethod,
    "dora": DoRAMethod,
    "adapter": AdapterMethod,
}


def finetune_model(model, tokenizer, train_dataset, config, eval_dataset=None):
    technique = config["fine_tuning"].get("technique", "lora").lower()
    MethodClass = FINETUNING_METHODS.get(technique)
    if MethodClass is None:
        raise ValueError(
            f"Unsupported fine-tuning technique: {technique}. "
            f"Available methods: {list(FINETUNING_METHODS.keys())}"
        )
    
    finetuner = MethodClass(model, tokenizer, config)
    model = finetuner.prepare_model()
    return finetuner.train(train_dataset, eval_dataset)
