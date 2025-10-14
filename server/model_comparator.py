import os
import json
import torch
import logging
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict, Optional
from contextlib import contextmanager

class ModelComparator:
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)

    @contextmanager
    def _load_model_context(self, model_name: str, adapter_path: Optional[str] = None,
                           device: str = "cpu", use_8bit: bool = False):
        model = None
        tokenizer = None
        try:
            hf_token = self.config.get("huggingface", {}).get("token")
            if not hf_token:
                hf_token = os.environ.get("HUGGINGFACE_TOKEN")
            
            dtype = torch.float32 if device in ["cpu", "mps"] else torch.float16
            load_in_8bit = use_8bit and device == "cuda"
            
            self.logger.info(f"Loading model: {model_name} on {device} with {dtype}")
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token,
                trust_remote_code=True,
                torch_dtype=dtype,
                load_in_8bit=load_in_8bit,
                device_map=None,
                low_cpu_mem_usage=True
            )
            
            if adapter_path:
                self.logger.info(f"Loading PEFT adapter from: {adapter_path}")
                model = PeftModel.from_pretrained(model, adapter_path)
                self.logger.info(f"Adapter loaded successfully. Trainable params: {model.print_trainable_parameters()}")

            
            model = model.to(device)
            model.eval()
            
            for param in model.parameters():
                param.requires_grad = False
            
            yield model, tokenizer
        
        finally:
            if model is not None:
                del model
            if tokenizer is not None:
                del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def compare_models(self, base_model_name: str, finetuned_model_path: str,
                      prompts_file: str, output_file: str, device: str = "cpu",
                      max_new_tokens: int = 100) -> List[Dict]:
        self.logger.info("="*60)
        self.logger.info("Starting Model Comparison")
        self.logger.info("="*60)
        
        prompts = self._load_prompts(prompts_file)
        if not prompts:
            self.logger.error("No prompts loaded.")
            return []
        
        self.logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
        
        base_responses = []
        finetuned_responses = []
        
        try:
            self.logger.info("\nGenerating responses from BASE MODEL")
            with self._load_model_context(base_model_name, None, device) as (base_model, tokenizer):
                for idx, prompt in enumerate(prompts):
                    self.logger.info(f"Base model - Prompt {idx + 1}/{len(prompts)}")
                    response = self._generate_response(base_model, tokenizer, prompt, device, max_new_tokens)
                    base_responses.append(response)
            
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            
            self.logger.info("\nGenerating responses from FINE-TUNED MODEL")
            with self._load_model_context(base_model_name, finetuned_model_path, device) as (ft_model, tokenizer):
                for idx, prompt in enumerate(prompts):
                    self.logger.info(f"Fine-tuned - Prompt {idx + 1}/{len(prompts)}")
                    response = self._generate_response(ft_model, tokenizer, prompt, device, max_new_tokens)
                    finetuned_responses.append(response)
            
            comparisons = []
            for idx, (prompt, base_resp, ft_resp) in enumerate(zip(prompts, base_responses, finetuned_responses)):
                comparisons.append({
                    "prompt_id": idx + 1,
                    "prompt": prompt,
                    "base_model_response": base_resp,
                    "finetuned_model_response": ft_resp,
                    "response_length_diff": len(ft_resp) - len(base_resp)
                })
            
            self._save_comparisons(comparisons, output_file)
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Comparison complete! Results: {output_file}")
            self.logger.info(f"{'='*60}")
            return comparisons
        
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _generate_response(self, model, tokenizer, prompt: str, device: str, max_new_tokens: int = 100) -> str:
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024, padding=False)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=20,
                    do_sample=False,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            
            generated_ids = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            del inputs, input_ids, attention_mask, outputs, generated_ids
            return response.strip()
        
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            return f"[Error: {str(e)}]"

    def _load_prompts(self, prompts_file: str) -> List[str]:
        if not os.path.exists(prompts_file):
            self.logger.error(f"File not found: {prompts_file}")
            return []
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
            if '\n\n' not in content:
                prompts = [line.strip() for line in content.split('\n') if line.strip()]
            else:
                prompts = [p.strip() for p in content.split('\n\n') if p.strip()]
            return prompts
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
            return []

    def _save_comparisons(self, comparisons: List[Dict], output_file: str):
        try:
            os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(comparisons, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved {len(comparisons)} comparisons to {output_file}")
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")
