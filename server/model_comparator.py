import json
import os
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging
import concurrent.futures

logger = logging.getLogger(__name__)


class ModelComparator:
    def __init__(self, config, metrics_dir):
        self.config = config
        self.metrics_dir = metrics_dir
        self.comparison_path = os.path.join(metrics_dir, "model_comparison.json")
        self.max_new_tokens = int(config.get("comparison_max_tokens", 150))
        self.temperature = float(config.get("comparison_temperature", 0.7))
        self.top_p = float(config.get("comparison_top_p", 0.9))
        self.do_sample = config.get("comparison_do_sample", "True").lower() == "true"
        
    def load_prompts_from_file(self, prompts_file):
        if not os.path.exists(prompts_file):
            logger.warning(f"Prompts file not found: {prompts_file}")
            return []
        
        prompts = []
        with open(prompts_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
            paragraphs = content.split('\n\n')
            
            for para in paragraphs:
                para = para.strip()
                if para:  
                    prompts.append(para)
        
        logger.info(f"Loaded {len(prompts)} prompts from {prompts_file}")
        return prompts
    
    def generate_response(self, model, tokenizer, prompt, device="cpu"):
        try:
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            if generated_text.startswith(prompt):
                response = generated_text[len(prompt):].strip()
            else:
                response = generated_text
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"[Error: {str(e)}]"

    def compare_models(self, base_model_name, finetuned_model_path, prompts_file, device="cpu", num_workers=1):
        logger.info("="*60)
        logger.info("Starting Model Comparison")
        logger.info("="*60)
        
        prompts = self.load_prompts_from_file(prompts_file)
        
        if not prompts:
            logger.warning("No prompts found. Skipping comparison.")
            return None
        
        logger.info(f"Loading tokenizer from {base_model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            use_auth_token=self.config.get("model", {}).get("use_auth_token", False)
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        logger.info(f"Loading base model: {base_model_name}...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            use_auth_token=self.config.get("model", {}).get("use_auth_token", False)
        )
        base_model.to(device)
        base_model.eval()
        
        logger.info(f"Loading fine-tuned model from: {finetuned_model_path}...")
        finetuned_model = AutoModelForCausalLM.from_pretrained(finetuned_model_path)
        finetuned_model.to(device)
        finetuned_model.eval()
        comparisons = []
        def process_prompt(args):
            idx, prompt = args
            logger.info(f"Processing prompt {idx}/{len(prompts)}...")
            prompt_preview = prompt[:100] + "..." if len(prompt) > 100 else prompt
            base_response = self.generate_response(base_model, tokenizer, prompt, device)
            finetuned_response = self.generate_response(finetuned_model, tokenizer, prompt, device)
            return {
                "prompt_id": idx,
                "prompt": prompt,
                "prompt_preview": prompt_preview,
                "base_model_response": base_response,
                "finetuned_model_response": finetuned_response,
                "timestamp": datetime.now().isoformat()
            }

        prompt_args = list(enumerate(prompts, 1))

        if num_workers == 1:
            for args in prompt_args:
                comparison = process_prompt(args)
                comparisons.append(comparison)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
                for comparison in executor.map(process_prompt, prompt_args):
                    comparisons.append(comparison)
        
        comparison_report = {
            "metadata": {
                "base_model": base_model_name,
                "finetuned_model_path": finetuned_model_path,
                "prompts_file": prompts_file,
                "num_prompts": len(prompts),
                "generation_params": {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "do_sample": self.do_sample
                },
                "device": device,
                "comparison_date": datetime.now().isoformat()
            },
            "comparisons": comparisons
        }
        
        with open(self.comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_report, f, indent=2, ensure_ascii=False)
        
        logger.info("="*60)
        logger.info(f"Model comparison complete!")
        logger.info(f"Results saved to: {self.comparison_path}")
        logger.info(f"Total prompts processed: {len(comparisons)}")
        logger.info("="*60)
        
        del base_model
        del finetuned_model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        return comparison_report
