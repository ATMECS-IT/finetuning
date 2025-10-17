import os
import json
import torch
import logging
import gc
import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import List, Dict, Optional
from contextlib import contextmanager
from model_manager import load_model_from_config
from dataset_manager import DatasetFormatterFactory
import re

class ModelComparator:    
    def __init__(self, config: Dict):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    @contextmanager
    def _load_model_context(self, model_name, adapter_path=None, device="cpu", use_8bit=False):
        model, tokenizer = load_model_from_config(self.config)
        if adapter_path:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, adapter_path)
        model = model.to(device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        try:
            yield model, tokenizer
        finally:
            del model
            del tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()

    def compare_models_on_test_file(self, base_model_name: str, finetuned_model_path: str,
                                test_file_path: str, output_file: str, device: str = "cpu",
                                max_new_tokens: int = 100) -> Dict:
        self.logger.info("---------")
        self.logger.info("Starting Test Dataset Evaluation")
        try:
            with open(test_file_path, 'r', encoding='utf-8') as f:
                test_data = [json.loads(line) for line in f if line.strip()]
        except Exception as e:
            self.logger.error(f"Failed to load test file: {e}")
            return {}
        if not test_data:
            self.logger.error("Test file is empty")
            return {}
        self.logger.info(f"Loaded {len(test_data)} test samples from {test_file_path}")
        base_metrics = {"responses": [], "labels": []}
        finetuned_metrics = {"responses": [], "labels": []}
        prompts = []
        dataset_type = self.config.get('training_data', {}).get('dataset_type', 'text')
        
        try:
            self.logger.info("\n Generating from BASE MODEL")
            with self._load_model_context(base_model_name, None, device) as (base_model, tokenizer):
                for idx, sample in enumerate(test_data):
                    self.logger.info(f"  Base - Sample {idx + 1}/{len(test_data)}")
                    prompt, label = self._extract_test_sample_fields(sample, dataset_type)
                    
                    if not prompt or len(prompt.strip()) == 0:
                        self.logger.warning(f"  Skipping empty prompt at index {idx}")
                        continue
                    prompts.append(prompt)
                    base_metrics["labels"].append(label)
                    response = self._generate_response(base_model, tokenizer, prompt, device, max_new_tokens)
                    base_metrics["responses"].append(response)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            
            # Generate from FINE-TUNED MODEL
            self.logger.info("\n Generating from FINE-TUNED MODEL")
            with self._load_model_context(base_model_name, finetuned_model_path, device) as (ft_model, tokenizer):
                for idx in range(len(prompts)):
                    self.logger.info(f"  Fine-tuned - Sample {idx + 1}/{len(prompts)}")
                    
                    prompt = prompts[idx]
                    label = base_metrics["labels"][idx]
                    finetuned_metrics["labels"].append(label)
                    response = self._generate_response(ft_model, tokenizer, prompt, device, max_new_tokens)
                    finetuned_metrics["responses"].append(response)          
            # Compute metrics
            self.logger.info("\nComputing metrics...")
            base_eval = self._compute_all_metrics(
                base_metrics["responses"],
                base_metrics["labels"],
                prompts
            )
            finetuned_eval = self._compute_all_metrics(
                finetuned_metrics["responses"],
                finetuned_metrics["labels"],
                prompts
            )
            comparison_report = {
                "base_model_metrics": base_eval,
                "finetuned_model_metrics": finetuned_eval,
                "sample_comparisons": []
            }
            for idx in range(min(len(prompts), 2)):
                comparison_report["sample_comparisons"].append({
                    "prompt_id": idx + 1,
                    "prompt": prompts[idx][:200] + "..." if len(prompts[idx]) > 200 else prompts[idx],
                    "label": base_metrics["labels"][idx],
                    "base_model_response": base_metrics["responses"][idx],
                    "finetuned_model_response": finetuned_metrics["responses"][idx],
                    "response_length_diff": len(finetuned_metrics["responses"][idx]) - len(base_metrics["responses"][idx])
                })
            
            self._save_comparisons(comparison_report, output_file)
            
            self.logger.info(f"Evaluation Complete! Saved to {output_file}")
            self.logger.info(f"{'--------------------------------'}")
            
            return comparison_report
        
        except Exception as e:
            self.logger.error(f"Error during evaluation: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def compare_models_with_prompts(self, base_model_name: str, finetuned_model_path: str,
                                   prompts_file: str, output_file: str, device: str = "cpu",
                                   max_new_tokens: int = 150) -> Dict:
        self.logger.info(f"{'--------------------------------'}")
        self.logger.info("Starting Prompts-Based Comparison")
        
        prompts = self._load_prompts(prompts_file)
        if not prompts:
            self.logger.error(f"No prompts loaded from {prompts_file}")
            return {}
        
        self.logger.info(f"Loaded {len(prompts)} prompts")
        comparisons = []
        
        try:
            # BASE MODEL
            self.logger.info("\nGenerating from BASE MODEL")
            base_responses = []
            with self._load_model_context(base_model_name, None, device) as (base_model, tokenizer):
                for idx, prompt in enumerate(prompts):
                    self.logger.info(f"  Base - Prompt {idx + 1}/{len(prompts)}")
                    response = self._generate_response(base_model, tokenizer, prompt, device, max_new_tokens)
                    base_responses.append(response)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
            elif device == "mps":
                torch.mps.empty_cache()
            
            # FINE-TUNED MODEL
            self.logger.info("\nGenerating from FINE-TUNED MODEL")
            finetuned_responses = []
            with self._load_model_context(base_model_name, finetuned_model_path, device) as (ft_model, tokenizer):
                for idx, prompt in enumerate(prompts):
                    self.logger.info(f"  Fine-tuned - Prompt {idx + 1}/{len(prompts)}")
                    response = self._generate_response(ft_model, tokenizer, prompt, device, max_new_tokens)
                    finetuned_responses.append(response)
            
            # Build comparisons
            for idx, prompt in enumerate(prompts):
                comparisons.append({
                    "prompt_id": idx + 1,
                    "prompt": prompt,
                    "base_model_response": base_responses[idx],
                    "finetuned_model_response": finetuned_responses[idx],
                    "response_length_diff": len(finetuned_responses[idx]) - len(base_responses[idx])
                })
            self._save_comparisons({"comparisons": comparisons}, output_file)
            self.logger.info(f"Prompts Comparison Complete!")
            self.logger.info(f"{'--------------------------------'}")
            return {"comparisons": comparisons}
        
        except Exception as e:
            self.logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return {}

    def _compute_all_metrics(self, responses: List[str], labels: List[str], 
                            prompts: List[str]) -> Dict:
        metrics = {} 
        try:
            #TODO: implement for generic classification
            # if len(set(labels)) > 1:
            #     correct = sum(1 for p, l in zip(responses, labels) 
            #                 if p.strip().lower() == l.strip().lower())
            #     metrics["accuracy"] = correct / len(labels) if labels else 0.0
                
            #     try:
            #         binary_preds = [1 if p.strip().lower() == l.strip().lower() else 0 
            #                       for p, l in zip(responses, labels)]
            #         binary_labels = [1] * len(labels)
                    
            #         if len(set(binary_preds)) > 1:
            #             metrics["mcc"] = matthews_corrcoef(binary_labels, binary_preds)
            #         else:
            #             metrics["mcc"] = 0.0
                    
            #         precision, recall, f1, _ = precision_recall_fscore_support(
            #             binary_labels, binary_preds, average='binary', zero_division=0
            #         )
            #         metrics["precision"] = float(precision)
            #         metrics["recall"] = float(recall)
            #         metrics["f1"] = float(f1)
            #     except Exception as e:
            #         self.logger.warning(f"Classification metrics error: {e}")
            #         metrics["mcc"] = 0.0
            #         metrics["precision"] = 0.0
            #         metrics["recall"] = 0.0
            #         metrics["f1"] = 0.0
            # else:
            #     metrics["accuracy"] = 0.0
            #     metrics["mcc"] = 0.0
            #     metrics["precision"] = 0.0
            #     metrics["recall"] = 0.0
            #     metrics["f1"] = 0.0
            
            # BERTScore (optional - can be slow)
            try:
                from bert_score import score as bert_score_fn
                self.logger.info("  Computing BERTScore...")
                P, R, F1 = bert_score_fn(responses, labels, lang="en", verbose=False)
                metrics["bertscore_precision"] = float(P.mean())
                metrics["bertscore_recall"] = float(R.mean())
                metrics["bertscore_f1"] = float(F1.mean())
            except ImportError:
                self.logger.warning("bert_score not installed, skipping")
                metrics["bertscore_precision"] = 0.0
                metrics["bertscore_recall"] = 0.0
                metrics["bertscore_f1"] = 0.0
            except Exception as e:
                self.logger.warning(f"BERTScore failed: {e}")
                metrics["bertscore_precision"] = 0.0
                metrics["bertscore_recall"] = 0.0
                metrics["bertscore_f1"] = 0.0
            
            try:
                perplexities = []
                for response in responses:
                    if response and len(response) > 0:
                        words = response.split()
                        unique_ratio = len(set(words)) / len(words) if words else 0
                        perp = 1 / (unique_ratio + 0.01)
                        perplexities.append(perp)
                metrics["perplexity"] = float(np.mean(perplexities)) if perplexities else 100.0
            except Exception as e:
                self.logger.warning(f"Perplexity failed: {e}")
                metrics["perplexity"] = 100.0   

            try:
                smooth_fn = SmoothingFunction().method1
                bleu_scores = [
                    sentence_bleu([label.split()], resp.split(), smoothing_function=smooth_fn)
                    for resp, label in zip(responses, labels)
                ]
                metrics["bleu"] = float(np.mean(bleu_scores))     
            except Exception as e:
                self.logger.warning(f"BLEU computation failed: {e}")
                metrics["bleu"] = 0.0  

            try:
                scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
                rouge_scores = [scorer.score(label, resp) for resp, label in zip(responses, labels)]
                metrics["rouge1"] = float(np.mean([s["rouge1"].fmeasure for s in rouge_scores]))
                metrics["rougeL"] = float(np.mean([s["rougeL"].fmeasure for s in rouge_scores]))  
            except Exception as e:      
                self.logger.warning(f"ROUGE computation failed: {e}")
                metrics["rouge1"] = 0.0
                metrics["rougeL"] = 0.0
        except Exception as e:
            self.logger.error(f"Error computing metrics: {e}")
            import traceback
            traceback.print_exc()
        
        return metrics

    def _generate_response(self, model, tokenizer, prompt: str, device: str, 
                          max_new_tokens: int = 100) -> str:
        try:
            if not prompt or len(prompt.strip()) == 0:
                self.logger.warning("Empty prompt")
                return "[Error: Empty prompt]"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, 
                             max_length=1024, padding=False)
            if inputs["input_ids"].shape[1] == 0:
                return "[Error: Empty tokenization]"
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            #TODO: add generation config options and use while generating actual responses
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=10,
                    do_sample=True,
                    temperature=0.3,  # Lower = more focused (medical domain)
                    top_p=0.85,  # Nucleus sampling
                    top_k=40,  
                    repetition_penalty=1.5,  # Strong penalty for repetition
                    no_repeat_ngram_size=3,  # Prevent 3-gram repetition
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True
                )
            generated_ids = outputs[0][input_ids.shape[1]:]
            response = tokenizer.decode(generated_ids, skip_special_tokens=True)
            response = response.strip()
            del inputs, input_ids, attention_mask, outputs, generated_ids
            return response
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            import traceback
            traceback.print_exc()
            return f"[Error: {str(e)}]"

    def _load_prompts(self, prompts_file: str) -> List[str]:
        try:
            with open(prompts_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '\n\n' in content:
                prompts = [p.strip() for p in content.split('\n\n') if p.strip()]
            else:
                prompts = [line.strip() for line in content.split('\n') if line.strip()]
            
            return prompts
        except Exception as e:
            self.logger.error(f"Error loading prompts: {e}")
            return []

    def _save_comparisons(self, new_data: Dict, output_file: str):
        try:
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
            else:
                existing = {}
            for k, v in new_data.items():
                existing[k] = v
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved to {output_file}")        
        except Exception as e:
            self.logger.error(f"Failed to save: {e}")

    def _extract_test_sample_fields(self, sample, dataset_type=None):
        if dataset_type is None:
            dataset_type = self.config['training_data'].get('dataset_type', 'mcq_jsonl')
        formatter = DatasetFormatterFactory.create_formatter(dataset_type)
        prompt, label = formatter.format_sample(sample)
        return prompt, label