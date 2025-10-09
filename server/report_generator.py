import json
import torch
from transformers import pipeline

class ReportGenerator:
    def __init__(self, config):
        self.config = config
        self.report_path = config.get("report_path", "./results/report.json")
        self.test_prompts = config.get("test_prompts", [])
    
    def generate_comparison_report(self, base_model, base_tokenizer, 
                                   finetuned_model, finetuned_tokenizer):
        """Generate side-by-side comparison of base vs finetuned model"""
        comparisons = []
        
        for prompt in self.test_prompts:
            base_output = self._generate_response(base_model, base_tokenizer, prompt)
            finetuned_output = self._generate_response(finetuned_model, finetuned_tokenizer, prompt)
            
            comparisons.append({
                "prompt": prompt,
                "base_model_response": base_output,
                "finetuned_model_response": finetuned_output,
            })
        
        report = {
            "model_comparison": comparisons,
            "metadata": {
                "base_model": self.config["model"]["name"],
                "technique": self.config["fine_tuning"]["technique"],
                "epochs": self.config["fine_tuning"]["epochs"],
            }
        }
        
        with open(self.report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_response(self, model, tokenizer, prompt, max_new_tokens=100):
        """Generate response from model for given prompt"""
        inputs = tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
            )
        
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
