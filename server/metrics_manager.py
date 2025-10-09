import time
import json
import os
from datetime import datetime
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np

class MetricsCallback(TrainerCallback):
    def __init__(self, config):
        self.config = config
        self.log_interval = int(config.get("log_interval", 10))
        self.step_metrics = []
        self.epoch_metrics = []
        self.eval_metrics = []
        self.epoch_start_time = None
        
        # Create timestamped output directory
        self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = config.get("project", {}).get("name", "default_project")
        model_name = config.get("model", {}).get("name", "unknown_model").replace("/", "_")
        
        # results/project_name/model_name/run_timestamp/
        self.metrics_dir = os.path.join("results", project_name, model_name, self.run_timestamp)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Define paths for three separate files
        self.step_metrics_path = os.path.join(self.metrics_dir, "step_metrics.json")
        self.epoch_metrics_path = os.path.join(self.metrics_dir, "epoch_metrics.json")
        self.final_metrics_path = os.path.join(self.metrics_dir, "final_metrics.json")
        
        print(f"Metrics will be saved to: {self.metrics_dir}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.log_interval == 0:
            metrics = {
                "step": state.global_step,
                "loss": state.log_history[-1].get("loss") if state.log_history else None,
                "learning_rate": state.log_history[-1].get("learning_rate") if state.log_history else None,
                "grad_norm": state.log_history[-1].get("grad_norm") if state.log_history else None,
                "timestamp": datetime.now().isoformat()
            }
            self.step_metrics.append(metrics)
            
            # Save step metrics incrementally
            with open(self.step_metrics_path, 'w') as f:
                json.dump(self.step_metrics, f, indent=2)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, model, **kwargs):
        epoch_time = time.time() - self.epoch_start_time
        
        # Get latest losses and metrics from log history
        loss = None
        eval_loss = None
        eval_metrics = {}
        
        for log in reversed(state.log_history):
            if 'loss' in log and loss is None:
                loss = log['loss']
            if 'eval_loss' in log and eval_loss is None:
                eval_loss = log['eval_loss']
            # Capture any evaluation metrics
            for key in ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 
                       'eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'eval_bleu']:
                if key in log and key not in eval_metrics:
                    eval_metrics[key] = log[key]
            
            if loss is not None and (eval_loss is not None or 'eval_loss' not in str(state.log_history)):
                break
        
        metrics = {
            "epoch": int(state.epoch),
            "epoch_time": epoch_time,
            "train_loss": loss,
            "eval_loss": eval_loss,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add evaluation metrics if available
        metrics.update(eval_metrics)
        
        self.epoch_metrics.append(metrics)
        
        # Save epoch metrics incrementally
        with open(self.epoch_metrics_path, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)

    def on_train_end(self, args, state, control, **kwargs):
        # Calculate final statistics
        training_summary = {
            "total_steps": state.global_step,
            "total_epochs": int(state.epoch),
            "total_training_time": sum(m['epoch_time'] for m in self.epoch_metrics),
            "avg_epoch_time": sum(m['epoch_time'] for m in self.epoch_metrics) / len(self.epoch_metrics) if self.epoch_metrics else 0,
            "final_train_loss": self.epoch_metrics[-1]['train_loss'] if self.epoch_metrics else None,
            "final_eval_loss": self.epoch_metrics[-1].get('eval_loss') if self.epoch_metrics else None,
            "best_train_loss": min((m['train_loss'] for m in self.epoch_metrics if m['train_loss']), default=None),
            "best_eval_loss": min((m.get('eval_loss') for m in self.epoch_metrics if m.get('eval_loss')), default=None),
        }
        
        # Add best evaluation metrics if available
        eval_metric_keys = ['eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall', 
                           'eval_rouge1', 'eval_rouge2', 'eval_rougeL', 'eval_bleu']
        
        for metric_key in eval_metric_keys:
            values = [m.get(metric_key) for m in self.epoch_metrics if m.get(metric_key) is not None]
            if values:
                training_summary[f'best_{metric_key}'] = max(values)
                training_summary[f'final_{metric_key}'] = self.epoch_metrics[-1].get(metric_key)
        
        final_metrics = {
            "training_summary": training_summary,
            "configuration": {
                "model": self.config.get("model", {}).get("name"),
                "technique": self.config.get("fine_tuning", {}).get("technique"),
                "learning_rate": self.config.get("fine_tuning", {}).get("learning_rate"),
                "batch_size": self.config.get("fine_tuning", {}).get("batch_size"),
                "epochs": self.config.get("fine_tuning", {}).get("epochs"),
                "lora_r": self.config.get("lora_r"),
                "lora_alpha": self.config.get("lora_alpha"),
                "dataset_type": self.config.get("training_data", {}).get("dataset_type"),
            },
            "metrics_files": {
                "step_metrics": self.step_metrics_path,
                "epoch_metrics": self.epoch_metrics_path,
                "final_metrics": self.final_metrics_path,
            },
            "run_info": {
                "timestamp": self.run_timestamp,
                "metrics_directory": self.metrics_dir,
                "completed_at": datetime.now().isoformat()
            }
        }
        
        # Save final comprehensive metrics
        with open(self.final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Metrics saved to: {self.metrics_dir}")
        print(f"  - Step metrics: {os.path.basename(self.step_metrics_path)}")
        print(f"  - Epoch metrics: {os.path.basename(self.epoch_metrics_path)}")
        print(f"  - Final metrics: {os.path.basename(self.final_metrics_path)}")
        print(f"{'='*60}\n")


def compute_metrics(eval_pred):
    """
    Compute comprehensive metrics for evaluation.
    Works for both classification and generation tasks.
    """
    predictions, labels = eval_pred
    
    # Handle logits (model outputs)
    if len(predictions.shape) > 2:
        # For language modeling: (batch, seq_len, vocab_size)
        predictions = predictions.argmax(axis=-1)
    elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
        # For classification: (batch, num_classes)
        predictions = predictions.argmax(axis=-1)
    
    # Filter out padding tokens (-100) for language modeling
    mask = labels != -100
    
    # For sequence tasks, flatten
    if len(labels.shape) > 1:
        predictions_flat = predictions[mask]
        labels_flat = labels[mask]
    else:
        predictions_flat = predictions
        labels_flat = labels
    
    # Ensure we have valid data
    if len(predictions_flat) == 0 or len(labels_flat) == 0:
        return {
            "accuracy": 0.0,
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
    
    metrics = {}
    
    try:
        # Classification metrics
        metrics["accuracy"] = float(accuracy_score(labels_flat, predictions_flat))
        metrics["f1"] = float(f1_score(labels_flat, predictions_flat, average="weighted", zero_division=0))
        metrics["precision"] = float(precision_score(labels_flat, predictions_flat, average="weighted", zero_division=0))
        metrics["recall"] = float(recall_score(labels_flat, predictions_flat, average="weighted", zero_division=0))
    except Exception as e:
        print(f"Warning: Could not compute classification metrics: {e}")
        metrics["accuracy"] = 0.0
        metrics["f1"] = 0.0
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
    
    # Note: ROUGE and BLEU require text strings, not token IDs
    # These should be computed separately with a tokenizer
    # We'll add them in a separate evaluation function
    
    return metrics


def compute_generation_metrics(predictions, references, tokenizer):
    """
    Compute ROUGE and BLEU for text generation tasks.
    Call this separately when you have decoded text.
    """
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        # Initialize ROUGE scorer
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
            # ROUGE scores
            scores = scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)
            
            # BLEU score
            reference_tokens = [ref.split()]
            prediction_tokens = pred.split()
            smoothie = SmoothingFunction().method4
            bleu = sentence_bleu(reference_tokens, prediction_tokens, smoothing_function=smoothie)
            bleu_scores.append(bleu)
        
        return {
            "rouge1": float(np.mean(rouge1_scores)),
            "rouge2": float(np.mean(rouge2_scores)),
            "rougeL": float(np.mean(rougeL_scores)),
            "bleu": float(np.mean(bleu_scores)),
        }
    
    except ImportError:
        print("Warning: rouge-score or nltk not installed. Install with: pip install rouge-score nltk")
        return {}
