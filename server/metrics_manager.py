import time
import json
import os
from datetime import datetime
from transformers import TrainerCallback
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from pymongo import MongoClient, ASCENDING

class MetricsCallback(TrainerCallback):
    def __init__(self, config, timestamp=None):
        self.debug = []
        self.config = config
        self.log_interval = int(config.get("log_interval", 10))
        self.step_metrics = []
        self.epoch_metrics = []
        self.eval_metrics = []
        self.epoch_start_time = None
        if timestamp:
            self.run_timestamp = timestamp
        else:
            self.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = config.get("project", {}).get("project_name", "default_project")
        model_name = config.get("model", {}).get("model_name", "unknown_model").replace("/", "_")
        
        self.metrics_dir = os.path.join("results", project_name, model_name, self.run_timestamp)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        self.step_metrics_path = os.path.join(self.metrics_dir, "step_metrics.json")
        self.epoch_metrics_path = os.path.join(self.metrics_dir, "epoch_metrics.json")
        self.final_metrics_path = os.path.join(self.metrics_dir, "final_metrics.json")
        
        print(f"Metrics will be saved to: {self.metrics_dir}")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        from datetime import datetime

        last_log = state.log_history[-1] if state.log_history else {}
        step_metric = {
            "step": state.global_step,
            "timestamp": datetime.now().isoformat(),
        }
        for k, v in last_log.items():
            if k not in ("step", "epoch"):
                step_metric[k] = v
        self.step_metrics.append(step_metric)
        with open(self.step_metrics_path, 'w') as f:
            json.dump(self.step_metrics, f, indent=2)

    def on_epoch_begin(self, args, state, control, **kwargs):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        import time
        from datetime import datetime

        epoch_time = time.time() - self.epoch_start_time if self.epoch_start_time else None

        eval_log = None
        for log in reversed(state.log_history):
            if any(k.startswith("eval_") for k in log.keys()):
                eval_log = log
                break

        metrics = {
            "epoch": int(state.epoch) if state.epoch is not None else None,
            "epoch_time": epoch_time,
            "timestamp": datetime.now().isoformat(),
        }

        for log in reversed(state.log_history):
            if "loss" in log:
                metrics["train_loss"] = log["loss"]
                break

        if eval_log:
            for k, v in eval_log.items():
                if k.startswith("eval_"):
                    metrics[k] = v

        self.epoch_metrics.append(metrics)

        with open(self.epoch_metrics_path, 'w') as f:
            json.dump(self.epoch_metrics, f, indent=2)

    def set_database_manager(self, db_manager, run_id):
        self.db_manager = db_manager
        self.run_id = run_id

    def _upload_to_mongodb(self):
        MONGO_URI = self.config['database']['mongodb_uri']
        DB_NAME = 'llm_finetuning'

        step_metrics_path = self.step_metrics_path
        epoch_metrics_path = self.epoch_metrics_path
        final_metrics_path = self.final_metrics_path

        project_name = self.config['project']['project_name']
        model_name = self.config['model']['model_name']
        timestamp = self.run_timestamp  
        run_id = self.run_id            

        client = MongoClient(MONGO_URI)
        db = client[DB_NAME]

        for col_name in ['step_metrics', 'epoch_metrics', 'final_metrics']:
            col = db[col_name]
            col.create_index([('project_name', ASCENDING)])
            col.create_index([('model_name', ASCENDING)])
            col.create_index([('timestamp', ASCENDING)])
            col.create_index([('run_id', ASCENDING)])

        with open(step_metrics_path, 'r') as f:
            step_metrics = json.load(f)
        for entry in step_metrics:
            entry['project_name'] = project_name
            entry['model_name'] = model_name
            entry['timestamp'] = timestamp
            entry['run_id'] = run_id
            db.step_metrics.insert_one(entry)

        with open(epoch_metrics_path, 'r') as f:
            epoch_metrics = json.load(f)
        for entry in epoch_metrics:
            entry['project_name'] = project_name
            entry['model_name'] = model_name
            entry['timestamp'] = timestamp
            entry['run_id'] = run_id
            db.epoch_metrics.insert_one(entry)

        with open(final_metrics_path, 'r') as f:
            final_metrics = json.load(f)
        final_metrics['project_name'] = project_name
        final_metrics['model_name'] = model_name
        final_metrics['timestamp'] = timestamp
        final_metrics['run_id'] = run_id
        db.final_metrics.insert_one(final_metrics)

        client.close()
        print(f"✓ Uploaded metrics for {project_name}/{model_name}/{timestamp} (run: {run_id})")

    def on_train_end(self, args, state, control, **kwargs):
        import os, json
        from datetime import datetime

        with open(self.step_metrics_path, 'r') as f:
            step_metrics = json.load(f)
        with open(self.epoch_metrics_path, 'r') as f:
            epoch_metrics = json.load(f)

        all_metric_keys = set()
        for m in epoch_metrics:
            all_metric_keys.update(m.keys())
        eval_metric_keys = [k for k in all_metric_keys if k.startswith('eval_')]
        common_keys = ['train_loss', 'eval_loss', 'epoch_time']

        training_summary = {
            "total_steps": state.global_step,
            "total_epochs": int(state.epoch),
            "total_training_time": sum(m.get('epoch_time', 0) for m in epoch_metrics),
            "avg_epoch_time": sum(m.get('epoch_time', 0) for m in epoch_metrics) / len(epoch_metrics) if epoch_metrics else 0,
            "final_train_loss": epoch_metrics[-1].get('train_loss') if epoch_metrics else None,
            "final_eval_loss": epoch_metrics[-1].get('eval_loss') if epoch_metrics else None,
            "best_train_loss": min((m.get('train_loss') for m in epoch_metrics if m.get('train_loss') is not None), default=None),
            "best_eval_loss": min((m.get('eval_loss') for m in epoch_metrics if m.get('eval_loss') is not None), default=None),
        }
        for metric_key in eval_metric_keys:
            values = [m.get(metric_key) for m in epoch_metrics if m.get(metric_key) is not None]
            if values:
                training_summary[f'best_{metric_key}'] = max(values)
                training_summary[f'final_{metric_key}'] = epoch_metrics[-1].get(metric_key)

        for metric_key in common_keys:
            values = [m.get(metric_key) for m in epoch_metrics if m.get(metric_key) is not None]
            if values:
                training_summary[f'final_{metric_key}'] = epoch_metrics[-1].get(metric_key)
                training_summary[f'best_{metric_key}'] = min(values) if 'loss' in metric_key else max(values)

        final_metrics = {
            "training_summary": training_summary,
            "configuration": {
                "model": self.config.get("model", {}).get("model_name"),
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
        with open(self.final_metrics_path, 'w') as f:
            json.dump(final_metrics, f, indent=2)
        if hasattr(self, 'db_manager') and hasattr(self, 'run_id'):
            self._upload_to_mongodb()
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        print(f"Metrics saved to: {self.metrics_dir}")
        print(f"  - Step metrics: {os.path.basename(self.step_metrics_path)}")
        print(f"  - Epoch metrics: {os.path.basename(self.epoch_metrics_path)}")
        print(f"  - Final metrics: {os.path.basename(self.final_metrics_path)}")
        print(f"{'='*60}\n")

    def save_step_metrics(self, run_id, step_data):
        self.db['step_metrics'].insert_many(step_data)
    def save_epoch_metrics(self, run_id, epoch_data):
        self.db['epoch_metrics'].insert_many(epoch_data)
    def save_final_metrics(self, run_id, final_data):
        self.db['final_metrics'].insert_one(final_data)
# def compute_metrics(eval_pred):
#     """
#     Compute comprehensive metrics for evaluation.
#     Works for both classification and generation tasks.
#     """
#     predictions, labels = eval_pred
    
#     # Handle logits (model outputs)
#     if len(predictions.shape) > 2:
#         # For language modeling: (batch, seq_len, vocab_size)
#         predictions = predictions.argmax(axis=-1)
#     elif len(predictions.shape) == 2 and predictions.shape[1] > 1:
#         # For classification: (batch, num_classes)
#         predictions = predictions.argmax(axis=-1)
    
#     # Filter out padding tokens (-100) for language modeling
#     mask = labels != -100
    
#     # For sequence tasks, flatten
#     if len(labels.shape) > 1:
#         predictions_flat = predictions[mask]
#         labels_flat = labels[mask]
#     else:
#         predictions_flat = predictions
#         labels_flat = labels
    
#     # Ensure we have valid data
#     if len(predictions_flat) == 0 or len(labels_flat) == 0:
#         return {
#             "accuracy": 0.0,
#             "f1": 0.0,
#             "precision": 0.0,
#             "recall": 0.0,
#         }
    
#     metrics = {}
    
#     try:
#         # Classification metrics
#         metrics["accuracy"] = float(accuracy_score(labels_flat, predictions_flat))
#         metrics["f1"] = float(f1_score(labels_flat, predictions_flat, average="weighted", zero_division=0))
#         metrics["precision"] = float(precision_score(labels_flat, predictions_flat, average="weighted", zero_division=0))
#         metrics["recall"] = float(recall_score(labels_flat, predictions_flat, average="weighted", zero_division=0))
#     except Exception as e:
#         print(f"Warning: Could not compute classification metrics: {e}")
#         metrics["accuracy"] = 0.0
#         metrics["f1"] = 0.0
#         metrics["precision"] = 0.0
#         metrics["recall"] = 0.0
    
#     # Note: ROUGE and BLEU require text strings, not token IDs
#     # These should be computed separately with a tokenizer
#     # We'll add them in a separate evaluation function
    
#     return metrics

import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import evaluate

bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")

def compute_metrics(eval_pred, tokenizer=None, task_type="auto"):
    logits, labels = eval_pred

    if task_type == "auto":
        # Classification: logits [batch, num_classes], labels [batch]
        if logits.ndim == 2 and labels.ndim == 1:
            task_type = "classification"
        # Multi-label: logits [batch, num_labels], labels [batch, num_labels]
        elif logits.ndim == 2 and labels.ndim == 2 and set(np.unique(labels)) <= {0, 1}:
            task_type = "multilabel"
        # Sequence: logits [batch, seq_len, vocab_size], labels [batch, seq_len]
        elif logits.ndim == 3 and labels.ndim == 2:
            task_type = "sequence"
        # Generation: logits [batch, seq_len] or [batch, seq_len, vocab_size]
        else:
            task_type = "other"

    metrics = {}

    if task_type == "classification":
        predictions = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        precision = precision_score(labels, predictions, average="macro", zero_division=0)
        f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        metrics.update({
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })
    elif task_type == "multilabel":
        predictions = (logits > 0.5).astype(int)
        acc = accuracy_score(labels, predictions)
        recall = recall_score(labels, predictions, average="macro", zero_division=0)
        precision = precision_score(labels, predictions, average="macro", zero_division=0)
        f1 = f1_score(labels, predictions, average="macro", zero_division=0)
        metrics.update({
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })
    elif task_type == "sequence":
        predictions = np.argmax(logits, axis=-1)
        mask = labels != -100
        labels_flat = labels[mask].flatten()
        predictions_flat = predictions[mask].flatten()
        acc = accuracy_score(labels_flat, predictions_flat)
        recall = recall_score(labels_flat, predictions_flat, average="macro", zero_division=0)
        precision = precision_score(labels_flat, predictions_flat, average="macro", zero_division=0)
        f1 = f1_score(labels_flat, predictions_flat, average="macro", zero_division=0)
        metrics.update({
            "accuracy": acc,
            "recall": recall,
            "precision": precision,
            "f1": f1,
        })
        if tokenizer is not None:
            preds_text = tokenizer.batch_decode(predictions, skip_special_tokens=True)
            labels_text = tokenizer.batch_decode(labels, skip_special_tokens=True)
            bleu = bleu_metric.compute(predictions=preds_text, references=[[l] for l in labels_text])
            rouge = rouge_metric.compute(predictions=preds_text, references=labels_text)
            metrics["bleu"] = bleu.get("bleu", None)
            metrics["rougeL"] = rouge.get("rougeL", None)
    else:
        metrics = {}

    return metrics

def compute_generation_metrics(predictions, references, tokenizer):
    try:
        from rouge_score import rouge_scorer
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []
        bleu_scores = []
        
        for pred, ref in zip(predictions, references):
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
