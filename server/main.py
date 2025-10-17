import argparse
import os
import torch
import json
import logging
import glob
from datetime import datetime
from config_parser import load_config_from_env, merge_configs
from model_manager import load_model_from_config
from dataset_manager import DatasetManager
from finetuning_method_manager import finetune_model
from model_comparator import ModelComparator
from databse_manager import DatabaseManager  

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Framework")
    
    parser.add_argument("--config", "-config", type=str, default=None, help="Path to configuration .env file")
    
    parser.add_argument("--project", "--project_name", type=str, default=None, help="Project name")
    parser.add_argument("--model", "--model_name", type=str, default=None, help="Model name or path")
    parser.add_argument("--tokenizer", "--tokenizer", type=str, default=None, help="Tokenizer name or path")
    
    parser.add_argument("--data", "--data_path", type=str, default=None, help="Path to training data file")
    parser.add_argument("--dataset_type", "--dataset_type", type=str, default=None, help="Dataset type (mcq_jsonl, instruction_jsonl, text)") #TODO: update the list
    parser.add_argument("--val_path", type=str, default=None, help="Path to validation data file")
    parser.add_argument("--test_path", type=str, default=None, help="Path to test data file")
    parser.add_argument("--technique", type=str, default=None, help="Fine-tuning technique (lora, qlora, dora)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Training device (cpu, cuda, mps, auto)")
    parser.add_argument("--lora_r", type=int, default=None, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=None, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=None, help="LoRA dropout")
    parser.add_argument("--output", "--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--prompts_file", type=str, default=None, help="Path to prompts file for model comparison")
    
    return parser.parse_args()



def main():
    args = parse_args()
    
    logger.info("Loading default configuration from /Users/shonil/ft/finetuning/server/default.env")
    default_config = load_config_from_env("default.env")
    config = default_config
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.config:
        logger.info(f"Loading custom configuration from {args.config}")
        custom_config = load_config_from_env(args.config)
        config = merge_configs(config, custom_config)
    logger.info(f"Final merged configuration: {config}")
    logger.info("----------------------------------")
    if any(vars(args).values()):
        logger.info("Applying CLI argument overrides")
        cli_overrides = {}
        if args.project: cli_overrides.setdefault("project", {})["project_name"] = args.project
        if args.epochs: cli_overrides.setdefault("training", {})["epochs"] = str(args.epochs)
        if args.batch_size: cli_overrides.setdefault("training", {})["batch_size"] = str(args.batch_size)
        if args.learning_rate: cli_overrides.setdefault("training", {})["learning_rate"] = str(args.learning_rate)
        if args.device: cli_overrides.setdefault("training", {})["device"] = args.device
        if args.output: cli_overrides.setdefault("training", {})["output_dir"] = args.output
        if args.prompts_file: cli_overrides.setdefault("training", {})["prompts_file"] = args.prompts_file
        
        if args.model: cli_overrides.setdefault("model", {})["model_name"] = args.model
        if args.tokenizer: cli_overrides.setdefault("model", {})["tokenizer"] = args.tokenizer
        
        if args.data: cli_overrides.setdefault("data", {})["train_data_path"] = args.data
        if args.dataset_type: cli_overrides.setdefault("data", {})["dataset_type"] = args.dataset_type
        if args.val_path: cli_overrides.setdefault("data", {})["val_data_path"] = args.val_path
        if args.test_path: cli_overrides.setdefault("data", {})["test_data_path"] = args.test_path
        
        if args.technique: cli_overrides.setdefault("fine_tuning", {})["technique"] = args.technique
        if args.lora_r: cli_overrides.setdefault("fine_tuning", {})["lora_r"] = str(args.lora_r)
        if args.lora_alpha: cli_overrides.setdefault("fine_tuning", {})["lora_alpha"] = str(args.lora_alpha)
        if args.lora_dropout: cli_overrides.setdefault("fine_tuning", {})["lora_dropout"] = str(args.lora_dropout)
        
        config = merge_configs(config, cli_overrides)
    
    logger.info(f"Project: {config['project']['project_name']}")
    logger.info(f"Model: {config['model']['model_name']}")
    logger.info(f"Data: {config['data']['train_data_path']}")
    logger.info(f"Dataset Type: {config['data']['dataset_type']}")
    logger.info(f"Technique: {config['fine_tuning']['technique']}")
    logger.info(f"Output: {config['training'].get('output_dir', './results')}")
    
    db_manager = DatabaseManager(config)
    run_metadata = {
        "project_name": config["project"]["project_name"],
        "model_name": config["model"]["model_name"],
        "dataset_type": config["data"]["dataset_type"],
        "technique": config["fine_tuning"]["technique"],
        "configuration": config
    }
    run_id = db_manager.create_training_run(run_metadata)
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_from_config(config)
    
    logger.info("Loading dataset...")
    dataset_manager = DatasetManager(config, tokenizer)
    train_dataset, val_dataset, test_dataset = dataset_manager.load_and_split(timestamp)
    logger.info(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset) if val_dataset else 0}, test={len(test_dataset) if test_dataset else 0}")
    logger.info("Starting fine-tuning...")
    finetuned_model, tokenizer, metrics_dir = finetune_model(
        model, 
        tokenizer, 
        train_dataset, 
        config, 
        val_dataset,
        timestamp
    )
    
    logger.info(f"Fine-tuning complete! Model saved to {config['training'].get('output_dir', './results')}")
    if run_id and db_manager.enabled:
        try:
            logger.info("Uploading metrics to database...")
            logger.info(f"Metrics directory: {metrics_dir}")
            step_metrics_file = os.path.join(metrics_dir, "step_metrics.json")
            if os.path.exists(step_metrics_file):
                with open(step_metrics_file, 'r') as f:
                    step_metrics = json.load(f)
                db_manager.save_step_metrics(run_id, step_metrics,config["project"]["project_name"], config["model"]["model_name"],timestamp)
            
            epoch_metrics_file = os.path.join(metrics_dir, "epoch_metrics.json")
            if os.path.exists(epoch_metrics_file):
                with open(epoch_metrics_file, 'r') as f:
                    epoch_metrics = json.load(f)
                db_manager.save_epoch_metrics(run_id, epoch_metrics,config["project"]["project_name"], config["model"]["model_name"],timestamp)

            final_metrics_file = os.path.join(metrics_dir, "final_metrics.json")
            if os.path.exists(final_metrics_file):
                with open(final_metrics_file, 'r') as f:
                    final_metrics = json.load(f)
                db_manager.save_final_metrics(run_id, final_metrics,config["project"]["project_name"], config["model"]["model_name"],timestamp)
            
            logger.info("Successfully uploaded all metrics to database")
            
        except Exception as e:
            logger.error(f"Failed to upload metrics to database: {e}")
    
    try:    
        comparator = ModelComparator(config)
        comparator_file = os.path.join(metrics_dir, "evaluation_comparator.json")
        project_name = config["project"]["project_name"]
        model_name = config["model"]["model_name"]
        with open(comparator_file, "w", encoding="utf-8") as f:
            json.dump({
                "project_name": project_name,
                "model_name": model_name,
                "timestamp": timestamp
            }, f, indent=2)
        device = config["fine_tuning"].get("device", "cpu")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        
        finetuned_model_path = os.path.join(metrics_dir, "model")
        split_files = glob.glob(os.path.join(metrics_dir, '**', 'test_split.jsonl'), recursive=True)
        test_split_path = split_files[0] if split_files else None
        logger.info(f"metrics_dir: {metrics_dir}, test_split_path: {test_split_path}")
        # 1. Test Set Evaluation (quantitative)
        if test_split_path and os.path.exists(test_split_path):
            logger.info("\n" + "----------------------------------")
            logger.info("TEST DATASET EVALUATION WITH ADVANCED METRICS:")
            test_output = os.path.join(metrics_dir, "evaluation_comparator.json")
            test_comparison = comparator.compare_models_on_test_file(
                base_model_name=config['model']['model_name'],
                finetuned_model_path=finetuned_model_path,
                test_file_path=test_split_path,
                output_file=test_output,
                device=device,
                max_new_tokens=config.get("comparison", {}).get("max_new_tokens", 100)
            )
            if run_id and db_manager.enabled and test_comparison:
                db_manager.save_model_comparison(run_id, test_comparison)
            logger.info(f"Test evaluation saved to: {test_output}")
        else:
            logger.warning("No test_split.jsonl found. Skipping test evaluation.")
        
        # 2. Prompts-Based Comparison (qualitative)
        prompts_file = config.get("comparison", {}).get("prompts_file") or args.prompts_file
        if prompts_file and os.path.exists(prompts_file):
            logger.info("\n" + "----------------------------------")
            logger.info("PROMPTS-BASED QUALITATIVE COMPARISON")
            prompts_output = os.path.join(metrics_dir, "evaluation_comparator.json")
            prompts_comparison = comparator.compare_models_with_prompts(
                base_model_name=config['model']['model_name'],
                finetuned_model_path=finetuned_model_path,
                prompts_file=prompts_file,
                output_file=prompts_output,
                device=device,
                max_new_tokens=config.get("comparison", {}).get("max_new_tokens", 100)
            )
            logger.info(f"Prompts comparison saved to: {prompts_output}")
        if db_manager.enabled and os.path.exists(comparator_file):
            with open(comparator_file, "r", encoding="utf-8") as f:
                eval_data = json.load(f)
            db_manager.save_evaluation_comparator(eval_data)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
        db_manager.close()

if __name__ == "__main__":
    main()