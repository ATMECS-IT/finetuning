import argparse
import os
import logging
from config_parser import load_config_from_env, merge_configs
from model_manager import load_model_from_config
from dataset_manager import load_dataset_from_config
from finetuning_method_manager import finetune_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="LLM Fine-tuning Framework")
    
    parser.add_argument('-config', '--config', type=str, default=None,
                        help='Path to configuration .env file')
    
    parser.add_argument('-project', '--project_name', type=str, default=None,
                        help='Project name')
    
    parser.add_argument('-model', '--model_name', type=str, default=None,
                        help='Model name or path')
    parser.add_argument('-tokenizer', '--tokenizer', type=str, default=None,
                        help='Tokenizer name or path')
    
    parser.add_argument('-data', '--data_path', type=str, default=None,
                        help='Path to training data file')
    parser.add_argument('-dataset_type', '--dataset_type', type=str, default=None,
                        choices=['plain_text', 'mcq_jsonl', 'instruction_jsonl'],
                        help='Type of dataset')
    parser.add_argument('--val_path', type=str, default=None,
                       help='Path to validation data file (optional)')
    parser.add_argument('--test_path', type=str, default=None,
                       help='Path to test data file (optional)')
    parser.add_argument('--train_ratio', type=float, default=None,
                       help='Training data ratio (if splitting from single file)')
    parser.add_argument('--val_ratio', type=float, default=None,
                       help='Validation data ratio (if splitting from single file)')
    parser.add_argument('--test_ratio', type=float, default=None,
                       help='Test data ratio (if splitting from single file)')
    parser.add_argument('--split_seed', type=int, default=None,
                       help='Random seed for dataset splitting')
    parser.add_argument('-technique', '--technique', type=str, default=None,
                        choices=['lora', 'qlora', 'dora', 'adapter'],
                        help='Fine-tuning technique')
    parser.add_argument('-savepath', '--save_path', type=str, default=None,
                        help='Path where model will be saved')
    parser.add_argument('-epochs', '--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('-batch_size', '--batch_size', type=int, default=None,
                        help='Batch size for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('-device', '--device', type=str, default=None,
                        choices=['auto', 'cuda', 'mps', 'cpu'],
                        help='Device to use for training')
    parser.add_argument('--eval_steps', type=int, default=None,
                       help='Evaluate every N steps')
    
    parser.add_argument('-lora_r', '--lora_r', type=int, default=None,
                        help='LoRA rank')
    parser.add_argument('-lora_alpha', '--lora_alpha', type=int, default=None,
                        help='LoRA alpha parameter')
    parser.add_argument('-lora_dropout', '--lora_dropout', type=float, default=None,
                        help='LoRA dropout rate')
    
    return parser.parse_args()


def build_config_from_args(args):
    cli_config = {}
    
    if args.project_name:
        cli_config.setdefault('project', {})['name'] = args.project_name
    
    if args.model_name:
        cli_config.setdefault('model', {})['name'] = args.model_name
    if args.tokenizer:
        cli_config.setdefault('model', {})['tokenizer'] = args.tokenizer
    
    if args.data_path:
        cli_config.setdefault('training_data', {})['path'] = args.data_path
    if args.dataset_type:
        cli_config.setdefault('training_data', {})['dataset_type'] = args.dataset_type
    if args.val_path:
        cli_config.setdefault('training_data', {})['val_path'] = args.val_path
    if args.test_path:
        cli_config.setdefault('training_data', {})['test_path'] = args.test_path
    if args.train_ratio is not None:
        cli_config.setdefault('training_data', {})['train_ratio'] = str(args.train_ratio)
    if args.val_ratio is not None:
        cli_config.setdefault('training_data', {})['val_ratio'] = str(args.val_ratio)
    if args.test_ratio is not None:
        cli_config.setdefault('training_data', {})['test_ratio'] = str(args.test_ratio)
    if args.split_seed is not None:
        cli_config.setdefault('training_data', {})['split_seed'] = str(args.split_seed)
    
    if args.technique:
        cli_config.setdefault('fine_tuning', {})['technique'] = args.technique
    if args.save_path:
        cli_config.setdefault('fine_tuning', {})['output_dir'] = args.save_path
    if args.epochs:
        cli_config.setdefault('fine_tuning', {})['epochs'] = str(args.epochs)
    if args.batch_size:
        cli_config.setdefault('fine_tuning', {})['batch_size'] = str(args.batch_size)
    if args.learning_rate:
        cli_config.setdefault('fine_tuning', {})['learning_rate'] = str(args.learning_rate)
    if args.device:
        cli_config.setdefault('fine_tuning', {})['device'] = args.device
    
    if args.eval_steps is not None:
        cli_config['eval_steps'] = str(args.eval_steps)
    
    if args.lora_r:
        cli_config['lora_r'] = str(args.lora_r)
    if args.lora_alpha:
        cli_config['lora_alpha'] = str(args.lora_alpha)
    if args.lora_dropout:
        cli_config['lora_dropout'] = str(args.lora_dropout)
    
    return cli_config


def main():
    args = parse_args()
    
    default_config_path = os.path.join(os.path.dirname(__file__), 'default.env')
    logger.info(f"Loading default configuration from {default_config_path}")
    config = load_config_from_env(default_config_path)
    
    if args.config:
        logger.info(f"Loading custom configuration from {args.config}")
        custom_config = load_config_from_env(args.config)
        config = merge_configs(config, custom_config)
    
    cli_config = build_config_from_args(args)
    if cli_config:
        logger.info("Applying CLI argument overrides")
        config = merge_configs(config, cli_config)
    
    logger.info(f"Project: {config['project']['name']}")
    logger.info(f"Model: {config['model']['name']}")
    logger.info(f"Data: {config['training_data']['path']}")
    logger.info(f"Technique: {config['fine_tuning']['technique']}")
    logger.info(f"Output: {config['fine_tuning']['output_dir']}")
    
    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_from_config(config)
    
    logger.info("Loading dataset...")
    datasets = load_dataset_from_config(config, tokenizer)
    
    train_dataset = datasets['train']
    val_dataset = datasets.get('val')
    test_dataset = datasets.get('test')
    
    logger.info(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation samples: {len(val_dataset)}")
    if test_dataset:
        logger.info(f"Test samples: {len(test_dataset)}")
    
    logger.info("Starting fine-tuning...")
    model, tokenizer = finetune_model(model, tokenizer, train_dataset, config, val_dataset)
    
    logger.info(f"Fine-tuning complete! Model saved to {config['fine_tuning']['output_dir']}")
    test_prompts_str = config.get("test_prompts", "")
    if test_prompts_str:
        logger.info("Generating comparison report...")
        from report_generator import ReportGenerator
        
        test_prompts = [p.strip() for p in test_prompts_str.split(",") if p.strip()]
        config["test_prompts"] = test_prompts
        
        base_model, base_tokenizer = load_model_from_config(config)
        
        report_gen = ReportGenerator(config)
        report = report_gen.generate_comparison_report(
            base_model, base_tokenizer,
            model, tokenizer
        )
        logger.info(f"Comparison report saved to {config.get('report_path')}")


if __name__ == "__main__":
    main()
