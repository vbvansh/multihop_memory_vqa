import os
import sys
import argparse

# Inject project root directory into sys.path to guarantee clean package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainers.trainer import ColPaliTrainer

def main():
    parser = argparse.ArgumentParser(description="ColPali Multi-Doc VQA - Evaluation Entrypoint")
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="./configs/model.yaml", 
        help="Path to model config file"
    )
    parser.add_argument(
        "--train_config", 
        type=str, 
        default="./configs/train.yaml", 
        help="Path to training config file"
    )
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ColPaliTrainer(
        model_config_path=args.model_config,
        train_config_path=args.train_config
    )
    
    # Load validation dataloader (is_train=False)
    loader = trainer.get_dataloader(is_train=False)
    
    # Run evaluation
    trainer.logger.info("Starting validation evaluation...")
    trainer.evaluate(loader)

if __name__ == "__main__":
    main()
