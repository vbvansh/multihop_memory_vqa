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
    parser.add_argument(
        "--split", 
        type=str, 
        default="val", 
        choices=["train", "val", "test"], 
        help="Dataset split to evaluate"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="./checkpoints/checkpoint_best.pt", 
        help="Path to checkpoint file"
    )
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ColPaliTrainer(
        model_config_path=args.model_config,
        train_config_path=args.train_config
    )
    
    # Load checkpoint if exists
    checkpoint_path = args.checkpoint
    if checkpoint_path == "./checkpoints/checkpoint_best.pt":
        checkpoint_path = os.path.join(trainer.checkpoints_dir, "checkpoint_best.pt")
        
    if os.path.exists(checkpoint_path):
        trainer.load_checkpoint(checkpoint_path)
    else:
        trainer.logger.warning(f"No checkpoint found at {checkpoint_path}. Evaluating with default/random weights!")
    
    # Load dataloader for specified split
    loader = trainer.get_dataloader(split=args.split)
    
    # Run evaluation
    trainer.logger.info(f"Starting evaluation on split '{args.split}'...")
    trainer.evaluate(loader)

if __name__ == "__main__":
    main()
