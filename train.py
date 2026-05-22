import argparse
from trainers.trainer import ColPaliTrainer

def main():
    parser = argparse.ArgumentParser(description="ColPali Multi-Doc VQA - Training Entrypoint")
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
    
    # Start training and overfit evaluation
    trainer.run()

if __name__ == "__main__":
    main()
