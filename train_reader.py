import os
import sys
import argparse

# Inject project root directory into sys.path to guarantee clean package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainers.reader_trainer import ReaderTrainer


def main():
    parser = argparse.ArgumentParser(description="PaliGemma Reader (Stage B) - Training Entrypoint")
    parser.add_argument("--model_config", type=str, default="./configs/model.yaml", help="Path to model config file")
    parser.add_argument("--train_config", type=str, default="./configs/train.yaml", help="Path to training config file")
    parser.add_argument("--router_checkpoint", type=str, default=None,
                        help="Optional router checkpoint for end-to-end eval (overrides reader_training.router_checkpoint)")
    parser.add_argument("--skip_zeroshot", action="store_true",
                        help="Skip the pre-training zero-shot reader eval (saves ~28 min on retries)")
    args = parser.parse_args()

    trainer = ReaderTrainer(
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )
    if args.skip_zeroshot:
        trainer.rcfg["skip_zeroshot"] = True

    if args.router_checkpoint:
        # CLI override for the end-to-end router checkpoint
        trainer.rcfg["router_checkpoint"] = args.router_checkpoint
        if trainer.router_trainer is None and os.path.exists(args.router_checkpoint):
            from trainers.trainer import ColPaliTrainer
            trainer.logger.info(f"Loading router checkpoint (CLI): {args.router_checkpoint}")
            trainer.router_trainer = ColPaliTrainer(args.model_config, args.train_config)
            trainer.router_trainer.load_checkpoint(args.router_checkpoint)

    trainer.run()


if __name__ == "__main__":
    main()
