import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainers.reranker_trainer import RerankerTrainer


def main():
    parser = argparse.ArgumentParser(description="MaxSim-anchored multi-hop page reranker (beat MaxSim's page acc)")
    parser.add_argument("--model_config", type=str, default="./configs/model.yaml")
    parser.add_argument("--train_config", type=str, default="./configs/train.yaml")
    args = parser.parse_args()

    trainer = RerankerTrainer(args.model_config, args.train_config)
    trainer.run()


if __name__ == "__main__":
    main()
