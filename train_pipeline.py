import os
import sys
import argparse

# Inject project root directory into sys.path to guarantee clean package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainers.trainer import ColPaliTrainer
from trainers.reader_trainer import ReaderTrainer


def main():
    """
    Single-command, two-stage training for the retrieve-then-read pipeline.

        Stage A: train the sparse memory-routing retriever (page selection).
        Stage B: train the PaliGemma reader; the just-trained router is reused
                 for the end-to-end ANLS/EM evaluation (no reload, no extra command).

    The two stages stay separate internally (the page hand-off to the image reader
    is non-differentiable, so they cannot share one loss) — but you run ONE command.
    """
    parser = argparse.ArgumentParser(description="Retrieve-then-Read - Full Pipeline (Stage A + Stage B)")
    parser.add_argument("--model_config", type=str, default="./configs/model.yaml", help="Path to model config file")
    parser.add_argument("--train_config", type=str, default="./configs/train.yaml", help="Path to training config file")
    parser.add_argument("--skip_router", action="store_true",
                        help="Skip Stage A and reuse an existing router checkpoint_best.pt")
    parser.add_argument("--router_checkpoint", type=str, default=None,
                        help="Router checkpoint to use for Stage B end-to-end eval (defaults to checkpoints/checkpoint_best.pt)")
    parser.add_argument("--skip_zeroshot", action="store_true",
                        help="Skip the pre-training zero-shot reader eval (saves ~28 min on retries)")
    args = parser.parse_args()

    # ---------------- Stage A: retriever / page selection ----------------
    router_trainer = ColPaliTrainer(args.model_config, args.train_config)
    best_router = args.router_checkpoint or os.path.join(router_trainer.checkpoints_dir, "checkpoint_best.pt")

    if args.skip_router:
        router_trainer.logger.info("=" * 70)
        router_trainer.logger.info("Skipping Stage A (router). Reusing existing router checkpoint.")
        router_trainer.logger.info("=" * 70)
    else:
        router_trainer.logger.info("=" * 70)
        router_trainer.logger.info("STAGE A: Training the sparse memory-routing retriever (page selection)")
        router_trainer.logger.info("=" * 70)
        router_trainer.run()

    # Load the best router weights so Stage B's end-to-end eval uses the best page selector
    if os.path.exists(best_router):
        router_trainer.load_checkpoint(best_router)
    else:
        router_trainer.logger.warning(
            f"Router checkpoint '{best_router}' not found; end-to-end eval will be skipped in Stage B."
        )
        router_trainer = None

    # ---------------- Stage B: PaliGemma reader ----------------
    reader_trainer = ReaderTrainer(args.model_config, args.train_config)
    reader_trainer.logger.info("=" * 70)
    reader_trainer.logger.info("STAGE B: Training the PaliGemma reader (answer generation)")
    reader_trainer.logger.info("=" * 70)

    # Reuse the in-memory router (no reload) for end-to-end ANLS/EM
    if router_trainer is not None:
        reader_trainer.router_trainer = router_trainer

    if args.skip_zeroshot:
        reader_trainer.rcfg["skip_zeroshot"] = True

    reader_trainer.run()


if __name__ == "__main__":
    main()
