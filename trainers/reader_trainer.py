import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from transformers import get_linear_schedule_with_warmup
from models.reader.paligemma_reader import PaliGemmaReader
from datasets.docvqa import DocVQADataset, collate_fn
from utils.logger import setup_logger
from utils.metrics import calculate_anls, calculate_exact_match


class ReaderTrainer:
    """
    Stage-B trainer for the retrieve-then-read READER.

    Trains PaliGemma (QLoRA) on the GOLD page (answer_page_idx) with teacher
    forcing, which decouples reader quality from retriever errors. Reports:
      - ORACLE-page ANLS/EM: reader on the gold page  = the reader ceiling.
      - END-TO-END ANLS/EM (optional): router picks the page, reader reads it.
        Requires a trained router checkpoint (reader_training.router_checkpoint).
    """

    def __init__(self, model_config_path, train_config_path):
        with open(model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        with open(train_config_path, "r") as f:
            self.train_config = yaml.safe_load(f)
        self.config = {**self.model_config, **self.train_config}
        self.model_config_path = model_config_path
        self.train_config_path = train_config_path

        self.rcfg = self.config.get("reader_training", {}) or {}
        self.logs_dir = self.config["paths"].get("logs_dir", "./logs")
        self.checkpoints_dir = self.config["paths"].get("checkpoints_dir", "./checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.logger = setup_logger(log_dir=self.logs_dir, name="ReaderTrainer")
        self.logger.info("Initializing PaliGemma Reader (Stage B)...")

        self.reader = PaliGemmaReader(self.config)

        base_lr = float(self.rcfg.get("lr", 2e-4))
        weight_decay = float(self.rcfg.get("weight_decay", 0.0))
        self.optimizer = optim.AdamW(
            self.reader.trainable_parameters(), lr=base_lr, weight_decay=weight_decay
        )
        self.grad_accum = int(self.rcfg.get("grad_accum", 8))
        self.grad_clip = float(self.rcfg.get("grad_clip", 1.0))

        # Optional router for end-to-end evaluation (lazy: only if a checkpoint is given)
        self.router_trainer = None
        router_ckpt = self.rcfg.get("router_checkpoint", "") or ""
        if router_ckpt and os.path.exists(router_ckpt):
            self.logger.info(f"Loading router checkpoint for end-to-end eval: {router_ckpt}")
            from trainers.trainer import ColPaliTrainer
            self.router_trainer = ColPaliTrainer(self.model_config_path, self.train_config_path)
            self.router_trainer.load_checkpoint(router_ckpt)
        elif router_ckpt:
            self.logger.warning(f"router_checkpoint '{router_ckpt}' not found. Skipping end-to-end eval.")

    # ---------------- data ----------------

    def get_dataloader(self, split="train", shuffle=None):
        debug_mode = self.config["debug"]["enable"]
        dataset = DocVQADataset(self.config, split=split, debug=debug_mode)
        if shuffle is None:
            shuffle = (split == "train")
        return DataLoader(
            dataset,
            batch_size=int(self.rcfg.get("batch_size", 2)) if not debug_mode else 2,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=False,
        )

    @staticmethod
    def _select_pages(batch, page_indices):
        """Picks one PIL page per sample given a list of page indices (clamped)."""
        images = []
        for imgs, idx in zip(batch["images"], page_indices):
            if not imgs:
                continue
            j = max(0, min(int(idx), len(imgs) - 1))
            images.append(imgs[j])
        return images

    def _gold_pages(self, batch):
        return self._select_pages(batch, batch["answer_page_idxs"])

    # ---------------- train ----------------

    def train_epoch(self, loader, epoch):
        self.reader.model.train()
        epoch_loss = 0.0
        steps = 0
        self.optimizer.zero_grad()
        loop = tqdm(loader, desc=f"[Reader] Epoch {epoch+1}/{self.rcfg.get('epochs', 3)}")

        for step, batch in enumerate(loop):
            images = self._gold_pages(batch)
            loss = self.reader.compute_loss(images, batch["questions"], batch["answers"])
            (loss / self.grad_accum).backward()

            if (step + 1) % self.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(self.reader.trainable_parameters(), self.grad_clip)
                self.optimizer.step()
                if getattr(self, "scheduler", None) is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad()

            epoch_loss += loss.item()
            steps += 1
            loop.set_postfix(loss=loss.item())

            if step % 20 == 0:
                self.logger.info(f"[Reader Step {step}] lm_loss={loss.item():.4f}")

        avg = epoch_loss / max(steps, 1)
        self.logger.info(f"[Reader] Epoch {epoch+1} complete. Avg LM loss: {avg:.4f}")
        return avg

    # ---------------- eval ----------------

    def _score(self, predictions, ground_truths, name):
        total_anls, total_em = 0.0, 0.0
        for pred, gts in zip(predictions, ground_truths):
            total_anls += calculate_anls(pred, gts)
            total_em += calculate_exact_match(pred, gts)
        n = max(len(predictions), 1)
        anls, em = total_anls / n, total_em / n
        self.logger.info(f"[{name}] ANLS: {anls:.4f} | EM: {em:.4f} | N={len(predictions)}")
        return anls, em

    @torch.no_grad()
    def evaluate_oracle(self, loader, name="Oracle"):
        """Reader on the GOLD page = reader ceiling."""
        self.reader.model.eval()
        predictions, ground_truths = [], []
        for batch in tqdm(loader, desc=f"[Reader Eval:{name}]"):
            images = self._gold_pages(batch)
            preds = self.reader.generate(images, batch["questions"])
            predictions.extend(preds)
            ground_truths.extend(batch["answers_all"])
        for p, g in zip(predictions[:5], ground_truths[:5]):
            self.logger.info(f"[{name}] Pred: '{p}' | GT: {g}")
        return self._score(predictions, ground_truths, name)

    @torch.no_grad()
    def evaluate_end_to_end(self, loader, name="End2End"):
        """Router selects the page, reader reads it. Requires a router checkpoint."""
        if self.router_trainer is None:
            self.logger.info("No router loaded; skipping end-to-end eval.")
            return None, None
        self.reader.model.eval()
        self.router_trainer.dual_stream.eval()
        self.router_trainer.router.eval()
        self.router_trainer.gumbel.eval()

        predictions, ground_truths = [], []
        for batch in tqdm(loader, desc=f"[Reader Eval:{name}]"):
            _, _, page_logits, page_padding_mask, _ = self.router_trainer._route(batch)
            pred_page = torch.argmax(page_logits.masked_fill(page_padding_mask, -1e9), dim=1).tolist()
            images = self._select_pages(batch, pred_page)
            preds = self.reader.generate(images, batch["questions"])
            predictions.extend(preds)
            ground_truths.extend(batch["answers_all"])
        for p, g in zip(predictions[:5], ground_truths[:5]):
            self.logger.info(f"[{name}] Pred: '{p}' | GT: {g}")
        return self._score(predictions, ground_truths, name)

    # ---------------- loop ----------------

    def save_lora(self, tag="best"):
        path = os.path.join(self.checkpoints_dir, f"reader_lora_{tag}")
        self.reader.save_lora(path)
        self.logger.info(f"Saved reader LoRA adapters to {path}")

    def run(self):
        debug_mode = self.config["debug"]["enable"]
        run_val = self.config["training"].get("run_val", True)
        epochs = int(self.rcfg.get("epochs", 3))

        train_loader = self.get_dataloader("train", shuffle=True)
        eval_loader = self.get_dataloader("val", shuffle=False) if (run_val and not debug_mode) else train_loader
        eval_name = "Val" if (run_val and not debug_mode) else "Train"

        num_training_steps = epochs * max(len(train_loader) // self.grad_accum, 1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=int(self.rcfg.get("warmup_steps", 20)),
            num_training_steps=num_training_steps,
        )

        # Zero-shot baseline (mix model is VQA-tuned) before any training
        self.logger.info("=== Zero-shot reader baseline (before LoRA training) ===")
        self.evaluate_oracle(eval_loader, name=f"{eval_name}-Oracle-ZeroShot")
        self.evaluate_end_to_end(eval_loader, name=f"{eval_name}-End2End-ZeroShot")

        best_anls = -1.0
        for epoch in range(epochs):
            self.train_epoch(train_loader, epoch)
            oracle_anls, oracle_em = self.evaluate_oracle(eval_loader, name=f"{eval_name}-Oracle")
            e2e_anls, e2e_em = self.evaluate_end_to_end(eval_loader, name=f"{eval_name}-End2End")

            self.save_lora(tag="last")
            if oracle_anls > best_anls:
                best_anls = oracle_anls
                self.save_lora(tag="best")

            e2e_str = f"{e2e_anls:.4f}" if e2e_anls is not None else "n/a"
            self.logger.info(
                f"[Summary] Reader Epoch {epoch+1}/{epochs} | "
                f"Oracle ANLS: {oracle_anls:.4f} (EM {oracle_em:.4f}) | "
                f"End2End ANLS: {e2e_str} | Best Oracle ANLS: {best_anls:.4f}"
            )
