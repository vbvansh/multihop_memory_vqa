import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from transformers import get_linear_schedule_with_warmup
from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder
from models.memory.memory_bank import MemoryBank
from models.reasoning.dual_stream import DualStreamProcessor
from models.memory.memory_router import MemoryRouter, aggregate_page_logits
from models.memory.gumbel_router import GumbelRouter
from models.reasoning.multi_hop import MultiHopReasoning
from models.decoder.answer_head import AnswerHead
from losses.sparsity_loss import SparsityLoss
from losses.routing_loss import RoutingLoss
from datasets.docvqa import DocVQADataset, collate_fn
from utils.logger import setup_logger
from utils.metrics import calculate_anls, calculate_exact_match

class ColPaliTrainer:
    """
    Stage-A training engine for the retrieve-then-read pipeline.
    Trains the sparse memory-routing retriever (dual-stream + policy router) to
    select the answer-bearing page, supervised by answer_page_idx. The PaliGemma
    reader (Stage B) and multi-hop routing variant slot in on top of this.
    Supports local precomputed embedding bypass to train at high speeds.
    """
    def __init__(self, model_config_path, train_config_path):
        # 1. Load Configurations
        with open(model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        with open(train_config_path, "r") as f:
            self.train_config = yaml.safe_load(f)

        # Merge model and train configs
        self.config = {**self.model_config, **self.train_config}

        # 2. Setup Logging and Checkpoint Directories
        self.logs_dir = self.config["paths"].get("logs_dir", "./logs")
        self.checkpoints_dir = self.config["paths"].get("checkpoints_dir", "./checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        self.logger = setup_logger(log_dir=self.logs_dir)
        self.logger.info("Initializing Sparse Routing ColPali Trainer...")

        # 3. Setup Hardware Device
        self.device = torch.device(self.config["model"]["device"] if torch.cuda.is_available() else "cpu")
        self.dtype_str = self.config["model"]["dtype"]
        self.dtype = torch.bfloat16 if self.dtype_str == "bfloat16" else (torch.float16 if self.dtype_str == "float16" else torch.float32)

        self.logger.info(f"Targeting device: {self.device} with data type: {self.dtype}")

        model_name = self.config["model"]["name"]

        # 4. Handle precomputed embeddings mode vs standard VLM mode
        self.use_precomputed = self.config["debug"].get("use_precomputed_embeddings", False)

        if self.use_precomputed:
            self.logger.info("Using PRECOMPUTED embeddings mode. Skipping backbone VLM loading to save VRAM and speed up training!")
            from transformers import ColPaliProcessor
            processor = ColPaliProcessor.from_pretrained(model_name)
            self.tokenizer = processor.tokenizer
            self.vision_encoder = None
            self.question_encoder = None
            self.shared_model = None
        else:
            # Initialize Shared Model Backbone (Avoid duplicate 3B VRAM loading)
            quantize = self.config["model"].get("quantize_4bit", False)
            from transformers import ColPaliForRetrieval

            if quantize and torch.cuda.is_available():
                self.logger.info(f"Loading shared ColPali backbone {model_name} in 4-bit quantization...")
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    llm_int8_skip_modules=["embedding_proj_layer"]
                )
                self.shared_model = ColPaliForRetrieval.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                    low_cpu_mem_usage=True
                )
            else:
                self.logger.info(f"Loading shared ColPali backbone from {model_name} in standard {self.dtype}...")
                self.shared_model = ColPaliForRetrieval.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map=self.config["model"]["device"],
                    low_cpu_mem_usage=True
                )

            # Freeze all parameters of the shared ColPali backbone.
            for param in self.shared_model.parameters():
                param.requires_grad = False

            # 5. Initialize Components
            self.vision_encoder = ColPaliVisionEncoder(
                model_name=model_name,
                device=self.device,
                dtype=self.dtype,
                shared_model=self.shared_model
            )
            self.question_encoder = ColPaliQuestionEncoder(
                model_name=model_name,
                device=self.device,
                dtype=self.dtype,
                shared_model=self.shared_model
            )

            # Pull tokenizer from the processor to get dynamic vocabulary size
            self.tokenizer = self.vision_encoder.processor.tokenizer

        vocab_size = self.tokenizer.vocab_size
        self.logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")

        # Instantiate sparse memory routing network components
        self.memory_bank = MemoryBank(self.config)
        self.dual_stream = DualStreamProcessor(self.config).to(self.device).to(self.dtype)
        self.router = MemoryRouter(self.config).to(self.device).to(self.dtype)
        self.gumbel = GumbelRouter(temperature=0.5).to(self.device)
        self.multi_hop = MultiHopReasoning(self.config).to(self.device).to(self.dtype)
        self.answer_head = AnswerHead(self.config, vocab_size=vocab_size).to(self.device).to(self.dtype)

        # Stage-A retriever loss: supervised page selection (answer_page_idx) + sparsity + entropy
        self.routing_loss_fn = RoutingLoss(self.config)
        self.slots_per_page = self.memory_bank.num_slots_per_page

        # 6. Set up Optimizer (Stage A trains only the retriever: dual_stream + router).
        # multi_hop / answer_head stay instantiated for the later multi-hop-routing variant
        # and the custom-head ablation, but are NOT optimized in the page-selection stage.
        base_lr = float(self.config["training"]["lr"])
        weight_decay = float(self.config["training"]["weight_decay"])

        self.trainable_modules = [self.dual_stream, self.router]
        optimizer_grouped_parameters = [
            {
                "params": [p for m in self.trainable_modules for p in m.parameters()],
                "lr": base_lr,
                "weight_decay": weight_decay
            }
        ]

        self.optimizer = optim.AdamW(optimizer_grouped_parameters)


    def get_dataloader(self, split="train", shuffle=None):
        """Loads either the debug dataset or the real dataset split."""
        debug_mode = self.config["debug"]["enable"]
        dataset = DocVQADataset(self.config, split=split, debug=debug_mode)

        # Shuffle train split by default, keep validation/test order deterministic
        if shuffle is None:
            shuffle = (split == "train")

        loader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"] if not debug_mode else 2,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=False
        )
        return loader

    def prepare_batch_tensors(self, batch):
        """Prepares batched, dynamically padded vision and question tensors across a batch."""
        batch_size = len(batch["questions"])
        precomputed_dir = self.config["debug"]["precomputed_dir"]

        page_embs_list = []
        query_embs_list = []
        page_counts = []

        for i in range(batch_size):
            question_id = batch["question_ids"][i]
            if self.use_precomputed:
                emb_path = os.path.join(precomputed_dir, f"vision_{question_id}.pt")
                page_emb = torch.load(emb_path, map_location=self.device).to(self.dtype)
                q_emb_path = os.path.join(precomputed_dir, f"question_{question_id}.pt")
                query_emb = torch.load(q_emb_path, map_location=self.device).to(self.dtype)
            else:
                images = batch["images"][i]
                page_embs = []
                with torch.no_grad():
                    for img in images:
                        emb = self.vision_encoder([img])
                        page_embs.append(emb.cpu())
                page_emb = torch.cat(page_embs, dim=0).to(self.device).to(self.dtype)
                question = batch["questions"][i]
                with torch.no_grad():
                    query_emb = self.question_encoder([question]).to(self.device).to(self.dtype)

            page_embs_list.append(page_emb)
            if query_emb.dim() == 3:
                query_emb = query_emb.squeeze(0)
            query_embs_list.append(query_emb)
            page_counts.append(page_emb.shape[0])

        max_pages = max(page_counts)
        num_patches = page_embs_list[0].shape[1]
        D = page_embs_list[0].shape[2]

        padded_page_embs = torch.zeros(batch_size, max_pages, num_patches, D, device=self.device, dtype=self.dtype)
        slot_padding_mask = torch.ones(batch_size, max_pages * self.slots_per_page, device=self.device, dtype=torch.bool)

        for i in range(batch_size):
            p_count = page_counts[i]
            padded_page_embs[i, :p_count] = page_embs_list[i]
            slot_padding_mask[i, :p_count * self.slots_per_page] = False

        q_lengths = [q.shape[0] for q in query_embs_list]
        max_q_len = max(q_lengths)
        padded_query_embs = torch.zeros(batch_size, max_q_len, D, device=self.device, dtype=self.dtype)
        for i in range(batch_size):
            padded_query_embs[i, :q_lengths[i]] = query_embs_list[i]

        return padded_page_embs, padded_query_embs, slot_padding_mask

    def _route(self, batch):
        """Retriever forward pass: returns routing tensors + aggregated per-page logits."""
        padded_page_embs, padded_query_embs, slot_padding_mask = self.prepare_batch_tensors(batch)

        memory_outputs = self.memory_bank(padded_page_embs)
        chunk_embeddings = memory_outputs["embeddings"].to(self.device).to(self.dtype)

        c_q, m_tilde, _ = self.dual_stream(
            padded_query_embs, chunk_embeddings, slot_padding_mask=slot_padding_mask
        )
        router_logits = self.router(c_q, m_tilde)
        z = self.gumbel(router_logits, hard=True, slot_padding_mask=slot_padding_mask)

        page_logits, page_padding_mask = aggregate_page_logits(
            router_logits, slot_padding_mask, self.slots_per_page
        )
        return router_logits, z, page_logits, page_padding_mask, slot_padding_mask

    def _page_targets(self, batch, num_pages):
        """Gold answer-page indices clamped to the batch's padded page count."""
        idx = torch.tensor(batch["answer_page_idxs"], device=self.device, dtype=torch.long)
        return idx.clamp(min=0, max=num_pages - 1)

    def train_epoch(self, loader, epoch):
        self.dual_stream.train()
        self.router.train()
        self.gumbel.train()

        totals = {"loss": 0.0, "page": 0.0, "sparsity": 0.0, "entropy": 0.0, "acc": 0.0, "active": 0.0}
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")

        for step, batch in enumerate(loop):
            self.optimizer.zero_grad()

            router_logits, z, page_logits, page_padding_mask, slot_padding_mask = self._route(batch)
            target = self._page_targets(batch, page_logits.shape[1])

            loss, details = self.routing_loss_fn(
                page_logits, page_padding_mask, target, router_logits, z, slot_padding_mask
            )

            loss.backward()

            trainable_params = [p for m in self.trainable_modules for p in m.parameters()]
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config["training"]["grad_clip"])
            self.optimizer.step()
            if getattr(self, "scheduler", None) is not None:
                self.scheduler.step()

            totals["loss"] += details["loss_total"]
            totals["page"] += details["loss_page"]
            totals["sparsity"] += details["loss_sparsity"]
            totals["entropy"] += details["loss_entropy"]
            totals["acc"] += details["page_acc"]
            totals["active"] += details["active_ratio"]

            loop.set_postfix(
                loss=details["loss_total"],
                page=details["loss_page"],
                acc=f"{details['page_acc']:.2%}",
                act=f"{details['active_ratio']:.2%}"
            )

            if step % 10 == 0:
                with torch.no_grad():
                    pred_page = torch.argmax(page_logits.masked_fill(page_padding_mask, -1e9), dim=1)
                    grad_dual = sum(p.grad.data.norm(2).item() ** 2 for p in self.dual_stream.parameters() if p.grad is not None) ** 0.5
                    grad_router = sum(p.grad.data.norm(2).item() ** 2 for p in self.router.parameters() if p.grad is not None) ** 0.5
                self.logger.info(
                    f"[Step {step}] page_acc={details['page_acc']:.2%} | active_ratio={details['active_ratio']:.2%} | "
                    f"loss_page={details['loss_page']:.3f} | GradNorms [dual/router]: {grad_dual:.2e}/{grad_router:.2e} | "
                    f"pred_pages={pred_page[:4].tolist()} | gt_pages={target[:4].tolist()}"
                )

        num_batches = max(len(loader), 1)
        avg = {k: v / num_batches for k, v in totals.items()}
        self.logger.info(
            f"Epoch {epoch+1} [Router Stage A] "
            f"loss_total: {avg['loss']:.4f} | loss_page: {avg['page']:.4f} | "
            f"loss_sparsity: {avg['sparsity']:.4f} | loss_entropy: {avg['entropy']:.4f} | "
            f"page_acc: {avg['acc']:.4f} ({avg['acc']:.2%}) | active_ratio: {avg['active']:.4f} ({avg['active']:.2%})"
        )
        return avg["loss"], avg["acc"]

    def save_checkpoint(self, epoch, val_page_acc, best_page_acc, is_best=False):
        """Saves retriever weights + optimizer state. Metric tracked is page-selection accuracy."""
        checkpoint = {
            "epoch": epoch,
            "best_page_acc": best_page_acc,
            "val_page_acc": val_page_acc,
            "dual_stream_state": self.dual_stream.state_dict(),
            "router_state": self.router.state_dict(),
            "multi_hop_state": self.multi_hop.state_dict(),
            "answer_head_state": self.answer_head.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "config": self.config
        }

        last_path = os.path.join(self.checkpoints_dir, "checkpoint_last.pt")
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = os.path.join(self.checkpoints_dir, "checkpoint_best.pt")
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved new best checkpoint to {best_path} with Page Acc: {val_page_acc:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Loads retriever weights from a saved checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        self.logger.info(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.dual_stream.load_state_dict(checkpoint["dual_stream_state"])
        self.router.load_state_dict(checkpoint["router_state"])
        if "multi_hop_state" in checkpoint:
            self.multi_hop.load_state_dict(checkpoint["multi_hop_state"])
        if "answer_head_state" in checkpoint:
            self.answer_head.load_state_dict(checkpoint["answer_head_state"])
        return checkpoint

    def evaluate(self, loader, name="Validation"):
        """Stage-A evaluation: measures answer-page selection accuracy (top-1)."""
        self.dual_stream.eval()
        self.router.eval()
        self.gumbel.eval()

        total_loss = 0.0
        total_correct = 0
        total_active = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in loader:
                batch_size = len(batch["questions"])
                router_logits, z, page_logits, page_padding_mask, slot_padding_mask = self._route(batch)
                target = self._page_targets(batch, page_logits.shape[1])

                loss, details = self.routing_loss_fn(
                    page_logits, page_padding_mask, target, router_logits, z, slot_padding_mask
                )

                pred_page = torch.argmax(page_logits.masked_fill(page_padding_mask, -1e9), dim=1)
                total_loss += details["loss_total"] * batch_size
                total_correct += (pred_page == target).sum().item()
                total_active += details["active_ratio"] * batch_size
                total_samples += batch_size

                for i in range(min(batch_size, 3)):
                    self.logger.info(
                        f"[{name}] Q: '{batch['questions'][i][:60]}' | "
                        f"pred_page={pred_page[i].item()} | gt_page={target[i].item()}"
                    )

        avg_loss = total_loss / total_samples if total_samples else 0.0
        page_acc = total_correct / total_samples if total_samples else 0.0
        avg_active = total_active / total_samples if total_samples else 0.0

        self.logger.info(
            f"[{name}] Router Eval Complete. Loss: {avg_loss:.4f} | "
            f"Page Acc: {page_acc:.4f} ({page_acc:.2%}) | Active Ratio: {avg_active:.4f}"
        )
        return avg_loss, page_acc

    def run(self):
        """Stage-A training loop: trains the retriever and checkpoints on page-selection accuracy."""
        self.logger.info("Loading dataset loaders...")
        debug_mode = self.config["debug"]["enable"]

        run_val = self.config["training"].get("run_val", True)
        run_test = self.config["training"].get("run_test", True)

        train_loader = self.get_dataloader(split="train")
        val_loader = self.get_dataloader(split="val") if (run_val and not debug_mode) else None

        self.logger.info("Starting Stage-A (retriever / page-selection) training loop...")
        epochs = self.config["training"]["epochs"]

        num_training_steps = epochs * max(len(train_loader), 1)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=50,
            num_training_steps=num_training_steps
        )

        best_page_acc = -1.0

        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.logger.info(f"Starting Epoch {epoch+1}/{epochs} with Learning Rate: {current_lr:.2e}")
            train_loss, train_acc = self.train_epoch(train_loader, epoch)

            if debug_mode:
                self.logger.info("Evaluating page selection on debug samples:")
                val_loss, val_acc = self.evaluate(train_loader, name="Val-Debug")
            elif run_val:
                self.logger.info("Evaluating validation page selection:")
                val_loss, val_acc = self.evaluate(val_loader, name="Val")
            else:
                self.logger.info("Evaluating training page selection:")
                val_loss, val_acc = self.evaluate(train_loader, name="Train")

            is_best = val_acc > best_page_acc
            if is_best:
                best_page_acc = val_acc

            self.save_checkpoint(epoch + 1, val_acc, best_page_acc, is_best=is_best)

            self.logger.info(
                f"[Summary] Epoch {epoch+1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Page Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} | Val Page Acc: {val_acc:.4f} | "
                f"Best Page Acc: {best_page_acc:.4f}"
            )

            # Gumbel temperature annealing (decay 10% per epoch, floor at 0.1)
            new_temp = max(0.1, self.gumbel.temperature * 0.9)
            self.gumbel.set_temperature(new_temp)
            self.logger.info(f"Annealed Gumbel router temperature to: {new_temp:.4f}")

        if run_test and not debug_mode:
            self.logger.info("Starting final page-selection evaluation on test split...")
            best_path = os.path.join(self.checkpoints_dir, "checkpoint_best.pt")
            if os.path.exists(best_path):
                self.load_checkpoint(best_path)
            else:
                self.logger.warning("checkpoint_best.pt not found. Evaluating test split on last epoch weights.")

            test_loader = self.get_dataloader(split="test")
            test_loss, test_acc = self.evaluate(test_loader, name="Test")
            self.logger.info(
                f"[Final Test Metrics] Test Loss: {test_loss:.4f} | Test Page Acc: {test_acc:.4f} ({test_acc:.2%})"
            )
