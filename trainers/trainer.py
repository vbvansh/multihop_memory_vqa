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
from models.memory.memory_router import MemoryRouter
from models.memory.gumbel_router import GumbelRouter
from models.reasoning.multi_hop import MultiHopReasoning
from models.decoder.answer_head import AnswerHead
from losses.sparsity_loss import SparsityLoss
from datasets.docvqa import DocVQADataset, collate_fn
from utils.logger import setup_logger
from utils.metrics import calculate_anls, calculate_exact_match

class ColPaliTrainer:
    """
    Unified training engine supporting sparse memory routing,
    multi-hop reasoning, and gated answer generation.
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
        
        # Sparsity & Entropy Regularization Loss
        self.sparsity_loss_fn = SparsityLoss(self.config, ignore_index=self.tokenizer.pad_token_id)
        
        # 6. Set up Optimizer (All parameter groups at base_lr = 1e-4)
        base_lr = float(self.config["training"]["lr"])
        weight_decay = float(self.config["training"]["weight_decay"])
        
        optimizer_grouped_parameters = [
            {
                "params": list(self.dual_stream.parameters()) + list(self.router.parameters()) + list(self.multi_hop.parameters()),
                "lr": base_lr,
                "weight_decay": weight_decay
            },
            {
                "params": list(self.answer_head.parameters()),
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
        slot_padding_mask = torch.ones(batch_size, max_pages * 4, device=self.device, dtype=torch.bool)
        
        for i in range(batch_size):
            p_count = page_counts[i]
            padded_page_embs[i, :p_count] = page_embs_list[i]
            slot_padding_mask[i, :p_count * 4] = False
            
        q_lengths = [q.shape[0] for q in query_embs_list]
        max_q_len = max(q_lengths)
        padded_query_embs = torch.zeros(batch_size, max_q_len, D, device=self.device, dtype=self.dtype)
        for i in range(batch_size):
            padded_query_embs[i, :q_lengths[i]] = query_embs_list[i]
            
        return padded_page_embs, padded_query_embs, slot_padding_mask

    def train_epoch(self, loader, epoch):
        self.dual_stream.train()
        self.router.train()
        self.multi_hop.train()
        self.answer_head.train()
        if self.shared_model is not None:
            self.shared_model.eval()
        
        epoch_loss = 0.0
        epoch_vqa_loss = 0.0
        epoch_sp_loss = 0.0
        epoch_ent_loss = 0.0
        epoch_active_ratio = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        for step, batch in enumerate(loop):
            self.optimizer.zero_grad()
            batch_size = len(batch["questions"])
            
            # Prepare batched tensors with dynamic padding
            padded_page_embs, padded_query_embs, slot_padding_mask = self.prepare_batch_tensors(batch)
            
            # Partition into visual memory slots
            memory_outputs = self.memory_bank(padded_page_embs)
            chunk_embeddings = memory_outputs["embeddings"].to(self.device).to(self.dtype)
            
            # Decoupled dual-stream processing
            c_q, m_tilde, contextualized_patches = self.dual_stream(padded_query_embs, chunk_embeddings)
            
            # Dynamic routing & gating
            router_logits = self.router(c_q, m_tilde)
            z = self.gumbel(router_logits, hard=True, slot_padding_mask=slot_padding_mask)
            
            # Multi-Hop Reasoning over contextualized patches
            updated_query, key_padding_mask = self.multi_hop(padded_query_embs, contextualized_patches, z, slot_padding_mask=slot_padding_mask)
            
            # Gated Answer prediction
            flat_memory = contextualized_patches.reshape(batch_size, -1, self.config["model"]["embedding_dim"])
            pred_logits = self.answer_head(updated_query, flat_memory, key_padding_mask=key_padding_mask)
            
            # Tokenize target labels in parallel
            targets = self.tokenizer(
                batch["answers"],
                padding="max_length",
                max_length=self.config["decoder"]["max_answer_len"],
                truncation=True,
                return_tensors="pt"
            )["input_ids"].to(self.device)
            
            flat_logits = pred_logits.view(-1, pred_logits.size(-1))
            flat_targets = targets.view(-1)
            
            loss, loss_details = self.sparsity_loss_fn(
                flat_logits,
                flat_targets,
                router_logits,
                z
            )
            
            if epoch == 0 and step == 0:
                self.logger.info("--- QUESTION 3: Differential LR Check ---")
                for i, group in enumerate(self.optimizer.param_groups):
                    self.logger.info(f"Group {i}: lr={group['lr']:.2e}, num_params={len(group['params'])}")
                    
                self.logger.info("--- QUESTION 5 & QUESTION 2: Loss Input Shapes & Target Samples ---")
                self.logger.info(f"Question 5 - logits.shape: {flat_logits.shape}, targets.shape: {flat_targets.shape}")
                self.logger.info(f"Question 2 - flat_logits sample (first 2 tokens, first 5 logits):\n{flat_logits[:2, :5].tolist()}")
                self.logger.info(f"Question 2 - flat_targets sample (first 10 target token IDs):\n{flat_targets[:10].tolist()}")
                
                self.logger.info("--- QUESTION 4: Backward Call Check ---")
                print("backward called")
                self.logger.info("backward called")
            
            loss.backward()
            
            if epoch == 0 and step == 0:
                self.logger.info("--- QUESTION 1: AnswerHead Parameter Gradients ---")
                for name, param in self.answer_head.named_parameters():
                    if param.grad is not None:
                        self.logger.info(f"[Grad Check] {name}: grad mean abs = {param.grad.abs().mean().item():.6e}")
                    else:
                        self.logger.info(f"[Grad Check] {name}: NO GRADIENT")
            
            # Step parameters
            trainable_params = (
                list(self.dual_stream.parameters()) + 
                list(self.router.parameters()) + 
                list(self.multi_hop.parameters()) + 
                list(self.answer_head.parameters())
            )
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config["training"]["grad_clip"])
            self.optimizer.step()
            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()
            
            epoch_loss += loss.item()
            epoch_vqa_loss += loss_details["loss_vqa"]
            epoch_sp_loss += loss_details["loss_sparsity"]
            epoch_ent_loss += loss_details["loss_entropy"]
            epoch_active_ratio += loss_details["active_ratio"]
            
            loop.set_postfix(
                loss=loss.item(),
                vqa=loss_details["loss_vqa"],
                sp=loss_details["loss_sparsity"],
                ent=loss_details["loss_entropy"],
                act=f"{loss_details['active_ratio']:.2%}"
            )
            
            # Log detailed activation and gradient diagnostics every 10 steps
            if step % 10 == 0:
                with torch.no_grad():
                    grad_dual = sum(p.grad.data.norm(2).item() ** 2 for p in self.dual_stream.parameters() if p.grad is not None) ** 0.5
                    grad_router = sum(p.grad.data.norm(2).item() ** 2 for p in self.router.parameters() if p.grad is not None) ** 0.5
                    grad_mhop = sum(p.grad.data.norm(2).item() ** 2 for p in self.multi_hop.parameters() if p.grad is not None) ** 0.5
                    grad_head = sum(p.grad.data.norm(2).item() ** 2 for p in self.answer_head.parameters() if p.grad is not None) ** 0.5
                    
                    has_nan_dual = torch.isnan(c_q).any().item() or torch.isnan(m_tilde).any().item()
                    
                    pred_ids = torch.argmax(pred_logits, dim=-1)
                    decoded_pred = self.tokenizer.decode(pred_ids[0].tolist(), skip_special_tokens=True).strip()
                    
                    self.logger.info(
                        f"[Step {step}] "
                        f"c_q norm: {c_q.norm().item():.3f} | m_tilde norm: {m_tilde.norm().item():.3f} | "
                        f"logits mean/min/max: {router_logits.mean().item():.3f}/{router_logits.min().item():.3f}/{router_logits.max().item():.3f} | "
                        f"z active: {z.sum().item()}/{z.numel()} ({z.mean().item():.2%}) | "
                        f"updated_query norm: {updated_query.norm().item():.3f} | "
                        f"GradNorms [dual/router/mhop/head]: {grad_dual:.2e}/{grad_router:.2e}/{grad_mhop:.2e}/{grad_head:.2e} | "
                        f"NaNs: {has_nan_dual} | "
                        f"Pred: '{decoded_pred}' | GT: '{batch['answers'][0]}'"
                    )
            
        num_batches = len(loader)
        avg_loss = epoch_loss / num_batches
        avg_vqa = epoch_vqa_loss / num_batches
        avg_sp = epoch_sp_loss / num_batches
        avg_ent = epoch_ent_loss / num_batches
        avg_active = epoch_active_ratio / num_batches
        
        self.logger.info(
            f"Epoch {epoch+1} Complete. Loss Breakdown -> "
            f"loss_total: {avg_loss:.4f} | loss_vqa: {avg_vqa:.4f} | "
            f"loss_sparsity: {avg_sp:.4f} | loss_entropy: {avg_ent:.4f} | active_ratio: {avg_active:.4f} ({avg_active:.2%})"
        )
        return avg_loss
        
    def save_checkpoint(self, epoch, val_anls, best_anls, is_best=False):
        """Saves a checkpoint containing only the trainable parameters and optimizer states."""
        checkpoint = {
            "epoch": epoch,
            "best_anls": best_anls,
            "val_anls": val_anls,
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
            self.logger.info(f"Saved new best checkpoint to {best_path} with ANLS: {val_anls:.4f}")

    def load_checkpoint(self, checkpoint_path):
        """Loads weights from a saved checkpoint into the trainable layers."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        self.logger.info(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.dual_stream.load_state_dict(checkpoint["dual_stream_state"])
        self.router.load_state_dict(checkpoint["router_state"])
        self.multi_hop.load_state_dict(checkpoint["multi_hop_state"])
        self.answer_head.load_state_dict(checkpoint["answer_head_state"])
        return checkpoint

    def evaluate(self, loader, name="Validation"):
        """Runs validation and decodes predicted answer tokens back to text strings."""
        self.dual_stream.eval()
        self.router.eval()
        self.multi_hop.eval()
        self.answer_head.eval()
        
        predictions = []
        ground_truths = []
        total_eval_loss = 0.0
        sample_count = 0
        
        with torch.no_grad():
            for batch in loader:
                batch_size = len(batch["questions"])
                sample_count += batch_size
                
                padded_page_embs, padded_query_embs, slot_padding_mask = self.prepare_batch_tensors(batch)
                
                memory_outputs = self.memory_bank(padded_page_embs)
                chunk_embeddings = memory_outputs["embeddings"].to(self.device).to(self.dtype)
                
                c_q, m_tilde, contextualized_patches = self.dual_stream(padded_query_embs, chunk_embeddings)
                router_logits = self.router(c_q, m_tilde)
                z = self.gumbel(router_logits, hard=True, slot_padding_mask=slot_padding_mask)
                
                updated_query, key_padding_mask = self.multi_hop(padded_query_embs, contextualized_patches, z, slot_padding_mask=slot_padding_mask)
                flat_memory = contextualized_patches.reshape(batch_size, -1, self.config["model"]["embedding_dim"])
                pred_logits = self.answer_head(updated_query, flat_memory, key_padding_mask=key_padding_mask)
                
                targets = self.tokenizer(
                    batch["answers"],
                    padding="max_length",
                    max_length=self.config["decoder"]["max_answer_len"],
                    truncation=True,
                    return_tensors="pt"
                )["input_ids"].to(self.device)
                
                flat_logits = pred_logits.view(-1, pred_logits.size(-1))
                flat_targets = targets.view(-1)
                
                loss, _ = self.sparsity_loss_fn(
                    flat_logits,
                    flat_targets,
                    router_logits,
                    z
                )
                total_eval_loss += loss.item() * batch_size
                
                pred_ids = torch.argmax(pred_logits, dim=-1)
                
                eos_id = tokenizer_eos_id = self.tokenizer.eos_token_id
                pad_id = self.tokenizer.pad_token_id
                
                for i in range(batch_size):
                    seq_list = pred_ids[i].tolist()
                    indices = [idx_t for idx_t, token in enumerate(seq_list) if token in (eos_id, pad_id)]
                    if indices:
                        first_idx = indices[0]
                        truncated_seq = seq_list[:first_idx]
                    else:
                        truncated_seq = seq_list
                    
                    decoded_str = self.tokenizer.decode(truncated_seq, skip_special_tokens=True).strip()
                    predictions.append(decoded_str)
                    ground_truths.append([batch["answers"][i]])
                    
        total_anls = 0.0
        total_em = 0.0
        
        for pred, gts in zip(predictions, ground_truths):
            total_anls += calculate_anls(pred, gts)
            total_em += calculate_exact_match(pred, gts)
            self.logger.info(f"[{name}] Pred: '{pred}' | Ground Truth: '{gts[0]}'")
            
        avg_loss = total_eval_loss / sample_count if sample_count else 0.0
        avg_anls = total_anls / len(predictions) if predictions else 0.0
        avg_em = total_em / len(predictions) if predictions else 0.0
        
        self.logger.info(f"[{name}] Evaluation Complete. Loss: {avg_loss:.4f} | ANLS: {avg_anls:.4f} | Exact Match: {avg_em:.4f}")
        return avg_loss, avg_anls, avg_em
        
    def run(self):
        """Runs the main training and evaluation loop with validation and checkpoint saving."""
        self.logger.info("Loading dataset loaders...")
        debug_mode = self.config["debug"]["enable"]
        
        # Load configs for validation and test splits
        run_val = self.config["training"].get("run_val", True)
        run_test = self.config["training"].get("run_test", True)
        
        # Load train and validation splits
        train_loader = self.get_dataloader(split="train")
        train_eval_loader = self.get_dataloader(split="train", shuffle=False) if not debug_mode else train_loader
        val_loader = self.get_dataloader(split="val") if (run_val and not debug_mode) else None
        
        self.logger.info("Starting training loop...")
        epochs = self.config["training"]["epochs"]
        
        total_batches = len(train_loader)
        num_training_steps = epochs * total_batches
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=50,
            num_training_steps=num_training_steps
        )
        
        best_anls = -1.0
        
        for epoch in range(epochs):
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Starting Epoch {epoch+1}/{epochs} with Learning Rate: {current_lr:.2e}")
            train_loss = self.train_epoch(train_loader, epoch)
            
            if debug_mode:
                self.logger.info("Evaluating overfit performance on debug samples:")
                val_loss, val_anls, val_em = self.evaluate(train_loader, name="Val-Debug")
            else:
                self.logger.info("Evaluating training split performance:")
                train_loss_eval, train_anls, train_em = self.evaluate(train_eval_loader, name="Train")
                
                if run_val:
                    self.logger.info("Evaluating validation performance:")
                    val_loss, val_anls, val_em = self.evaluate(val_loader, name="Val")
                
            # Check for best ANLS performance (use train_anls if validation is disabled)
            eval_metric_for_checkpoint = val_anls if (debug_mode or run_val) else train_anls
            is_best = eval_metric_for_checkpoint > best_anls
            if is_best:
                best_anls = eval_metric_for_checkpoint
                
            # Save checkpoints
            self.save_checkpoint(epoch + 1, eval_metric_for_checkpoint, best_anls, is_best=is_best)
            
            if debug_mode:
                self.logger.info(
                    f"[Summary] Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val ANLS: {val_anls:.4f} | "
                    f"Val EM: {val_em:.4f} | "
                    f"Best ANLS: {best_anls:.4f}"
                )
            else:
                if run_val:
                    self.logger.info(
                        f"[Summary] Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train Eval Loss: {train_loss_eval:.4f} | "
                        f"Train ANLS: {train_anls:.4f} | "
                        f"Train EM: {train_em:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Val ANLS: {val_anls:.4f} | "
                        f"Val EM: {val_em:.4f} | "
                        f"Best ANLS: {best_anls:.4f}"
                    )
                else:
                    self.logger.info(
                        f"[Summary] Epoch {epoch+1}/{epochs} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Train Eval Loss: {train_loss_eval:.4f} | "
                        f"Train ANLS: {train_anls:.4f} | "
                        f"Train EM: {train_em:.4f} | "
                        f"Best Train ANLS: {best_anls:.4f}"
                    )
            
        if run_test:
            # Post-Training: Load the best checkpoint weights before running final evaluation on the test split
            self.logger.info("Starting final evaluation on test split...")
            best_path = os.path.join(self.checkpoints_dir, "checkpoint_best.pt")
            if os.path.exists(best_path):
                self.load_checkpoint(best_path)
            else:
                self.logger.warning("checkpoint_best.pt not found. Evaluating test split on last epoch weights.")
                
            test_loader = self.get_dataloader(split="test")
            test_loss, test_anls, test_em = self.evaluate(test_loader, name="Test")
            self.logger.info(
                f"[Final Test Metrics] Test Loss: {test_loss:.4f} | "
                f"Test ANLS: {test_anls:.4f} | "
                f"Test EM: {test_em:.4f}"
            )
