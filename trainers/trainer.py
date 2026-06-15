import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder
from models.memory.memory_bank import MemoryBank
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
            
        # 2. Setup Logging
        self.logger = setup_logger()
        self.logger.info("Initializing Sparse Routing ColPali Trainer...")
        
        # 3. Setup Hardware Device
        self.device = torch.device(self.config["model"]["device"] if torch.cuda.is_available() else "cpu")
        self.dtype_str = self.config["model"]["dtype"]
        self.dtype = torch.bfloat16 if self.dtype_str == "bfloat16" else (torch.float16 if self.dtype_str == "float16" else torch.float32)
        
        self.logger.info(f"Targeting device: {self.device} with data type: {self.dtype}")
        
        model_name = self.config["model"]["name"]
        
        # 4. Handle precomputed embeddings mode vs standard VLM mode
        self.use_precomputed = self.config["debug"].get("use_precomputed_embeddings", False) and self.config["debug"].get("enable", False)
        
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
                    device_map="auto"
                )
            else:
                self.logger.info(f"Loading shared ColPali backbone from {model_name} in standard {self.dtype}...")
                self.shared_model = ColPaliForRetrieval.from_pretrained(
                    model_name,
                    torch_dtype=self.dtype,
                    device_map=self.config["model"]["device"]
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
        self.router = MemoryRouter(self.config).to(self.device).to(self.dtype)
        self.gumbel = GumbelRouter(temperature=0.5).to(self.device)
        self.multi_hop = MultiHopReasoning(self.config).to(self.device).to(self.dtype)
        self.answer_head = AnswerHead(self.config, vocab_size=vocab_size).to(self.device).to(self.dtype)
        
        # Sparsity & Entropy Regularization Loss
        self.sparsity_loss_fn = SparsityLoss(self.config, ignore_index=self.tokenizer.pad_token_id)
        
        # 6. Set up Optimizer
        # Optimize the dynamic router, multi-hop cell, and answer prediction layers
        trainable_params = (
            list(self.router.parameters()) +
            list(self.multi_hop.parameters()) +
            list(self.answer_head.parameters())
        )
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=float(self.config["training"]["lr"]),
            weight_decay=float(self.config["training"]["weight_decay"])
        )
        
    def get_dataloader(self, is_train=True):
        """Loads either the debug dataset or the real dataset."""
        debug_mode = self.config["debug"]["enable"]
        dataset = DocVQADataset(self.config, is_train=is_train, debug=debug_mode)
        
        loader = DataLoader(
            dataset,
            batch_size=self.config["training"]["batch_size"] if not debug_mode else 2,
            shuffle=is_train,
            collate_fn=collate_fn,
            drop_last=False
        )
        return loader
        
    def train_epoch(self, loader, epoch):
        self.router.train()
        self.multi_hop.train()
        self.answer_head.train()
        if self.shared_model is not None:
            self.shared_model.eval()
        
        epoch_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{self.config['training']['epochs']}")
        
        precomputed_dir = self.config["debug"]["precomputed_dir"]
        
        for step, batch in enumerate(loop):
            self.optimizer.zero_grad()
            
            batch_size = len(batch["questions"])
            batch_loss = 0.0
            
            # Process each document in the batch sequentially to save GPU memory (VRAM preservation)
            for i in range(batch_size):
                question_id = batch["question_ids"][i]
                answer = batch["answers"][i]
                
                # 1. Load or encode visual page representations
                if self.use_precomputed:
                    # Read precomputed embeddings directly from disk (takes 0 GPU VRAM)
                    emb_path = os.path.join(precomputed_dir, f"vision_{question_id}.pt")
                    page_emb = torch.load(emb_path, map_location=self.device).to(self.dtype)
                else:
                    # Encode page images sequentially to stay under the 4GB VRAM limit
                    images = batch["images"][i]
                    page_embs = []
                    with torch.no_grad():
                        for img in images:
                            emb = self.vision_encoder([img])  # [1, num_patches, D]
                            page_embs.append(emb.cpu())
                    page_emb = torch.cat(page_embs, dim=0).to(self.device).to(self.dtype)
                
                # 2. Partition into visual memory slots
                memory_outputs = self.memory_bank(page_emb)
                chunk_embeddings = memory_outputs["embeddings"].to(self.device).to(self.dtype)
                
                # 3. Load or encode question text
                if self.use_precomputed:
                    q_emb_path = os.path.join(precomputed_dir, f"question_{question_id}.pt")
                    query_emb = torch.load(q_emb_path, map_location=self.device).to(self.dtype)
                else:
                    question = batch["questions"][i]
                    with torch.no_grad():
                        query_emb = self.question_encoder([question]).to(self.device).to(self.dtype)
                    
                # 4. Run dynamic routing & gating
                router_logits = self.router(query_emb, chunk_embeddings)
                z = self.gumbel(router_logits, hard=True)
                
                # 5. Multi-Hop Reasoning
                updated_query, key_padding_mask = self.multi_hop(query_emb, chunk_embeddings, z)
                
                # 6. Gated Answer prediction
                flat_memory = chunk_embeddings.reshape(1, -1, self.config["model"]["embedding_dim"])
                pred_logits = self.answer_head(updated_query, flat_memory, key_padding_mask=key_padding_mask)
                
                # 7. Tokenize target labels
                targets = self.tokenizer(
                    [answer],
                    padding="max_length",
                    max_length=self.config["decoder"]["max_answer_len"],
                    truncation=True,
                    return_tensors="pt"
                )["input_ids"].to(self.device)
                
                # Reshape for loss computation
                flat_logits = pred_logits.view(-1, pred_logits.size(-1))
                flat_targets = targets.view(-1)
                
                # 8. Compute composite loss (vqa + sparsity + entropy)
                loss, loss_details = self.sparsity_loss_fn(
                    flat_logits,
                    flat_targets,
                    router_logits,
                    z
                )
                
                # Scale loss by batch_size for correct gradient accumulation
                loss_scaled = loss / batch_size
                loss_scaled.backward()
                
                batch_loss += loss.item()
            
            # Step parameters after accumulating gradients across the batch
            trainable_params = (
                list(self.router.parameters()) + 
                list(self.multi_hop.parameters()) + 
                list(self.answer_head.parameters())
            )
            torch.nn.utils.clip_grad_norm_(trainable_params, self.config["training"]["grad_clip"])
            self.optimizer.step()
            
            epoch_loss += batch_loss / batch_size
            loop.set_postfix(loss=batch_loss / batch_size)
            
        avg_loss = epoch_loss / len(loader)
        self.logger.info(f"Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f}")
        return avg_loss
        
    def evaluate(self, loader):
        """Runs validation and decodes predicted answer tokens back to text strings."""
        self.router.eval()
        self.multi_hop.eval()
        self.answer_head.eval()
        
        predictions = []
        ground_truths = []
        
        precomputed_dir = self.config["debug"]["precomputed_dir"]
        
        with torch.no_grad():
            for batch in loader:
                batch_size = len(batch["questions"])
                for i in range(batch_size):
                    question_id = batch["question_ids"][i]
                    answer = batch["answers"][i]
                    
                    # 1. Load or encode visual page representations
                    if self.use_precomputed:
                        emb_path = os.path.join(precomputed_dir, f"vision_{question_id}.pt")
                        page_emb = torch.load(emb_path, map_location=self.device).to(self.dtype)
                    else:
                        images = batch["images"][i]
                        page_embs = []
                        for img in images:
                            emb = self.vision_encoder([img])
                            page_embs.append(emb.cpu())
                        page_emb = torch.cat(page_embs, dim=0).to(self.device).to(self.dtype)
                    
                    # Memory bank
                    memory_outputs = self.memory_bank(page_emb)
                    chunk_embeddings = memory_outputs["embeddings"].to(self.device).to(self.dtype)
                    
                    # Question
                    if self.use_precomputed:
                        q_emb_path = os.path.join(precomputed_dir, f"question_{question_id}.pt")
                        query_emb = torch.load(q_emb_path, map_location=self.device).to(self.dtype)
                    else:
                        question = batch["questions"][i]
                        query_emb = self.question_encoder([question]).to(self.device).to(self.dtype)
                    
                    # Routing
                    router_logits = self.router(query_emb, chunk_embeddings)
                    z = self.gumbel(router_logits, hard=True)
                    
                    # Multi-Hop
                    updated_query, key_padding_mask = self.multi_hop(query_emb, chunk_embeddings, z)
                    
                    # Gated Answer Head
                    flat_memory = chunk_embeddings.reshape(1, -1, self.config["model"]["embedding_dim"])
                    pred_logits = self.answer_head(updated_query, flat_memory, key_padding_mask=key_padding_mask)
                    
                    pred_ids = torch.argmax(pred_logits, dim=-1) # [1, max_answer_len]
                    decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                    
                    predictions.extend(decoded_preds)
                    ground_truths.append([answer])
                    
        # Calculate metrics
        total_anls = 0.0
        total_em = 0.0
        
        for pred, gts in zip(predictions, ground_truths):
            total_anls += calculate_anls(pred, gts)
            total_em += calculate_exact_match(pred, gts)
            self.logger.info(f"Pred: '{pred}' | Ground Truth: '{gts[0]}'")
            
        avg_anls = total_anls / len(predictions) if predictions else 0.0
        avg_em = total_em / len(predictions) if predictions else 0.0
        
        self.logger.info(f"Evaluation Complete. ANLS: {avg_anls:.4f} | Exact Match: {avg_em:.4f}")
        return avg_anls, avg_em
        
    def run(self):
        """Runs the main training and evaluation loop."""
        self.logger.info("Loading dataset loader...")
        loader = self.get_dataloader(is_train=True)
        
        self.logger.info("Starting training loop...")
        epochs = self.config["training"]["epochs"]
        
        for epoch in range(epochs):
            self.train_epoch(loader, epoch)
            
            # Run quick evaluation on training debug samples to check overfitting
            if self.config["debug"]["enable"]:
                self.logger.info("Evaluating overfit performance on debug samples:")
                self.evaluate(loader)
