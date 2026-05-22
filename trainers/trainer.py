import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder
from models.decoder.answer_head import AnswerHead
from datasets.docvqa import DocVQADataset, collate_fn
from utils.logger import setup_logger
from utils.metrics import calculate_anls, calculate_exact_match

class ColPaliTrainer:
    """
    Main training engine for Phase 1 VQA Baseline.
    Manages shared models, loaders, optimizers, backprop, and overfitting sanity checks.
    """
    def __init__(self, model_config_path, train_config_path):
        # 1. Load Configurations
        with open(model_config_path, "r") as f:
            self.model_config = yaml.safe_load(f)
        with open(train_config_path, "r") as f:
            self.train_config = yaml.safe_load(f)
            
        # 2. Setup Logging
        self.logger = setup_logger()
        self.logger.info("Initializing ColPali Trainer...")
        
        # 3. Setup Hardware Device
        self.device = torch.device(self.model_config["model"]["device"] if torch.cuda.is_available() else "cpu")
        self.dtype_str = self.model_config["model"]["dtype"]
        self.dtype = torch.bfloat16 if self.dtype_str == "bfloat16" else (torch.float16 if self.dtype_str == "float16" else torch.float32)
        
        self.logger.info(f"Targeting device: {self.device} with data type: {self.dtype}")
        
        # 4. Initialize Shared Model Backbone (Avoid duplicate 3B VRAM loading)
        model_name = self.model_config["model"]["name"]
        
        # Load unified model first
        from transformers import ColPaliForRetrieval
        self.logger.info(f"Loading shared ColPali backbone from {model_name}...")
        self.shared_model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            device_map=self.model_config["model"]["device"]
        )
        
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
        
        # Initialize decoder/answer prediction head
        self.answer_head = AnswerHead(self.model_config, vocab_size=vocab_size).to(self.device)
        self.answer_head = self.answer_head.to(self.dtype)
        
        # 6. Set up Optimizer and Loss Function
        # We optimize the projection layers of the encoders and all layers in the new AnswerHead
        trainable_params = list(self.answer_head.parameters())
        # (Optional: Add encoder LoRA/adapters parameters to optimizer here if needed)
        
        self.optimizer = optim.AdamW(
            trainable_params,
            lr=float(self.train_config["training"]["lr"]),
            weight_decay=float(self.train_config["training"]["weight_decay"])
        )
        
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
    def get_dataloader(self, is_train=True):
        """Loads either the debug dataset or the real dataset."""
        debug_mode = self.train_config["debug"]["enable"]
        dataset = DocVQADataset(self.train_config, is_train=is_train, debug=debug_mode)
        
        loader = DataLoader(
            dataset,
            batch_size=self.train_config["training"]["batch_size"] if not debug_mode else 2,
            shuffle=is_train,
            collate_fn=collate_fn,
            drop_last=False
        )
        return loader
        
    def train_epoch(self, loader, epoch):
        self.answer_head.train()
        self.shared_model.eval() # Keep backbone frozen in early phases
        
        epoch_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{self.train_config['training']['epochs']}")
        
        for step, batch in enumerate(loop):
            self.optimizer.zero_grad()
            
            # 1. Encode Images and Questions
            with torch.no_grad():
                # Extract image patch embeddings: [batch_size, num_patches, D]
                doc_feats = self.vision_encoder(batch["images"])
                # Extract query token embeddings: [batch_size, num_query_tokens, D]
                query_feats = self.question_encoder(batch["questions"])
                
            # Cast outputs to standard dtype and device for the projection/attention layers
            doc_feats = doc_feats.to(self.device).to(self.dtype)
            query_feats = query_feats.to(self.device).to(self.dtype)
            
            # 2. Forward pass through prediction head: [batch, max_answer_len, vocab_size]
            logits = self.answer_head(query_feats, doc_feats)
            
            # 3. Tokenize answers for Cross-Entropy target labels
            # Tokenize answers and pad/truncate to max_answer_len
            targets = self.tokenizer(
                batch["answers"],
                padding="max_length",
                max_length=self.model_config["decoder"]["max_answer_len"],
                truncation=True,
                return_tensors="pt"
            )["input_ids"].to(self.device)
            
            # Calculate classification loss
            loss = self.loss_fn(
                logits.view(-1, logits.size(-1)), # [batch_size * max_len, vocab_size]
                targets.view(-1)                  # [batch_size * max_len]
            )
            
            # 4. Backward Pass & Parameter Optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.answer_head.parameters(), self.train_config["training"]["grad_clip"])
            self.optimizer.step()
            
            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_loss = epoch_loss / len(loader)
        self.logger.info(f"Epoch {epoch+1} Complete. Average Training Loss: {avg_loss:.4f}")
        return avg_loss
        
    def evaluate(self, loader):
        """Runs validation and decodes predicted answer tokens back to text strings."""
        self.answer_head.eval()
        
        predictions = []
        ground_truths = []
        
        with torch.no_grad():
            for batch in loader:
                doc_feats = self.vision_encoder(batch["images"]).to(self.device).to(self.dtype)
                query_feats = self.question_encoder(batch["questions"]).to(self.device).to(self.dtype)
                
                # Get logits [batch, max_answer_len, vocab_size]
                logits = self.answer_head(query_feats, doc_feats)
                pred_ids = torch.argmax(logits, dim=-1) # [batch, max_answer_len]
                
                # Decode tokens back into text
                decoded_preds = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
                
                predictions.extend(decoded_preds)
                ground_truths.extend([[ans] for ans in batch["answers"]])
                
        # Calculate metrics
        total_anls = 0.0
        total_em = 0.0
        
        for pred, gts in zip(predictions, ground_truths):
            total_anls += calculate_anls(pred, gts)
            total_em += calculate_exact_match(pred, gts)
            self.logger.info(f"Q: ... | Pred: '{pred}' | Ground Truth: '{gts[0]}'")
            
        avg_anls = total_anls / len(predictions) if predictions else 0.0
        avg_em = total_em / len(predictions) if predictions else 0.0
        
        self.logger.info(f"Evaluation Complete. ANLS: {avg_anls:.4f} | Exact Match: {avg_em:.4f}")
        return avg_anls, avg_em
        
    def run(self):
        """Runs the main training and evaluation loop."""
        self.logger.info("Loading dataset loader...")
        loader = self.get_dataloader(is_train=True)
        
        self.logger.info("Starting training loop...")
        epochs = self.train_config["training"]["epochs"]
        
        for epoch in range(epochs):
            self.train_epoch(loader, epoch)
            
            # Run quick evaluation on training debug samples to check overfitting
            if self.train_config["debug"]["enable"]:
                self.logger.info("Evaluating overfit performance on debug samples:")
                self.evaluate(loader)
