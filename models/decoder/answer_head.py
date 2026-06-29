import torch
import torch.nn as nn
import torch.nn.functional as F

class AnswerHead(nn.Module):
    """
    VQA Answer Head for Phase 1.
    Performs late-interaction cross-attention to fuse query and document embeddings,
    then uses a linear sequence projector to predict answer token logits.
    """
    def __init__(self, config, vocab_size=256000): # Standard Gemma tokenizer has ~256k vocab size
        super().__init__()
        self.config = config
        self.embedding_dim = config["model"]["embedding_dim"]
        self.projection_dim = config["model"]["projection_dim"]
        self.max_answer_len = config["decoder"]["max_answer_len"]
        
        # 1. Late-Interaction Cross-Attention Fusion
        # Query tokens (Q) attend to Document patches (K, V)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config["fusion"]["num_heads"],
            dropout=config["fusion"]["dropout"],
            batch_first=True
        )
        
        # 2. Projection layers
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        
        # Maps query-attended context to a single pooled representation
        self.pooler = nn.Sequential(
            nn.Linear(self.embedding_dim, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim)
        )
        
        # 3. Sequence Generator Head: predicts token logits for [max_answer_len] tokens
        # Outputs tensor shape: [batch_size, max_answer_len, vocab_size]
        self.decoder_projection = nn.Sequential(
            nn.Linear(self.projection_dim, self.projection_dim),
            nn.ReLU(),
            nn.Linear(self.projection_dim, self.max_answer_len * self.projection_dim)
        )
        
        self.dropout = nn.Dropout(p=0.3)
        
        # Final output projection to vocabulary logits
        self.vocab_projection = nn.Linear(self.projection_dim, vocab_size)
        
    def forward(self, query_embeddings, doc_embeddings, key_padding_mask=None):
        """
        Fuses query and document embeddings and predicts answer token logits.
        Args:
            query_embeddings: [batch_size, num_query_tokens, D]
            doc_embeddings: [batch_size, num_patches, D]
            key_padding_mask: [batch_size, num_patches] optional mask indicating inactive patches
        Returns:
            logits: Predicted token logits [batch_size, max_answer_len, vocab_size]
        """
        batch_size = query_embeddings.shape[0]
        
        # 1. Cross-Attention: Query attends to Document patches
        # Query is target (query), Document is key/value (source)
        attn_output, _ = self.cross_attention(
            query=query_embeddings,
            key=doc_embeddings,
            value=doc_embeddings,
            key_padding_mask=key_padding_mask
        )
        
        # Residual connection and normalization
        fused = self.layer_norm(attn_output + query_embeddings) # [batch_size, num_query_tokens, D]
        
        # 2. Global pooling over query tokens to get a single vector per sample
        pooled = torch.mean(fused, dim=1) # [batch_size, D]
        features = self.pooler(pooled) # [batch_size, projection_dim]
        
        # 3. Project to sequence of tokens
        seq_features = self.decoder_projection(features) # [batch_size, max_answer_len * projection_dim]
        seq_features = seq_features.view(batch_size, self.max_answer_len, self.projection_dim)
        
        # Apply dropout to hidden states before final vocabulary projection
        seq_features = self.dropout(seq_features)
        
        # Map to vocabulary logits
        logits = self.vocab_projection(seq_features) # [batch_size, max_answer_len, vocab_size]
        
        return logits
