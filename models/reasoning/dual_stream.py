import torch
import torch.nn as nn

class DualStreamProcessor(nn.Module):
    """
    Decoupled Dual-Stream pre-fusion processing (Stream A and Stream B).
    Separates question-visual context processing (Stream A) from document memory
    reasoning (Stream B) prior to the routing stage.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config["model"]["embedding_dim"]
        self.projection_dim = config["model"]["projection_dim"]
        
        # Dual-stream config defaults
        ds_config = config.get("dual_stream", {})
        self.num_heads = ds_config.get("num_heads", config["fusion"]["num_heads"])
        self.num_layers = ds_config.get("num_layers", 1)
        self.dropout = ds_config.get("dropout", config["fusion"]["dropout"])
        
        # Stream A: Cross-Attention query attends to global visual slot context
        self.stream_a_cross_attn = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=self.dropout,
            batch_first=True
        )
        self.stream_a_norm = nn.LayerNorm(self.embedding_dim)
        
        # Stream B: Transformer Encoder for slot-level self-attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_dim,
            nhead=self.num_heads,
            dim_feedforward=self.projection_dim,
            dropout=self.dropout,
            batch_first=True
        )
        self.stream_b_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.num_layers
        )
        
    def forward(self, query_embeddings, slot_embeddings):
        """
        Fuses and contextualizes the streams separately.
        Args:
            query_embeddings: [batch_size, num_query_tokens, D]
            slot_embeddings: [batch_size, num_slots, num_patches_per_chunk, D]
        Returns:
            c_q: [batch_size, D] pooled query context vector
            contextualized_slots: [batch_size, num_slots, D] slot-level representations for routing
            contextualized_patches: [batch_size, num_slots, num_patches_per_chunk, D] residual-enriched patches
        """
        batch_size = query_embeddings.shape[0]
        num_slots = slot_embeddings.shape[1]
        
        # --- Stream B: Document Memory Reasoning Stream (Independent of Question) ---
        # 1. Pool slot embeddings over patches: [batch_size, num_slots, D]
        slot_pooled = torch.mean(slot_embeddings, dim=2)
        
        # 2. Run transformer encoder over the slot sequence to perform intra-memory reasoning
        contextualized_slots = self.stream_b_transformer(slot_pooled) # [batch_size, num_slots, D]
        
        # 3. Enrich the original patch-level embeddings residually: [batch_size, num_slots, num_patches, D]
        contextualized_patches = slot_embeddings + contextualized_slots.unsqueeze(2)
        
        # --- Stream A: Question-Visual Context Stream ---
        # 1. Query attends to contextualized slot representations
        # Query: [batch_size, num_query_tokens, D]
        # Key/Value: [batch_size, num_slots, D]
        attn_out, _ = self.stream_a_cross_attn(
            query=query_embeddings,
            key=contextualized_slots,
            value=contextualized_slots
        )
        
        # 2. Residual connection and normalization
        query_fused = self.stream_a_norm(attn_out + query_embeddings) # [batch_size, num_query_tokens, D]
        
        # 3. Pool query tokens to yield a single context vector: [batch_size, D]
        c_q = torch.mean(query_fused, dim=1)
        
        return c_q, contextualized_slots, contextualized_patches
