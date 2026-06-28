import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHopReasoning(nn.Module):
    """
    Recurrent Multi-Hop Reasoning cell with memory updates over gated visual slots.
    Performs T attention hops exclusively over active memory slots.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config["model"]["embedding_dim"]
        self.num_hops = config["training"].get("num_hops", 3)
        
        # Cross-Attention over memory bank patches
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config["fusion"]["num_heads"],
            dropout=config["fusion"]["dropout"],
            batch_first=True
        )
        
        # Recurrent state update cell
        self.gru_cell = nn.GRUCell(self.embedding_dim, self.embedding_dim)
        
    def forward(self, query_embeddings, memory_slots, gates, slot_padding_mask=None):
        """
        Args:
            query_embeddings: [batch_size, num_query_tokens, D]
            memory_slots: [batch_size, total_slots, num_patches_per_slot, D] (usually [B, S, 256, D])
            gates: [batch_size, total_slots] (binary gates 0.0 or 1.0)
            slot_padding_mask: Optional boolean tensor of shape [batch_size, total_slots]
        Returns:
            updated_query: [batch_size, num_query_tokens, D]
            key_padding_mask: [batch_size, total_slots * num_patches_per_slot]
        """
        batch_size, total_slots, num_patches, D = memory_slots.shape
        device = memory_slots.device
        
        # 1. Flatten memory slots: [batch_size, total_slots * num_patches, D]
        flat_memory = memory_slots.reshape(batch_size, total_slots * num_patches, D)
        
        # 2. Construct attention key padding mask from Gumbel routing gates
        # Expand gates from [batch_size, total_slots] to [batch_size, total_slots * num_patches]
        # gate = 1.0 means active (padding_mask = False)
        # gate = 0.0 means inactive (padding_mask = True)
        gates_expanded = gates.unsqueeze(-1).expand(-1, -1, num_patches)
        gates_expanded = gates_expanded.reshape(batch_size, -1) # [batch_size, total_slots * num_patches]
        
        key_padding_mask = (gates_expanded == 0.0)
        
        # Fallback to prevent all-masked division by zero NaNs:
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)
        if all_masked.any():
            if slot_padding_mask is not None:
                slot_mask_expanded = slot_padding_mask.unsqueeze(-1).expand(-1, -1, num_patches).reshape(batch_size, -1)
                key_padding_mask = torch.where(all_masked, slot_mask_expanded, key_padding_mask)
            else:
                key_padding_mask = torch.where(all_masked, torch.zeros_like(key_padding_mask, dtype=torch.bool), key_padding_mask)
        
        # 3. Initialize reasoning state by mean pooling query tokens: [batch_size, D]
        h_t = torch.mean(query_embeddings, dim=1)
        
        # 4. Perform T recurrent hops of reasoning
        for hop in range(self.num_hops):
            # Query attends to flat gated memory
            # query: [batch_size, 1, D]
            # key, value: [batch_size, total_slots * num_patches, D]
            context, _ = self.cross_attention(
                query=h_t.unsqueeze(1),
                key=flat_memory,
                value=flat_memory,
                key_padding_mask=key_padding_mask
            )
            context = context.squeeze(1) # [batch_size, D]
            
            # Recurrently update query state
            h_t = self.gru_cell(context, h_t) # [batch_size, D]
            
        # 5. Add reasoning state back to original query tokens to produce updated queries
        updated_query = query_embeddings + h_t.unsqueeze(1)
        
        return updated_query, key_padding_mask
