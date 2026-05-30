import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryRouter(nn.Module):
    """
    Policy Network (Router) that computes 2D activation logits
    for each memory slot based on the semantic query-chunk interaction.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config["model"]["embedding_dim"]
        self.projection_dim = config["model"]["projection_dim"]
        
        # Query-slot interaction vector dimension is 3 * D (c_q, m_k, and c_q * m_k)
        self.interaction_dim = 3 * self.embedding_dim
        
        # Policy network MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.interaction_dim, self.projection_dim),
            nn.LayerNorm(self.projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.projection_dim, 2)  # Logits: [0: Inactivate, 1: Activate]
        )
        
    def forward(self, query_embeddings, slot_embeddings):
        """
        Computes 2D logits representing activation preferences for all memory slots.
        Args:
            query_embeddings: Tensor of shape [batch_size, num_query_tokens, D]
            slot_embeddings: Tensor of shape [batch_size, num_slots, num_patches_per_chunk, D]
        Returns:
            logits: Tensor of shape [batch_size, num_slots, 2]
        """
        batch_size = query_embeddings.shape[0]
        num_slots = slot_embeddings.shape[1]
        
        # 1. Pool query embeddings: average over query tokens to get c_q [batch_size, D]
        c_q = torch.mean(query_embeddings, dim=1)  # [batch_size, D]
        
        # 2. Pool slot embeddings: average over visual patches to get m_k [batch_size, num_slots, D]
        m_k = torch.mean(slot_embeddings, dim=2)  # [batch_size, num_slots, D]
        
        # 3. Expand c_q along slot dimension: [batch_size, num_slots, D]
        c_q_expanded = c_q.unsqueeze(1).expand(-1, num_slots, -1)
        
        # 4. Construct interaction features: [c_q; m_k; c_q * m_k]
        # Shape: [batch_size, num_slots, 3 * D]
        interaction_features = torch.cat(
            [c_q_expanded, m_k, c_q_expanded * m_k], 
            dim=-1
        )
        
        # Flatten batch and slot dimensions to run through MLP
        # Shape: [batch_size * num_slots, 3 * D]
        interaction_flat = interaction_features.view(-1, self.interaction_dim)
        
        # Forward through Policy MLP -> [batch_size * num_slots, 2]
        logits_flat = self.mlp(interaction_flat)
        
        # Reshape back: [batch_size, num_slots, 2]
        logits = logits_flat.view(batch_size, num_slots, 2)
        
        return logits
