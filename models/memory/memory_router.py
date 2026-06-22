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
        
    def forward(self, c_q, m_tilde):
        """
        Computes 2D logits representing activation preferences for all memory slots.
        Args:
            c_q: Contextualized query tensor of shape [batch_size, D]
            m_tilde: Contextualized slot tensor of shape [batch_size, num_slots, D]
        Returns:
            logits: Tensor of shape [batch_size, num_slots, 2]
        """
        batch_size = c_q.shape[0]
        num_slots = m_tilde.shape[1]
        
        # 1. Expand c_q along slot dimension: [batch_size, num_slots, D]
        c_q_expanded = c_q.unsqueeze(1).expand(-1, num_slots, -1)
        
        # 2. Construct interaction features: [c_q; m_tilde; c_q * m_tilde]
        # Shape: [batch_size, num_slots, 3 * D]
        interaction_features = torch.cat(
            [c_q_expanded, m_tilde, c_q_expanded * m_tilde], 
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

