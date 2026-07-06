import torch
import torch.nn as nn
import torch.nn.functional as F


def aggregate_page_logits(router_logits, slot_padding_mask, slots_per_page=4):
    """
    Reduces per-slot 'activate' logits to per-page relevance logits.

    Slots are laid out contiguously per page (page p owns slots
    [p*slots_per_page : (p+1)*slots_per_page]), matching MemoryBank's
    [B, num_pages * slots_per_page, ...] ordering.

    Args:
        router_logits: [B, num_slots, 2] policy logits (index 1 = 'activate')
        slot_padding_mask: [B, num_slots] bool, True = padded dummy slot
        slots_per_page: quadrant slots per page (MemoryBank.num_slots_per_page)
    Returns:
        page_logits: [B, num_pages] mean activate-logit over each page's real quadrants
        page_padding_mask: [B, num_pages] bool, True = fully-padded (dummy) page
    """
    B, S, _ = router_logits.shape
    num_pages = S // slots_per_page

    act = router_logits[..., 1].view(B, num_pages, slots_per_page)          # [B, P, q]
    mask = slot_padding_mask.view(B, num_pages, slots_per_page)             # True = pad

    # torch.where (not multiply) so NaN/inf in padded slots can't contaminate the sum
    valid = (~mask).to(act.dtype)
    act_zeroed = torch.where(mask, torch.zeros_like(act), act)
    denom = valid.sum(dim=-1).clamp(min=1.0)                                # [B, P]
    page_logits = act_zeroed.sum(dim=-1) / denom                           # [B, P]

    page_padding_mask = mask.all(dim=-1)                                    # [B, P]
    return page_logits, page_padding_mask


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

