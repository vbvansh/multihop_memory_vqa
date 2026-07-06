import torch
import torch.nn as nn
import torch.nn.functional as F


class RoutingLoss(nn.Module):
    """
    Stage-A (retriever) loss for question-guided sparse memory routing.

        L_total = lambda_page * L_page
                + lambda_sparsity * L_sparsity
                + lambda_entropy * L_entropy

    - L_page: supervised cross-entropy of aggregated per-page logits vs the
      gold answer page (answer_page_idx). This is the primary training signal
      for the router; the reader consumes the selected page downstream.
    - L_sparsity: pulls the *real-slot* activation ratio toward target_sparsity
      (padded slots excluded, unlike the old SparsityLoss which diluted it).
    - L_entropy: pushes the per-slot policy toward confident 0/1 decisions.
    """

    def __init__(self, config):
        super().__init__()
        self.lambda_page = config["training"].get("lambda_page", 1.0)
        self.lambda_sparsity = config["training"].get("lambda_sparsity", 0.3)
        self.lambda_entropy = config["training"].get("lambda_entropy", 0.01)
        self.target_sparsity = config["training"].get("target_sparsity", 0.25)

    def forward(self, page_logits, page_padding_mask, answer_page_idx, router_logits, gates, slot_padding_mask):
        """
        Args:
            page_logits: [B, num_pages] per-page relevance logits
            page_padding_mask: [B, num_pages] bool, True = dummy page
            answer_page_idx: [B] long, gold answer page index
            router_logits: [B, num_slots, 2] policy logits
            gates: [B, num_slots] binary Gumbel gates (0/1)
            slot_padding_mask: [B, num_slots] bool, True = padded slot
        Returns:
            total_loss: scalar tensor
            loss_dict: floats for logging (incl. page_acc)
        """
        num_pages = page_logits.shape[1]

        # 1. Supervised page-selection loss (mask out dummy pages before CE)
        masked_page_logits = page_logits.masked_fill(page_padding_mask, -1e9)
        target = answer_page_idx.clamp(min=0, max=num_pages - 1)
        loss_page = F.cross_entropy(masked_page_logits, target)

        # 2. Sparsity over REAL slots only
        real = (~slot_padding_mask).to(gates.dtype)                 # [B, num_slots]
        real_counts = real.sum(dim=1).clamp(min=1.0)                # [B]
        active_ratio = (gates * real).sum(dim=1) / real_counts      # [B]
        loss_sparsity = torch.mean((active_ratio - self.target_sparsity) ** 2)

        # 3. Entropy of the per-slot activation policy (real slots only)
        probs = F.softmax(router_logits, dim=-1)                    # [B, num_slots, 2]
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # [B, num_slots]
        entropy = (entropy * real).sum(dim=1) / real_counts
        loss_entropy = torch.mean(entropy)

        total_loss = (
            self.lambda_page * loss_page
            + self.lambda_sparsity * loss_sparsity
            + self.lambda_entropy * loss_entropy
        )

        # Page-selection accuracy (top-1) for logging
        with torch.no_grad():
            pred_page = torch.argmax(masked_page_logits, dim=1)
            page_acc = (pred_page == target).float().mean().item()

        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_page": loss_page.item(),
            "loss_sparsity": loss_sparsity.item(),
            "loss_entropy": loss_entropy.item(),
            "active_ratio": torch.mean(active_ratio).item(),
            "page_acc": page_acc,
        }
