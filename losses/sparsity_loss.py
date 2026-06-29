import torch
import torch.nn as nn
import torch.nn.functional as F

class SparsityLoss(nn.Module):
    """
    Composite loss function for Question-Guided Sparse Memory Routing.
    Computes:
    L_total = L_vqa + lambda_sparsity * L_sparsity + lambda_entropy * L_entropy
    """
    def __init__(self, config, ignore_index=-100):
        super().__init__()
        self.config = config
        self.ignore_index = ignore_index
        
        # Loss components configurations
        self.lambda_sparsity = config["training"].get("lambda_sparsity", 0.1)
        self.lambda_entropy = config["training"].get("lambda_entropy", 0.01)
        self.target_sparsity = config["training"].get("target_sparsity", 0.25)
        
        self.vqa_loss_fn = nn.CrossEntropyLoss(ignore_index=self.ignore_index, label_smoothing=0.1)
        
    def forward(self, logits, targets, router_logits, gates):
        """
        Args:
            logits: [batch_size * max_answer_len, vocab_size] (predicted answer token logits)
            targets: [batch_size * max_answer_len] (ground truth target token IDs)
            router_logits: [batch_size, total_slots, 2] (unnormalized routing policy logits)
            gates: [batch_size, total_slots] (binary gates outputted by Gumbel Router, e.g. 0.0 or 1.0)
        Returns:
            total_loss: Scalar float tensor representing combined loss
            loss_dict: Dict containing individual loss values for logging
        """
        # 1. Answer Generation Loss (VQA Loss) with explicit padding mask filtering
        if self.ignore_index is not None:
            loss_mask = (targets != self.ignore_index)
            if loss_mask.any():
                loss_vqa = F.cross_entropy(logits[loss_mask], targets[loss_mask], label_smoothing=0.1)
            else:
                loss_vqa = F.cross_entropy(logits, targets, ignore_index=self.ignore_index, label_smoothing=0.1)
        else:
            loss_vqa = F.cross_entropy(logits, targets, label_smoothing=0.1)
        
        # 2. Sparsity Constraint Loss (L2 distance to target active ratio)
        # gates has values 0.0 or 1.0. Mean along slots gives the activation ratio per sample.
        active_ratio = torch.mean(gates, dim=1) # [batch_size]
        loss_sparsity = torch.mean((active_ratio - self.target_sparsity) ** 2)
        
        # 3. Entropy Regularization Loss (minimizing entropy pushes policy probabilities to be confident (0 or 1))
        # Compute probabilities: [batch_size, total_slots, 2]
        probs = F.softmax(router_logits, dim=-1)
        # Shannon Entropy per slot: - sum(p * log(p))
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1) # [batch_size, total_slots]
        loss_entropy = torch.mean(entropy)
        
        # 4. Combine
        total_loss = loss_vqa + (self.lambda_sparsity * loss_sparsity) + (self.lambda_entropy * loss_entropy)
        
        return total_loss, {
            "loss_total": total_loss.item(),
            "loss_vqa": loss_vqa.item(),
            "loss_sparsity": loss_sparsity.item(),
            "loss_entropy": loss_entropy.item(),
            "active_ratio": torch.mean(active_ratio).item()
        }
