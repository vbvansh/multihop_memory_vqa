import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelRouter(nn.Module):
    """
    Differentiable Gumbel-Softmax Sparse Gating module.
    Takes 2D logits from the Policy Router and produces differentiable binary decisions (g_k in {0, 1})
    to dynamically mask memory chunks.
    """
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature  # \tau parameter to control discreteness
        
    def set_temperature(self, temperature):
        """Sets the temperature value dynamically (used for annealing during training)."""
        self.temperature = temperature
        
    def forward(self, logits, hard=True):
        """
        Computes differentiable binary gates for memory slots.
        Args:
            logits: Tensor of shape [batch_size, num_slots, 2]
                    where index 0 is 'Inactivate' and index 1 is 'Activate'
            hard: If True, returns a strict one-hot binary vector in the forward pass,
                  but retains smooth continuous softmax gradients in the backward pass.
        Returns:
            z: Binary gate tensor of shape [batch_size, num_slots]
        """
        # gumbel_softmax handles:
        # 1. Adding Gumbel noise.
        # 2. Argmax in forward pass (if hard=True) -> outputting strict [0, 1] or [1, 0].
        # 3. Softmax in backward pass -> ensuring continuous gradients flow.
        binary_one_hot = F.gumbel_softmax(
            logits, 
            tau=self.temperature, 
            hard=hard
        )
        
        # We extract index 1 ('Activate') representing our final binary gate z
        # z contains strict 0.0 or 1.0 floats of shape [batch_size, num_slots]
        z = binary_one_hot[:, :, 1]
        
        return z
