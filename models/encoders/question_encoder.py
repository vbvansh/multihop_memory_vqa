import torch
import torch.nn as nn
from transformers import ColPaliForRetrieval, ColPaliProcessor

class ColPaliQuestionEncoder(nn.Module):
    """
    Encoder for questions using ColPali query projection.
    Maps natural language questions into the same late-interaction multi-vector space.
    """
    def __init__(self, model_name="vidore/colpali-v1.3-hf", device="cuda", dtype=torch.bfloat16, shared_model=None):
        super().__init__()
        self.device = device
        
        # Reference shared model to prevent duplicate VRAM occupancy
        if shared_model is not None:
            self.model = shared_model
        else:
            print(f"[ColPaliQuestionEncoder] Loading query model {model_name} in {dtype}...")
            self.model = ColPaliForRetrieval.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device
            )
            
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        
    def forward(self, questions):
        """
        Encodes a list of natural language question strings.
        Args:
            questions: List of string questions
        Returns:
            query_embeddings: Tensor of shape [batch_size, num_query_tokens, embedding_dim]
        """
        # Process query inputs
        inputs = self.processor(text=questions, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass through text/query projection
        outputs = self.model(**inputs)
        
        # ColPali outputs query token embeddings of shape [batch_size, num_query_tokens, embedding_dim]
        return outputs.embeddings
