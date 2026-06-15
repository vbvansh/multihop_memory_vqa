import torch
import torch.nn as nn
from PIL import Image
from transformers import ColPaliForRetrieval, ColPaliProcessor

class ColPaliVisionEncoder(nn.Module):
    """
    Encoder for document page images using ColPali.
    Passes PIL images through PaliGemma visual features to output multi-vector patch embeddings.
    """
    def __init__(self, model_name="vidore/colpali-v1.3-hf", device="cuda", dtype=torch.bfloat16, shared_model=None):
        super().__init__()
        self.device = device
        
        # Avoid duplicating 3B models in VRAM by referencing a shared instance
        if shared_model is not None:
            self.model = shared_model
        else:
            print(f"[ColPaliVisionEncoder] Loading visual model {model_name} in {dtype}...")
            self.model = ColPaliForRetrieval.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                low_cpu_mem_usage=True
            )
            
        # Initialize processor
        self.processor = ColPaliProcessor.from_pretrained(model_name)
        
    def forward(self, images):
        """
        Encodes a list of PIL Images into a batched representation of patch embeddings.
        Args:
            images: List of PIL.Image objects
        Returns:
            image_embeddings: Tensor of shape [batch_size, num_patches, embedding_dim]
        """
        # Process visual inputs
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass through ColPali vision tower
        outputs = self.model(**inputs)
        
        # ColPali outputs patch token embeddings of shape [batch_size, num_patches, embedding_dim]
        return outputs.embeddings
