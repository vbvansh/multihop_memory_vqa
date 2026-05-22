import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    """
    Structured visual memory bank that partitions ColPali page-level patch embeddings
    into localized, chunk-level memory slots (sub-page visual blocks).
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config["model"]["embedding_dim"]
        
        # Grid parameters: divide each page image into a GxG grid of chunks
        # e.g., a 2x2 grid yields 4 semantic visual slots per page
        self.grid_size = 2  # 2x2 quadrants
        self.num_slots_per_page = self.grid_size * self.grid_size
        
    def forward(self, page_embeddings):
        """
        Partitions page-level patch embeddings into chunk-level memory slots.
        Args:
            page_embeddings: Tensor of shape [batch_size, num_patches, D] (usually 1030 patches, D=128)
        Returns:
            memory_slots: Dict of chunk slot embeddings and visual bboxes
        """
        batch_size = page_embeddings.shape[0]
        device = page_embeddings.device
        
        # ColPali-v1.3 usually has 1030 tokens: 1024 patches (32x32 grid) + 6 prefix/suffix tokens
        # We extract the 1024 core visual patch tokens
        num_patches = page_embeddings.shape[1]
        
        if num_patches >= 1024:
            core_patches = page_embeddings[:, :1024, :]  # [batch_size, 1024, D]
        else:
            # Fallback if page has fewer patches
            core_patches = page_embeddings
            
        # Reshape to 2D grid: [batch_size, 32, 32, D]
        grid_dim = 32
        grid_patches = core_patches.view(batch_size, grid_dim, grid_dim, self.embedding_dim)
        
        # We slice the 32x32 grid into a self.grid_size x self.grid_size chunk grid
        # For a 2x2 grid, each chunk is 16x16 patches
        chunk_span = grid_dim // self.grid_size
        
        slots_embeddings = []
        slots_metadata = []
        
        for r_idx in range(self.grid_size):
            for c_idx in range(self.grid_size):
                # Slice patches in this grid quadrant
                row_start = r_idx * chunk_span
                row_end = (r_idx + 1) * chunk_span
                col_start = c_idx * chunk_span
                col_end = (c_idx + 1) * chunk_span
                
                # Extract chunk patches: [batch_size, chunk_span, chunk_span, D]
                chunk_patches = grid_patches[:, row_start:row_end, col_start:col_end, :]
                # Flatten spatial dimensions: [batch_size, num_patches_per_chunk, D] (16x16 = 256 patches)
                chunk_patches_flat = chunk_patches.reshape(batch_size, -1, self.embedding_dim)
                
                # Calculate normalized bounding box coordinates for this chunk in [0, 1000] space
                x1 = int((col_start / grid_dim) * 1000)
                y1 = int((row_start / grid_dim) * 1000)
                x2 = int((col_end / grid_dim) * 1000)
                y2 = int((row_end / grid_dim) * 1000)
                
                slots_embeddings.append(chunk_patches_flat)
                slots_metadata.append({
                    "bbox": [x1, y1, x2, y2],
                    "name": f"quadrant_{r_idx}_{c_idx}"
                })
                
        # Stack slots: shape [batch_size, num_slots, num_patches_per_chunk, D]
        # For a 2x2 grid: [batch_size, 4, 256, 128]
        stacked_slots = torch.stack(slots_embeddings, dim=1)
        
        return {
            "embeddings": stacked_slots, # [batch_size, num_slots, num_patches_per_chunk, D]
            "metadata": slots_metadata   # List of dicts describing spatial quadrants
        }
        
    def get_pooled_embeddings(self, stacked_slots):
        """
        Pools (averages) multi-vector patch embeddings inside each slot into a single vector.
        Useful for simple single-vector routing policy networks.
        Args:
            stacked_slots: [batch_size, num_slots, num_patches_per_chunk, D]
        Returns:
            pooled_slots: [batch_size, num_slots, D]
        """
        return torch.mean(stacked_slots, dim=2)
