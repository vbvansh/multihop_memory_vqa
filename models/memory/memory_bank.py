import torch
import torch.nn as nn
import torch.nn.functional as F

class MemoryBank(nn.Module):
    """
    Structured visual memory bank that partitions ColPali page-level patch embeddings
    into localized, chunk-level memory slots across all pages of a document.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_dim = config["model"]["embedding_dim"]
        
        # Grid parameters: divide each page image into a GxG grid of chunks
        self.grid_size = 2  # 2x2 quadrants
        self.num_slots_per_page = self.grid_size * self.grid_size
        
    def forward(self, page_embeddings):
        """
        Partitions page-level patch embeddings into chunk-level memory slots across all pages.
        Args:
            page_embeddings: Tensor of shape [num_pages, num_patches, D] or [batch_size, num_pages, num_patches, D]
        Returns:
            memory_slots: Dict containing consolidated slot embeddings and metadata
        """
        if page_embeddings.dim() == 4:
            batch_size, num_pages, num_patches, D = page_embeddings.shape
            if num_patches >= 1024:
                core_patches = page_embeddings[:, :, :1024, :]
            else:
                core_patches = page_embeddings
                
            grid_dim = 32
            grid_patches = core_patches.view(batch_size, num_pages, grid_dim, grid_dim, self.embedding_dim)
            chunk_span = grid_dim // self.grid_size
            
            slots_embeddings = []
            for r_idx in range(self.grid_size):
                for c_idx in range(self.grid_size):
                    row_start = r_idx * chunk_span
                    row_end = (r_idx + 1) * chunk_span
                    col_start = c_idx * chunk_span
                    col_end = (c_idx + 1) * chunk_span
                    
                    chunk = grid_patches[:, :, row_start:row_end, col_start:col_end, :]
                    chunk_flat = chunk.reshape(batch_size, num_pages, -1, self.embedding_dim)
                    slots_embeddings.append(chunk_flat)
                    
            # Interleave quadrants across pages to form [batch_size, num_pages * 4, 256, D]
            stacked = torch.stack(slots_embeddings, dim=2) # [B, num_pages, 4, 256, D]
            stacked_slots = stacked.view(batch_size, num_pages * self.num_slots_per_page, -1, self.embedding_dim)
            
            return {
                "embeddings": stacked_slots,
                "metadata": []
            }
            
        # 3D Tensor processing (legacy / single sample mode)
        num_pages = page_embeddings.shape[0]
        device = page_embeddings.device
        
        num_patches = page_embeddings.shape[1]
        
        if num_patches >= 1024:
            core_patches = page_embeddings[:, :1024, :]  # [num_pages, 1024, D]
        else:
            core_patches = page_embeddings
            
        grid_dim = 32
        grid_patches = core_patches.view(num_pages, grid_dim, grid_dim, self.embedding_dim)
        chunk_span = grid_dim // self.grid_size
        
        slots_embeddings = []
        slots_metadata = []
        
        for p_idx in range(num_pages):
            for r_idx in range(self.grid_size):
                for c_idx in range(self.grid_size):
                    row_start = r_idx * chunk_span
                    row_end = (r_idx + 1) * chunk_span
                    col_start = c_idx * chunk_span
                    col_end = (c_idx + 1) * chunk_span
                    
                    chunk_patches = grid_patches[p_idx:p_idx+1, row_start:row_end, col_start:col_end, :]
                    chunk_patches_flat = chunk_patches.reshape(1, -1, self.embedding_dim)
                    
                    x1 = int((col_start / grid_dim) * 1000)
                    y1 = int((row_start / grid_dim) * 1000)
                    x2 = int((col_end / grid_dim) * 1000)
                    y2 = int((row_end / grid_dim) * 1000)
                    
                    slots_embeddings.append(chunk_patches_flat)
                    slots_metadata.append({
                        "page_id": p_idx,
                        "bbox": [x1, y1, x2, y2],
                        "name": f"page_{p_idx}_quadrant_{r_idx}_{c_idx}"
                    })
                    
        stacked_slots = torch.stack(slots_embeddings, dim=1).squeeze(0) # [total_slots, 256, D]
        stacked_slots = stacked_slots.unsqueeze(0) # [1, total_slots, 256, D]
        
        return {
            "embeddings": stacked_slots, # [1, total_slots, 256, D]
            "metadata": slots_metadata   # List of dicts describing spatial quadrants across all pages
        }
        
    def get_pooled_embeddings(self, stacked_slots):
        """
        Pools (averages) multi-vector patch embeddings inside each slot into a single vector.
        Args:
            stacked_slots: [1, total_slots, num_patches_per_chunk, D]
        Returns:
            pooled_slots: [1, total_slots, D]
        """
        return torch.mean(stacked_slots, dim=2)
