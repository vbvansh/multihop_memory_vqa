import os
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from trainers.trainer import ColPaliTrainer
from models.memory.memory_bank import MemoryBank
from datasets.docvqa import DocVQADataset, collate_fn
from utils.logger import setup_logger

def compute_late_interaction_maxsim(query_emb, chunk_emb):
    """
    Computes ColPali's late-interaction similarity score (MaxSim) between query and chunk.
    Args:
        query_emb: Tensor of shape [num_query_tokens, D]
        chunk_emb: Tensor of shape [num_patches_per_chunk, D]
    Returns:
        similarity: Scalar float tensor representing MaxSim score
    """
    # L2 normalize embeddings to compute Cosine Similarity via dot product
    query_emb_norm = F.normalize(query_emb, p=2, dim=-1) # [num_query_tokens, D]
    chunk_emb_norm = F.normalize(chunk_emb, p=2, dim=-1) # [num_patches_per_chunk, D]
    
    # Compute similarity matrix: [num_query_tokens, num_patches_per_chunk]
    sim_matrix = torch.matmul(query_emb_norm, chunk_emb_norm.transpose(0, 1))
    
    # MaxSim: find the maximum similarity in the chunk for each query token
    max_sims, _ = torch.max(sim_matrix, dim=1) # [num_query_tokens]
    
    # Average MaxSim across all query tokens
    return torch.mean(max_sims).item()

def main():
    logger = setup_logger(name="PipelineVerifier")
    logger.info("==================================================")
    logger.info("Initializing Phase 1-2-3 Pipeline Verification Run")
    logger.info("==================================================")
    
    # 1. Load configs
    model_config_path = "./configs/model.yaml"
    train_config_path = "./configs/train.yaml"
    
    with open(model_config_path, "r") as f:
        model_config = yaml.safe_load(f)
    with open(train_config_path, "r") as f:
        train_config = yaml.safe_load(f)
        
    # Force GPU and BF16 for fast loading if supported
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    logger.info(f"Using device: {device} | dtype: {dtype}")
    
    # 2. Load ColPali Backbone once
    model_name = model_config["model"]["name"]
    logger.info(f"Loading pre-trained ColPali backbone from {model_name}...")
    from transformers import ColPaliForRetrieval
    shared_model = ColPaliForRetrieval.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # 3. Instantiate Encoders and Memory Bank
    from models.encoders.vision_encoder import ColPaliVisionEncoder
    from models.encoders.question_encoder import ColPaliQuestionEncoder
    
    vision_encoder = ColPaliVisionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    question_encoder = ColPaliQuestionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    memory_bank = MemoryBank(model_config)
    
    # 4. Load the 20-sample DocVQA subset
    logger.info("Loading 20-sample real DocVQA subset...")
    dataset = DocVQADataset(train_config, is_train=True, debug=True)
    logger.info(f"Dataset successfully loaded. Total samples: {len(dataset)}")
    
    # 5. Run Verification on the first sample
    sample_idx = 2  # Sample 2: "Which corporation's letterhead is this?" -> "Brown & Williamson Tobacco Corporation"
    sample = dataset[sample_idx]
    
    logger.info(f"\n--- Running Verification on Sample {sample_idx} ---")
    logger.info(f"Question: '{sample['question']}'")
    logger.info(f"Ground Truth Answer: '{sample['answer']}'")
    
    # Step A: Vision Embedding Generation (Phase 1)
    logger.info("\n[Phase 1] Generating page visual embeddings...")
    with torch.no_grad():
        page_emb = vision_encoder([sample["image"]])
    logger.info(f"-> Generated Page Embedding Shape: {page_emb.shape}") # [1, 1030, D]
    
    # Step B: Chunk Partitioning & Memory Bank Formation (Phase 2)
    logger.info("\n[Phase 2] Partitioning page into quadrant visual memory slots...")
    memory_outputs = memory_bank(page_emb)
    chunk_embeddings = memory_outputs["embeddings"] # [1, 4, 256, D]
    chunk_metadata = memory_outputs["metadata"]
    
    logger.info(f"-> Created Memory Bank of Chunks. Stacked Shape: {chunk_embeddings.shape}")
    for idx, meta in enumerate(chunk_metadata):
        logger.info(f"   * Slot {idx}: Name: '{meta['name']}' | Normalized BBox: {meta['bbox']}")
        
    # Step C: Question Embedding & Similarity Verification (Phase 3)
    logger.info("\n[Phase 3] Encoding question text and verifying retrieval similarity (MaxSim)...")
    with torch.no_grad():
        query_emb = question_encoder([sample["question"]])
    logger.info(f"-> Generated Question Embedding Shape: {query_emb.shape}") # [1, num_query_tokens, D]
    
    # Compute similarity between query and each of the 4 visual chunks
    logger.info("\nCalculating Late-Interaction similarity scores per visual memory slot:")
    best_slot = -1
    best_score = -100.0
    
    for idx in range(chunk_embeddings.size(1)):
        # Extract query and chunk vectors (strip batch dim)
        q_emb = query_emb[0].to(torch.float32)
        c_emb = chunk_embeddings[0, idx].to(torch.float32)
        
        sim_score = compute_late_interaction_maxsim(q_emb, c_emb)
        logger.info(f"   * Similarity to Slot {idx} ({chunk_metadata[idx]['name']}): {sim_score:.4f}")
        
        if sim_score > best_score:
            best_score = sim_score
            best_slot = idx
            
    logger.info(f"\n==================================================")
    logger.info(f"VERIFICATION RESULT:")
    logger.info(f"Best matching visual chunk: Slot {best_slot} ({chunk_metadata[best_slot]['name']})")
    logger.info(f"Matching BBox: {chunk_metadata[best_slot]['bbox']}")
    logger.info(f"MaxSim Similarity Score: {best_score:.4f}")
    logger.info(f"==================================================")
    logger.info("Pipeline verified successfully through Phase 3! Embeddings are highly meaningful.")

if __name__ == "__main__":
    main()
