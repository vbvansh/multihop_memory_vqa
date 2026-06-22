import os
import sys

# Inject project root directory into sys.path to guarantee clean package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from trainers.trainer import ColPaliTrainer
from models.memory.memory_bank import MemoryBank
from models.reasoning.dual_stream import DualStreamProcessor
from models.memory.memory_router import MemoryRouter
from models.memory.gumbel_router import GumbelRouter
from models.reasoning.multi_hop import MultiHopReasoning
from models.decoder.answer_head import AnswerHead
from losses.sparsity_loss import SparsityLoss
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
    
    # 2. Load ColPali Backbone once (Quantized in 4-bit if enabled in config)
    model_name = model_config["model"]["name"]
    quantize = model_config["model"].get("quantize_4bit", False)
    from transformers import ColPaliForRetrieval
    
    if quantize and torch.cuda.is_available():
        logger.info(f"Loading pre-trained ColPali backbone {model_name} in 4-bit quantization...")
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            llm_int8_skip_modules=["embedding_proj_layer"]
        )
        shared_model = ColPaliForRetrieval.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            low_cpu_mem_usage=True
        )
    else:
        logger.info(f"Loading pre-trained ColPali backbone {model_name} in standard {dtype}...")
        shared_model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        
    # Freeze all parameters of the shared ColPali backbone.
    # This prevents gradient tracking, saves memory, and fixes bitsandbytes precision casting bugs!
    for param in shared_model.parameters():
        param.requires_grad = False
    
    # Merge configurations
    config = {**model_config, **train_config}
    
    # 3. Instantiate Encoders, Memory Bank, Routers, Multi-Hop Reasoning and Answer Head
    from models.encoders.vision_encoder import ColPaliVisionEncoder
    from models.encoders.question_encoder import ColPaliQuestionEncoder
    
    vision_encoder = ColPaliVisionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    question_encoder = ColPaliQuestionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    
    tokenizer = vision_encoder.processor.tokenizer
    vocab_size = tokenizer.vocab_size
    logger.info(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    memory_bank = MemoryBank(config)
    dual_stream = DualStreamProcessor(config).to(device).to(dtype)
    router = MemoryRouter(config).to(device).to(dtype)
    gumbel = GumbelRouter(temperature=0.5).to(device)
    multi_hop = MultiHopReasoning(config).to(device).to(dtype)
    answer_head = AnswerHead(config, vocab_size=vocab_size).to(device).to(dtype)
    sparsity_loss_fn = SparsityLoss(config, ignore_index=tokenizer.pad_token_id)
    
    # 4. Load the 20-sample Multi-Page DocVQA subset
    logger.info("Loading 20-sample real Multi-Page DocVQA subset...")
    dataset = DocVQADataset(config, is_train=True, debug=True)
    logger.info(f"Dataset successfully loaded. Total samples: {len(dataset)}")
    
    # 5. Run Verification on the multi-page sample
    sample_idx = 162  # Sample 2: "What is the name of the company?" (Ground Truth: "ITC Limited", located on Page 11 of 12)
    sample = dataset[sample_idx]
    
    logger.info(f"\n--- Running Verification on Multi-Page Sample {sample_idx} ---")
    logger.info(f"Question: '{sample['question']}'")
    logger.info(f"Ground Truth Answer: '{sample['answer']}'")
    logger.info(f"Number of Pages in Document: {len(sample['images'])}")
    
    # Step A: Vision Embedding Generation (Phase 1)
    logger.info("\n[Phase 1] Generating multi-page visual embeddings sequentially to save VRAM...")
    page_embs = []
    with torch.no_grad():
        for p_idx, img in enumerate(sample["images"]):
            # Encode each page image individually to stay safely under 4GB VRAM limit
            emb = vision_encoder([img])  # Shape [1, 1030, D]
            page_embs.append(emb.cpu())  # Temporarily store on CPU to save active VRAM
            
    # Combine page embeddings along batch dimension (num_pages)
    page_emb = torch.cat(page_embs, dim=0).to(device)
    logger.info(f"-> Generated Multi-Page Embedding Shape: {page_emb.shape}") # [N_pages, 1030, D]
    
    # Step B: Chunk Partitioning & Memory Bank Formation (Phase 2)
    logger.info("\n[Phase 2] Partitioning all pages into quadrant visual memory slots...")
    memory_outputs = memory_bank(page_emb)
    chunk_embeddings = memory_outputs["embeddings"] # [1, total_slots, 256, D]
    chunk_metadata = memory_outputs["metadata"]
    
    logger.info(f"-> Created Memory Bank of Chunks. Stacked Shape: {chunk_embeddings.shape}")
    logger.info(f"-> Total Visual Memory Slots: {len(chunk_metadata)} ({len(sample['images'])} pages x 4 quadrants)")
        
    # Step C: Question Embedding (Phase 3)
    logger.info("\n[Phase 3] Encoding question text...")
    with torch.no_grad():
        query_emb = question_encoder([sample["question"]])
    logger.info(f"-> Generated Question Embedding Shape: {query_emb.shape}") # [1, num_query_tokens, D]
    
    # Step D: Decoupled Dual-Stream Processing (Phase 4)
    logger.info("\n[Phase 4] Running Decoupled Dual-Stream Pre-Fusion Processing...")
    with torch.no_grad():
        c_q, m_tilde, contextualized_patches = dual_stream(
            query_emb.to(device).to(dtype),
            chunk_embeddings.to(device).to(dtype)
        )
    logger.info(f"-> Stream A Context Vector c_q Shape: {c_q.shape}")
    logger.info(f"-> Stream B Contextualized Slots m_tilde Shape: {m_tilde.shape}")
    logger.info(f"-> Stream B Contextualized Patches Shape: {contextualized_patches.shape}")
    
    # Step D.2: Policy Router Logits (Phase 5)
    logger.info("\n[Phase 5] Running Question-Guided Policy Router MLP...")
    # Calculate 2D activation logits per memory slot: [1, total_slots, 2]
    with torch.no_grad():
        logits = router(c_q, m_tilde)
    logger.info(f"-> Generated Policy Logits Shape: {logits.shape}")
    
    # Step E: Gumbel-Softmax Gating (Phase 4 / 6)
    logger.info("\n[Phase 4/6] Running Differentiable Gumbel-Softmax Binary Router Gating...")
    with torch.no_grad():
        # z contains strict 0.0 or 1.0 decisions: [1, total_slots]
        z = gumbel(logits, hard=True)
    logger.info(f"-> Generated Gumbel Gates Shape: {z.shape}")
    
    # Calculate the Active Memory Ratio (percentage of chunks activated)
    active_slots_count = torch.sum(z[0]).item()
    active_memory_ratio = (active_slots_count / len(chunk_metadata)) * 100
    
    # Get active and frozen slots lists
    active_slots = []
    for idx, gate in enumerate(z[0]):
        if gate.item() == 1.0:
            active_slots.append(idx)
            
    # Calculate continuous probabilities for logs (via standard softmax over logits)
    probs = F.softmax(logits, dim=-1) # [1, total_slots, 2]
    
    # Combine everything for logging
    slot_details = []
    for idx in range(chunk_embeddings.size(1)):
        p_act = probs[0, idx, 1].item()
        logit_act = logits[0, idx, 1].item()
        gate_val = z[0, idx].item()
        slot_details.append((idx, logit_act, p_act, gate_val, chunk_metadata[idx]))
        
    # Sort slots by raw activation logit value descending (what the router likes most)
    slot_details.sort(key=lambda x: x[1], reverse=True)
    
    logger.info("\n--- Top 5 Slots Preferred by the Policy Router ---")
    for rank, (idx, logit, prob, gate, meta) in enumerate(slot_details[:5]):
        logger.info(
            f"Rank {rank+1}: Slot {idx} | Name: '{meta['name']}' | Page index: {meta['page_id']} | "
            f"Act Logit: {logit:.4f} | Prob: {prob*100:.2f}% | Gumbel Gate: {int(gate)} | BBox: {meta['bbox']}"
        )
        
    best_idx, best_logit, best_prob, best_gate, best_meta = slot_details[0]
    
    logger.info(f"\n==================================================")
    logger.info(f"DENSE POLICY GATING RESULTS:")
    logger.info(f"Best matching visual chunk: Slot {best_idx} ({best_meta['name']})")
    logger.info(f"Page ID (0-indexed): {best_meta['page_id']} (Page {best_meta['page_id'] + 1} of the document)")
    logger.info(f"Activation Preference Logit: {best_logit:.4f}")
    logger.info(f"Softmax Activation Probability: {best_prob*100:.2f}%")
    logger.info(f"Gumbel Gating Binary Output (z): {int(best_gate)}")
    logger.info(f"\n==================================================")
    logger.info(f"SPARSITY AUDIT SUMMARY:")
    logger.info(f"Total slots in memory bank: {len(chunk_metadata)}")
    logger.info(f"Activated chunks (g_k = 1): {int(active_slots_count)}")
    logger.info(f"Frozen chunks masked out (g_k = 0): {len(chunk_metadata) - int(active_slots_count)}")
    logger.info(f"Active Memory Ratio: {active_memory_ratio:.2f}%")
    logger.info(f"==================================================")
    
    # Step F: Multi-Hop Reasoning Cell (Phase 7)
    logger.info("\n[Phase 7] Running Multi-Hop Reasoning Cell over gated memory slots...")
    with torch.no_grad():
        updated_query_emb, key_padding_mask = multi_hop(
            query_emb.to(device).to(dtype),
            contextualized_patches.to(device).to(dtype),
            z.to(device).to(dtype)
        )
    logger.info(f"-> Updated Query Shape: {updated_query_emb.shape} (Reasoning state injected)")
    logger.info(f"-> Active/Inactive Mask Shape: {key_padding_mask.shape}")
    
    # Step G: Gated Answer Head (Phase 7)
    logger.info("\n[Phase 7] Fusing updated query and memory patches through Gated Answer Head...")
    # Flatten memory slots: [batch_size, total_slots * 256, D]
    flat_memory = contextualized_patches.reshape(1, -1, config["model"]["embedding_dim"]).to(device).to(dtype)
    with torch.no_grad():
        pred_logits = answer_head(
            updated_query_emb.to(dtype),
            flat_memory,
            key_padding_mask=key_padding_mask
        )
    logger.info(f"-> Predicted Answer Logits Shape: {pred_logits.shape} [batch, max_answer_len, vocab_size]")
    
    # Decode predicted token IDs for logging
    pred_token_ids = torch.argmax(pred_logits, dim=-1) # [1, max_answer_len]
    
    # Clean Decoding Logic: truncate sequence at first <eos> or <pad> token
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    pred_seq_list = pred_token_ids[0].tolist()
    indices = [idx_t for idx_t, token in enumerate(pred_seq_list) if token in (eos_id, pad_id)]
    if indices:
        first_idx = indices[0]
        truncated_seq = pred_seq_list[:first_idx]
    else:
        truncated_seq = pred_seq_list
        
    decoded_answer = tokenizer.decode(truncated_seq, skip_special_tokens=True).strip()
    logger.info(f"-> Decoded Gated VQA Prediction (Untrained) with Clean Decoding: '{decoded_answer}'")
    
    # Step H: Sparsity & Entropy Regularization Loss Calculation (Phase 8)
    logger.info("\n[Phase 8] Computing Sparsity and Entropy Regularization Loss...")
    targets = tokenizer(
        [sample["answer"]],
        padding="max_length",
        max_length=config["decoder"]["max_answer_len"],
        truncation=True,
        return_tensors="pt"
    )["input_ids"].to(device)
    
    # Flatten logits and targets to compute cross entropy
    flat_logits = pred_logits.view(-1, pred_logits.size(-1))
    flat_targets = targets.view(-1)
    
    with torch.no_grad():
        total_loss, loss_details = sparsity_loss_fn(
            flat_logits,
            flat_targets,
            logits.to(device),
            z.to(device)
        )
        
    logger.info(f"==================================================")
    logger.info(f"COMPOSITE LOSS AUDIT:")
    logger.info(f"-> Total Loss: {loss_details['loss_total']:.6f}")
    logger.info(f"-> VQA Cross-Entropy Loss: {loss_details['loss_vqa']:.6f}")
    logger.info(f"-> Sparsity Penalty (lambda={config['training']['lambda_sparsity']}): {loss_details['loss_sparsity']:.6f}")
    logger.info(f"-> Router Entropy Penalty (lambda={config['training']['lambda_entropy']}): {loss_details['loss_entropy']:.6f}")
    logger.info(f"==================================================")
    logger.info("Pipeline verified successfully through Gated Answer Prediction & Sparsity Loss!")

if __name__ == "__main__":
    main()
