import os
import sys
import torch
import yaml
import argparse
from PIL import Image

# Inject project root directory into sys.path to guarantee clean package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder
from models.memory.memory_bank import MemoryBank
from models.reasoning.dual_stream import DualStreamProcessor
from models.memory.memory_router import MemoryRouter
from models.memory.gumbel_router import GumbelRouter
from models.reasoning.multi_hop import MultiHopReasoning
from models.decoder.answer_head import AnswerHead
from datasets.docvqa import DocVQADataset
from utils.metrics import calculate_anls, calculate_exact_match

def parse_args():
    parser = argparse.ArgumentParser(description="Run Question Ablation Test on a specific document")
    parser.add_argument(
        "--model_config", 
        type=str, 
        default="./configs/model.yaml", 
        help="Path to model config file"
    )
    parser.add_argument(
        "--train_config", 
        type=str, 
        default="./configs/train.yaml", 
        help="Path to training config file"
    )
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        default="./checkpoints/checkpoint_last.pt", 
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--doc_id", 
        type=str, 
        default="jzbn0226", 
        help="Document ID/substring to filter by"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="train", 
        choices=["train", "val", "test"], 
        help="Dataset split to search"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Load Configurations
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    with open(args.train_config, "r") as f:
        train_config = yaml.safe_load(f)
        
    config = {**model_config, **train_config}
    
    # Bypass limits in dataset loading to search the entire split
    config["training"]["max_train_samples"] = None
    config["training"]["max_val_samples"] = None
    
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    dtype_str = config["model"]["dtype"]
    dtype = torch.bfloat16 if dtype_str == "bfloat16" else (torch.float16 if dtype_str == "float16" else torch.float32)
    
    print(f"Device: {device} | Dtype: {dtype}")
    
    # 2. Initialize Tokenizer & Processor
    model_name = config["model"]["name"]
    use_precomputed = config["debug"].get("use_precomputed_embeddings", False)
    
    if use_precomputed:
        print("Using PRECOMPUTED embeddings mode.")
        from transformers import ColPaliProcessor
        processor = ColPaliProcessor.from_pretrained(model_name)
        tokenizer = processor.tokenizer
        vision_encoder = None
        question_encoder = None
    else:
        # Standard backbone loading
        from transformers import ColPaliForRetrieval
        quantize = config["model"].get("quantize_4bit", False)
        if quantize and torch.cuda.is_available():
            print(f"Loading shared ColPali backbone {model_name} in 4-bit quantization...")
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
            print(f"Loading shared ColPali backbone {model_name} in standard {dtype}...")
            shared_model = ColPaliForRetrieval.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=config["model"]["device"],
                low_cpu_mem_usage=True
            )
        for param in shared_model.parameters():
            param.requires_grad = False
            
        vision_encoder = ColPaliVisionEncoder(
            model_name=model_name,
            device=device,
            dtype=dtype,
            shared_model=shared_model
        )
        question_encoder = ColPaliQuestionEncoder(
            model_name=model_name,
            device=device,
            dtype=dtype,
            shared_model=shared_model
        )
        tokenizer = vision_encoder.processor.tokenizer
        
    vocab_size = tokenizer.vocab_size
    print(f"Loaded tokenizer with vocabulary size: {vocab_size}")
    
    # 3. Instantiate model components
    memory_bank = MemoryBank(config)
    dual_stream = DualStreamProcessor(config).to(device).to(dtype)
    router = MemoryRouter(config).to(device).to(dtype)
    gumbel = GumbelRouter(temperature=0.5).to(device)
    multi_hop = MultiHopReasoning(config).to(device).to(dtype)
    answer_head = AnswerHead(config, vocab_size=vocab_size).to(device).to(dtype)
    
    # 4. Load Checkpoint
    checkpoint_path = args.checkpoint
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint weights from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        dual_stream.load_state_dict(checkpoint["dual_stream_state"])
        router.load_state_dict(checkpoint["router_state"])
        multi_hop.load_state_dict(checkpoint["multi_hop_state"])
        answer_head.load_state_dict(checkpoint["answer_head_state"])
        print(f"Checkpoint successfully loaded. Trained Epoch: {checkpoint.get('epoch', 'N/A')}")
    else:
        print(f"WARNING: Checkpoint path '{checkpoint_path}' does not exist! Running with random weights.")
        
    dual_stream.eval()
    router.eval()
    multi_hop.eval()
    answer_head.eval()
    
    # 5. Load and Filter Dataset
    print(f"Loading split '{args.split}'...")
    dataset = DocVQADataset(config, split=args.split, debug=False)
    
    # Filter dataset samples for the specified doc_id
    filtered_samples = []
    for sample in dataset.samples:
        is_match = False
        for path in sample.get("image_paths", []):
            if args.doc_id.lower() in os.path.basename(path).lower():
                is_match = True
                break
        if args.doc_id.lower() in str(sample.get("question_id", "")).lower():
            is_match = True
            
        if is_match:
            filtered_samples.append(sample)
            
    print(f"Found {len(filtered_samples)} questions matching Document ID '{args.doc_id}' in split '{args.split}'.")
    if len(filtered_samples) == 0:
        print("No samples found. Please double check the doc_id and split.")
        return
        
    # 6. Run Inference and Print Results
    precomputed_dir = config["debug"]["precomputed_dir"]
    predictions = []
    ground_truths = []
    
    print("\n" + "="*80)
    print(f"QUESTION ABLATION REPORT FOR DOCUMENT: {args.doc_id}")
    print("="*80)
    
    with torch.no_grad():
        for idx, sample in enumerate(filtered_samples):
            question_id = sample["question_id"]
            question_str = sample["question"]
            answers = sample["answers"]
            
            # Load or encode visual page representations
            if use_precomputed:
                emb_path = os.path.join(precomputed_dir, f"vision_{question_id}.pt")
                if not os.path.exists(emb_path):
                    print(f"Error: Embedding file {emb_path} does not exist. Skipping sample.")
                    continue
                page_emb = torch.load(emb_path, map_location=device).to(dtype)
            else:
                images = []
                for img_p in sample["image_paths"]:
                    if os.path.exists(img_p):
                        images.append(Image.open(img_p).convert("RGB"))
                if len(images) == 0:
                    print(f"Error: No images found for sample {question_id}. Skipping.")
                    continue
                page_embs = []
                for img in images:
                    emb = vision_encoder([img])
                    page_embs.append(emb.cpu())
                page_emb = torch.cat(page_embs, dim=0).to(device).to(dtype)
                
            # Partition into memory slots
            memory_outputs = memory_bank(page_emb)
            chunk_embeddings = memory_outputs["embeddings"].to(device).to(dtype)
            
            # Load or encode question
            if use_precomputed:
                q_emb_path = os.path.join(precomputed_dir, f"question_{question_id}.pt")
                if not os.path.exists(q_emb_path):
                    print(f"Error: Question embedding file {q_emb_path} does not exist. Skipping sample.")
                    continue
                query_emb = torch.load(q_emb_path, map_location=device).to(dtype)
            else:
                query_emb = question_encoder([question_str]).to(device).to(dtype)
                
            # Dual-Stream
            c_q, m_tilde, contextualized_patches = dual_stream(query_emb, chunk_embeddings)
            
            # Routing
            router_logits = router(c_q, m_tilde)
            z = gumbel(router_logits, hard=True)
            
            # Multi-Hop
            updated_query, key_padding_mask = multi_hop(query_emb, contextualized_patches, z)
            
            # Answer Head
            flat_memory = contextualized_patches.reshape(1, -1, config["model"]["embedding_dim"])
            pred_logits = answer_head(updated_query, flat_memory, key_padding_mask=key_padding_mask)
            
            # Decode
            pred_ids = torch.argmax(pred_logits, dim=-1) # [1, max_answer_len]
            
            eos_id = tokenizer.eos_token_id
            pad_id = tokenizer.pad_token_id
            
            seq_list = pred_ids[0].tolist()
            indices = [idx_t for idx_t, token in enumerate(seq_list) if token in (eos_id, pad_id)]
            if indices:
                first_idx = indices[0]
                truncated_seq = seq_list[:first_idx]
            else:
                truncated_seq = seq_list
                
            pred_str = tokenizer.decode(truncated_seq, skip_special_tokens=True).strip()
            
            # Calculate metrics for this sample
            anls = calculate_anls(pred_str, answers)
            em = calculate_exact_match(pred_str, answers)
            
            predictions.append(pred_str)
            ground_truths.append(answers)
            
            print(f"\n[Sample {idx+1}/{len(filtered_samples)}] QID: {question_id}")
            print(f"Q: '{question_str}'")
            print(f"Pred: '{pred_str}'")
            print(f"GT: {answers}")
            print(f"ANLS: {anls:.4f} | EM: {em:.4f}")
            
    # Calculate global metrics
    total_anls = 0.0
    total_em = 0.0
    for pred, gts in zip(predictions, ground_truths):
        total_anls += calculate_anls(pred, gts)
        total_em += calculate_exact_match(pred, gts)
        
    avg_anls = total_anls / len(predictions) if predictions else 0.0
    avg_em = total_em / len(predictions) if predictions else 0.0
    
    print("\n" + "="*80)
    print(f"OVERALL SUMMARY FOR DOCUMENT {args.doc_id}")
    print(f"Total Questions Evaluated: {len(predictions)}")
    print(f"Mean ANLS: {avg_anls:.4f}")
    print(f"Mean Exact Match: {avg_em:.4f}")
    print("="*80)

if __name__ == "__main__":
    main()
