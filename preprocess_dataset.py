import os
import sys
import torch
import yaml
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

# Inject project root directory into sys.path to guarantee clean package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder
from datasets.docvqa import DocVQADataset, collate_fn

def main():
    parser = argparse.ArgumentParser(description="Precompute ColPali embeddings in parallel batch mode.")
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8, 
        help="Batch size for parallel vision and question encoding (set 32 or 64 on A5000)"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="all", 
        choices=["train", "val", "test", "all"], 
        help="Which split of the dataset to precompute"
    )
    parser.add_argument(
        "--debug", 
        action="store_true", 
        help="Run in debug mode (uses local subset samples instead of full dataset)"
    )
    args = parser.parse_args()

    print("==================================================")
    print("Parallel Batch Precomputation of ColPali Embeddings")
    print(f"Batch Size: {args.batch_size} | Split Target: {args.split}")
    print("==================================================")
    
    # 1. Load Configurations
    with open("./configs/model.yaml", "r") as f:
        model_config = yaml.safe_load(f)
    with open("./configs/train.yaml", "r") as f:
        train_config = yaml.safe_load(f)
        
    config = {**model_config, **train_config}
    
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config["model"]["dtype"] == "bfloat16" and torch.cuda.is_available() else torch.float32
    
    precomputed_dir = config["debug"]["precomputed_dir"]
    os.makedirs(precomputed_dir, exist_ok=True)
    
    # 2. Load ColPali Backbone (Quantized in 4-bit if enabled in config)
    model_name = config["model"]["name"]
    quantize = config["model"].get("quantize_4bit", False)
    from transformers import ColPaliForRetrieval
    
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
        print(f"Loading shared ColPali backbone from {model_name} in standard {dtype}...")
        shared_model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        
    for param in shared_model.parameters():
        param.requires_grad = False
        
    # 3. Instantiate Encoders
    vision_encoder = ColPaliVisionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    question_encoder = ColPaliQuestionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    
    # Determine splits to process
    splits_to_process = ["train", "val", "test"] if args.split == "all" else [args.split]
    
    for split in splits_to_process:
        print(f"\n--- Processing Split: {split} ---")
        
        # 4. Load Dataset
        dataset = DocVQADataset(config, split=split, debug=args.debug)
        original_count = len(dataset)
        print(f"Loaded {original_count} samples for split '{split}'")
        
        # Filter out samples that have already been precomputed to allow fast resumption
        filtered_samples = []
        for sample in dataset.samples:
            q_id = sample["question_id"]
            v_path = os.path.join(precomputed_dir, f"vision_{q_id}.pt")
            q_path = os.path.join(precomputed_dir, f"question_{q_id}.pt")
            if not (os.path.exists(v_path) and os.path.exists(q_path)):
                filtered_samples.append(sample)
        dataset.samples = filtered_samples
        
        print(f"Resuming precomputation. Skipped {original_count - len(dataset)} already processed samples.")
        print(f"Remaining samples to process: {len(dataset)}")
        
        if len(dataset) == 0:
            print(f"Split '{split}' is already fully precomputed!")
            continue
            
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            drop_last=False
        )
        
        # 5. Extract and save embeddings in batches
        print("Extracting embeddings...")
        for batch in tqdm(loader, desc=f"Precomputing {split}"):
            # Flatten page images from all documents in the batch
            # doc_lengths keeps track of how many pages belong to each document
            all_images = [img for doc_imgs in batch["images"] for img in doc_imgs]
            doc_lengths = [len(doc_imgs) for doc_imgs in batch["images"]]
            
            # Skip if batch is somehow empty
            if not all_images:
                continue
                
            # Run parallel vision encoding over the flat list of images
            with torch.no_grad():
                # all_embs shape: [total_pages, num_patches, D]
                all_embs = vision_encoder(all_images)
                
            # Run parallel question encoding
            with torch.no_grad():
                # query_embs shape: [batch_size, num_query_tokens, D]
                query_embs = question_encoder(batch["questions"])
                
            # Split and save embeddings back to their respective document questionIds
            curr_idx = 0
            for j, length in enumerate(doc_lengths):
                q_id = batch["question_ids"][j]
                doc_emb = all_embs[curr_idx : curr_idx + length].cpu()
                curr_idx += length
                
                # Save visual embeddings
                torch.save(doc_emb, os.path.join(precomputed_dir, f"vision_{q_id}.pt"))
                
            # Save query embeddings
            for j, q_id in enumerate(batch["question_ids"]):
                q_emb = query_embs[j : j + 1].cpu()
                torch.save(q_emb, os.path.join(precomputed_dir, f"question_{q_id}.pt"))
                
            # Clean up memory to prevent CPU/GPU memory leaks
            for img in all_images:
                img.close()
            del all_images
            del all_embs
            del query_embs
            del batch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    print(f"\nAll requested splits successfully precomputed and saved in {precomputed_dir}!")

if __name__ == "__main__":
    main()
