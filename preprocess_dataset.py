import os
import sys
import torch
import yaml
from PIL import Image
from tqdm import tqdm

# Inject project root directory into sys.path to guarantee clean package imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder
from datasets.docvqa import DocVQADataset

def main():
    print("==================================================")
    print("Precomputing Dataset Embeddings (The 24m -> 2s Hack)")
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
    
    # 2. Load ColPali Backbone once (Quantized in 4-bit if enabled in config)
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
            device_map="auto"
        )
    else:
        print(f"Loading shared ColPali backbone from {model_name} in standard {dtype}...")
        shared_model = ColPaliForRetrieval.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu"
        )
        
    for param in shared_model.parameters():
        param.requires_grad = False
        
    # 3. Instantiate Encoders
    vision_encoder = ColPaliVisionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    question_encoder = ColPaliQuestionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    
    # 4. Load Dataset in debug mode
    print("Loading 20-sample real Multi-Page DocVQA subset...")
    dataset = DocVQADataset(config, is_train=True, debug=True)
    
    # 5. Extract and save embeddings
    print("Starting precomputation...")
    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        q_id = sample["question_id"]
        images = sample["images"]
        question = sample["question"]
        
        # Sequentially encode page images to prevent VRAM spikes
        page_embs = []
        with torch.no_grad():
            for img in images:
                emb = vision_encoder([img])  # [1, num_patches, D]
                page_embs.append(emb.cpu())
        page_emb = torch.cat(page_embs, dim=0) # [num_pages, num_patches, D]
        
        # Encode question text
        with torch.no_grad():
            query_emb = question_encoder([question]).cpu() # [1, num_query_tokens, D]
            
        # Save to disk
        torch.save(page_emb, os.path.join(precomputed_dir, f"vision_{q_id}.pt"))
        torch.save(query_emb, os.path.join(precomputed_dir, f"question_{q_id}.pt"))
        
    print(f"\nSuccessfully precomputed and saved embeddings for {len(dataset)} samples in {precomputed_dir}!")

if __name__ == "__main__":
    main()
