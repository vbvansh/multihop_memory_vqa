import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class DocVQADataset(Dataset):
    """
    Dataset class for loading DocVQA data (images, questions, answers).
    Supports a mock debug mode with local files for quick overfitting and sanity checks.
    """
    def __init__(self, config, is_train=True, debug=False):
        self.config = config
        self.is_train = is_train
        self.debug = debug
        
        self.samples = []
        if self.debug:
            self._load_debug_samples()
        else:
            self._load_real_samples()
            
    def _load_debug_samples(self):
        """Loads the 20-sample real Multi-Page DocVQA subset from raw_pdfs/mpdocvqa_subset_multipage."""
        data_dir = self.config.get("debug", {}).get("data_dir", "./debug_data/raw_pdfs/mpdocvqa_subset_multipage")
        json_path = os.path.join(data_dir, "all_samples_metadata.json")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"20-sample Multi-Page DocVQA subset metadata not found at: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        num_samples = self.config.get("debug", {}).get("num_samples", 20)
        for item in data[:num_samples]:
            sample_id = item["sample_id"]
            img_paths = []
            for pg_name in item["pages"]:
                img_path = os.path.join(data_dir, f"sample_{sample_id}", "pages", pg_name)
                img_paths.append(img_path)
                
            # Safely evaluate string representation of list or use directly
            try:
                import ast
                answers_list = ast.literal_eval(item["answers"])
            except Exception:
                answers_list = [item["answers"]]
                
            self.samples.append({
                "question_id": item["question_id"],
                "image_paths": img_paths,
                "question": item["question"],
                "answers": answers_list,
                "answer_page_idx": int(item["answer_page_idx"])
            })
            
    def _load_real_samples(self):
        """Loads real DocVQA dataset from configured JSON files."""
        json_path = (
            self.config["paths"]["train_data_json"]
            if self.is_train
            else self.config["paths"]["val_data_json"]
        )
        images_dir = self.config["paths"]["images_dir"]
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Real DocVQA dataset annotations not found at: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        for item in data["questions"]:
            img_path = os.path.join(images_dir, item["image_local_name"])
            self.samples.append({
                "question_id": item["questionId"],
                "image_path": img_path,
                "question": item["question"],
                "answers": item["answers"]
            })

    def _create_mock_debug_data(self, data_dir, json_path):
        """Creates dummy image files and json annotation file for local verification."""
        print(f"[DocVQADataset] Creating mock debug dataset in {data_dir}...")
        
        # 1. Create a dummy white PNG image
        img = Image.new("RGB", (800, 1000), color="white")
        mock_img_names = ["page_0.png", "page_1.png", "page_2.png", "page_3.png"]
        for img_name in mock_img_names:
            img.save(os.path.join(data_dir, img_name))
            
        # 2. Create mock json annotations
        annotations = {
            "dataset_name": "docvqa_mock_debug",
            "questions": [
                {
                    "questionId": 10001,
                    "question": "What is the invoice number?",
                    "image_local_name": "page_0.png",
                    "answers": ["INV-2026-991", "2026-991"]
                },
                {
                    "questionId": 10002,
                    "question": "Who is the primary contact?",
                    "image_local_name": "page_1.png",
                    "answers": ["John Doe", "John"]
                },
                {
                    "questionId": 10003,
                    "question": "What is the total amount due?",
                    "image_local_name": "page_2.png",
                    "answers": ["$1,540.00", "1540"]
                },
                {
                    "questionId": 10004,
                    "question": "What date was this document printed?",
                    "image_local_name": "page_3.png",
                    "answers": ["21-05-2026", "May 21, 2026"]
                }
            ]
        }
        
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(annotations, f, indent=2)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        images = []
        # Multi-page debug mode
        if self.debug:
            for img_path in sample["image_paths"]:
                try:
                    img = Image.open(img_path).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"[DocVQADataset] Warning: Failed to load {img_path}, skipping.")
        else:
            # Single-page real mode
            try:
                img = Image.open(sample["image_path"]).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[DocVQADataset] Warning: Failed to load {sample['image_path']}, using fallback.")
                images.append(Image.new("RGB", (800, 1000), color="white"))
                
        # Fallback if list is empty
        if not images:
            images.append(Image.new("RGB", (800, 1000), color="white"))
            
        return {
            "images": images,
            "question": sample["question"],
            "answer": sample["answers"][0] if sample["answers"] else ""
        }

def collate_fn(batch):
    """
    Custom collate function to pack lists of images and texts.
    """
    # images is a list of lists: each element is a list of PIL Images (pages of a document)
    images = [item["images"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    return {
        "images": images,
        "questions": questions,
        "answers": answers
    }
