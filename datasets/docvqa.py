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
        """Loads the 20-sample real DocVQA subset from raw_pdfs/docvqa_subset."""
        data_dir = self.config.get("debug", {}).get("data_dir", "./debug_data/raw_pdfs/docvqa_subset")
        json_path = os.path.join(data_dir, "metadata.json")
        images_dir = os.path.join(data_dir, "images")
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"20-sample DocVQA subset metadata not found at: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        num_samples = self.config.get("debug", {}).get("num_samples", 20)
        for item in data[:num_samples]:
            img_path = os.path.join(images_dir, item["image"])
            self.samples.append({
                "question_id": hash(item["question"]),
                "image_path": img_path,
                "question": item["question"],
                "answers": [item["answer"]]
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
        
        # Load visual page image
        try:
            image = Image.open(sample["image_path"]).convert("RGB")
        except Exception as e:
            # Fallback dummy image if load fails
            print(f"[DocVQADataset] Warning: Failed to load {sample['image_path']}, using mock.")
            image = Image.new("RGB", (800, 1000), color="white")
            
        return {
            "image": image,
            "question": sample["question"],
            "answer": sample["answers"][0] if sample["answers"] else ""
        }

def collate_fn(batch):
    """
    Custom collate function to pack raw images and texts.
    ColPali processor will handle tokenization and padding of images/texts during batching.
    """
    images = [item["image"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    
    return {
        "images": images,
        "questions": questions,
        "answers": answers
    }
