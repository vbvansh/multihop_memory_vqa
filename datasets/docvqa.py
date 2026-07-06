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
    def __init__(self, config, split="train", is_train=None, debug=False):
        self.config = config
        self.debug = debug
        
        # Backward compatibility for is_train Boolean flag
        if is_train is not None:
            self.split = "train" if is_train else "val"
        else:
            self.split = split
            
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
        """Loads real Multi-Page DocVQA dataset from configured JSON files based on the split name."""
        if self.split == "train":
            json_path = self.config["paths"]["train_data_json"]
        elif self.split == "val":
            json_path = self.config["paths"]["val_data_json"]
        elif self.split == "test":
            json_path = self.config["paths"]["test_data_json"]
        else:
            raise ValueError(f"Unknown dataset split: {self.split}")
            
        images_dir = self.config["paths"]["images_dir"]
        
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Real MPDocVQA dataset annotations not found for split '{self.split}' at: {json_path}")
            
        with open(json_path, "r", encoding="utf-8") as f:
            data_dict = json.load(f)
            
        # Parse data list from root dictionary
        data_items = data_dict.get("data", [])
        if self.split == "val":
            max_val = self.config.get("training", {}).get("max_val_samples", None)
            if max_val is not None:
                data_items = data_items[:max_val]
        elif self.split == "train":
            max_train = self.config.get("training", {}).get("max_train_samples", None)
            if max_train is not None:
                data_items = data_items[:max_train]
                
        for item in data_items:
            # Convert list of page ids to image file paths (adding .jpg extension)
            img_paths = [os.path.join(images_dir, f"{p_id}.jpg") for p_id in item["page_ids"]]
            self.samples.append({
                "question_id": item["questionId"],
                "image_paths": img_paths,
                "question": item["question"],
                "answers": item.get("answers", [""]),
                "answer_page_idx": int(item.get("answer_page_idx", 0))
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
        # Support loading page image paths for both debug and real mode
        img_paths = sample.get("image_paths", [])
        if not img_paths and "image_path" in sample:
            img_paths = [sample["image_path"]]
            
        for img_path in img_paths:
            try:
                img = Image.open(img_path).convert("RGB")
                images.append(img)
            except Exception as e:
                print(f"[DocVQADataset] Warning: Failed to load {img_path}. Error: {e}. Skipping.")
                
        # Fallback if list is empty
        if not images:
            images.append(Image.new("RGB", (800, 1000), color="white"))

        answers_all = sample["answers"] if sample.get("answers") else [""]
        return {
            "question_id": sample.get("question_id", idx),
            "images": images,
            "image_paths": img_paths,                       # page image paths (reader loads the routed page)
            "question": sample["question"],
            "answer": answers_all[0],                        # primary answer (target / teacher forcing)
            "answers_all": answers_all,                      # full GT list (multi-answer ANLS/EM)
            "answer_page_idx": int(sample.get("answer_page_idx", 0))  # page-level supervision for the router
        }

def collate_fn(batch):
    """
    Custom collate function to pack lists of images and texts.
    """
    # images is a list of lists: each element is a list of PIL Images (pages of a document)
    images = [item["images"] for item in batch]
    image_paths = [item["image_paths"] for item in batch]
    questions = [item["question"] for item in batch]
    answers = [item["answer"] for item in batch]
    answers_all = [item["answers_all"] for item in batch]
    answer_page_idxs = [item["answer_page_idx"] for item in batch]
    question_ids = [item["question_id"] for item in batch]

    return {
        "images": images,
        "image_paths": image_paths,
        "questions": questions,
        "answers": answers,              # primary answer per sample (teacher forcing target)
        "answers_all": answers_all,      # full GT list per sample (metrics)
        "answer_page_idxs": answer_page_idxs,
        "question_ids": question_ids
    }
