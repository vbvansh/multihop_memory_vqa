"""
Precompute ColPali embeddings for TAT-DQA (features for FET).

TAT-DQA is one page per doc, so:
  vision_<docUid>.pt   [1, num_patches, D]   (encoded once per doc)
  question_<qaUid>.pt  [1, num_tokens, D]

Usage (server, .doc env):
    python preprocess_tatdqa.py --split dev
    python preprocess_tatdqa.py --split train --batch_size 16
"""
import os
import sys
import gc
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
from PIL import Image
from tqdm import tqdm

from datasets.tatdqa import TATDQADataset
from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder

Image.MAX_IMAGE_PIXELS = None
BASE = "/c/ujjwalb/Vansh/Datasets/TATDQA"


def load_image(p):
    try:
        im = Image.open(p).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.BILINEAR)
        return im
    except Exception:
        return Image.new("RGB", (448, 448), color="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["train", "dev", "test"])
    ap.add_argument("--json", default="", help="override QA json path")
    ap.add_argument("--docs_dir", default="", help="override docs dir")
    ap.add_argument("--out_dir", default=os.path.join(BASE, "precomputed_embeddings"))
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    qa_json = args.json or os.path.join(
        BASE, f"tatdqa_dataset_{'test_gold' if args.split == 'test' else args.split}.json")
    docs_dir = args.docs_dir or os.path.join(BASE, f"tatdqa_docs_{args.split}", args.split)
    os.makedirs(args.out_dir, exist_ok=True)

    ds = TATDQADataset(qa_json, docs_dir)
    print(f"[{args.split}] {len(ds.samples)} QA over "
          f"{len(set(s['doc_uid'] for s in ds.samples))} docs")

    with open(args.model_config) as f:
        cfg = yaml.safe_load(f)["model"]
    device = torch.device(cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if cfg.get("dtype") == "bfloat16" and torch.cuda.is_available() else torch.float32

    from transformers import ColPaliForRetrieval
    print(f"Loading ColPali {cfg['name']} ...")
    shared = ColPaliForRetrieval.from_pretrained(
        cfg["name"], torch_dtype=dtype,
        device_map="cuda" if torch.cuda.is_available() else "cpu", low_cpu_mem_usage=True)
    for p in shared.parameters():
        p.requires_grad = False
    venc = ColPaliVisionEncoder(model_name=cfg["name"], device=device, dtype=dtype, shared_model=shared)
    qenc = ColPaliQuestionEncoder(model_name=cfg["name"], device=device, dtype=dtype, shared_model=shared)

    # ---- vision: once per doc ----
    docs, seen = [], set()
    for s in ds.samples:
        if s["doc_uid"] not in seen:
            seen.add(s["doc_uid"])
            docs.append(s)
    todo = [s for s in docs if not os.path.exists(os.path.join(args.out_dir, f"vision_{s['doc_uid']}.pt"))]
    print(f"Vision: {len(todo)} docs to encode ({len(docs)-len(todo)} cached).")
    for i in tqdm(range(0, len(todo), args.batch_size), desc="Vision"):
        chunk = todo[i:i + args.batch_size]
        imgs = [load_image(s["image_path"]) for s in chunk]
        with torch.no_grad():
            embs = venc(imgs).cpu()
        for k, s in enumerate(chunk):
            torch.save(embs[k:k + 1], os.path.join(args.out_dir, f"vision_{s['doc_uid']}.pt"))
        for im in imgs:
            im.close()
        del imgs, embs
        gc.collect()
        torch.cuda.is_available() and torch.cuda.empty_cache()

    # ---- questions ----
    qtodo = [s for s in ds.samples
             if s["question_id"] and
             not os.path.exists(os.path.join(args.out_dir, f"question_{s['question_id']}.pt"))]
    print(f"Questions: {len(qtodo)} to encode.")
    for i in tqdm(range(0, len(qtodo), args.batch_size), desc="Questions"):
        chunk = qtodo[i:i + args.batch_size]
        with torch.no_grad():
            qe = qenc([s["question"] for s in chunk]).cpu()
        for k, s in enumerate(chunk):
            torch.save(qe[k:k + 1], os.path.join(args.out_dir, f"question_{s['question_id']}.pt"))
        del qe
        gc.collect()
        torch.cuda.is_available() and torch.cuda.empty_cache()

    print(f"Done -> {args.out_dir}")


if __name__ == "__main__":
    main()
