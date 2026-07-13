"""
Precompute ColPali embeddings for the DUDE dataset.

Difference from MPDocVQA (preprocess_dataset.py):
  DUDE has ~8.26 questions per document. To avoid re-encoding a document's pages
  once per question, we DEDUPLICATE by docId:
    - vision embedding is saved ONCE per document  -> vision_<docId>.pt   [num_pages, num_patches, D]
    - question embedding is saved per QA sample     -> question_<questionId>.pt [1, num_tokens, D]
  The DUDE loader looks up vision by docId, so logically every QA still has its
  page embeddings (just stored once per doc -> ~8x less compute and disk).

Usage (on server):
  # 1) DRY RUN FIRST — verifies image discovery without encoding anything:
  python preprocess_dude.py --gt_json /c/ujjwalb/Vansh/Datasets/DUDE/data/<train_val_gt>.json \
      --images_root /c/ujjwalb/Vansh/Datasets/DUDE/data/DUDE_train-val-test_binaries/images \
      --out_dir /c/ujjwalb/Vansh/Datasets/DUDE/precomputed_embeddings \
      --dry_run

  # 2) If dry run reports correct page counts, run for real:
  python preprocess_dude.py --gt_json <same> --images_root <same> --out_dir <same> --batch_size 32

Run once per GT json (train/val, then the test PUBLIC json). Resumes automatically
(skips docs/questions already saved).
"""
import os
import sys
import re
import gc
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
from PIL import Image
from tqdm import tqdm

# DUDE has a few enormous scans (100+ megapixel). ColPali resizes to 448 anyway,
# so we cap the decode size: silence the bomb warning and downscale big images.
Image.MAX_IMAGE_PIXELS = None
MAX_SIDE = 1600  # longest side; well above 448 so no retrieval detail is lost

from models.encoders.vision_encoder import ColPaliVisionEncoder
from models.encoders.question_encoder import ColPaliQuestionEncoder

IMG_EXTS = ("jpg", "jpeg", "png", "JPG", "JPEG", "PNG")


def natural_key(path):
    """Sort page files by the trailing integer in the filename (page order)."""
    name = os.path.splitext(os.path.basename(path))[0]
    nums = re.findall(r"\d+", name)
    return int(nums[-1]) if nums else 0


def discover_page_images(images_root, doc_id, split):
    """
    Return the doc's page-image paths in page order.

    Tries the common DUDE layouts in order:
      <root>/<split>/<docId>/*.<ext>     (per-doc subfolder inside split)
      <root>/<docId>/*.<ext>             (per-doc subfolder, no split dir)
      <root>/<split>/<docId>_*.<ext>     (flat split dir, docId-prefixed files)
      <root>/<docId>_*.<ext>             (flat dir, docId-prefixed files)
    Adjust the CANDIDATE patterns below if your layout differs.
    """
    candidates = [
        os.path.join(images_root, split, doc_id),
        os.path.join(images_root, doc_id),
    ]
    for d in candidates:
        if os.path.isdir(d):
            files = []
            for ext in IMG_EXTS:
                files.extend(glob.glob(os.path.join(d, f"*.{ext}")))
            if files:
                return sorted(set(files), key=natural_key)

    flat_prefixes = [
        os.path.join(images_root, split, doc_id + "_*"),
        os.path.join(images_root, doc_id + "_*"),
        os.path.join(images_root, split, doc_id + "*"),
    ]
    for pat in flat_prefixes:
        files = [f for f in glob.glob(pat) if f.rsplit(".", 1)[-1] in IMG_EXTS]
        if files:
            return sorted(set(files), key=natural_key)

    return []


def load_gt(gt_json):
    """DUDE GT is either a plain list of QA dicts or {'data': [...]}. Handle both."""
    with open(gt_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        obj = obj.get("data", obj.get("questions", []))
    return obj


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        if max(img.size) > MAX_SIDE:
            img.thumbnail((MAX_SIDE, MAX_SIDE), Image.BILINEAR)
        return img
    except Exception as e:
        print(f"  [warn] could not open {path}: {e}")
        return Image.new("RGB", (448, 448), color="white")


def main():
    parser = argparse.ArgumentParser(description="Precompute ColPali embeddings for DUDE.")
    parser.add_argument("--gt_json", required=True, help="Path to a DUDE GT json (train/val or test PUBLIC).")
    parser.add_argument("--images_root", required=True, help="Root dir containing DUDE page images.")
    parser.add_argument("--out_dir", required=True, help="Where to write vision_<docId>.pt / question_<qid>.pt.")
    parser.add_argument("--split", default=None,
                        help="Optional: only process QA whose data_split == this. Default: all in the json.")
    parser.add_argument("--batch_size", type=int, default=32, help="Pages/questions per encoder forward pass.")
    parser.add_argument("--dry_run", action="store_true", help="Only report image discovery; encode nothing.")
    parser.add_argument("--max_docs", type=int, default=0, help="Process at most N docs (0 = all). For quick tests.")
    parser.add_argument("--model_config", default="./configs/model.yaml")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- parse GT: build docId -> pages, and qid -> (question, docId, split) ----
    qa = load_gt(args.gt_json)
    if args.split:
        qa = [x for x in qa if x.get("data_split") == args.split]
    print(f"Loaded {len(qa)} QA entries from {args.gt_json}"
          + (f" (filtered to split={args.split})" if args.split else ""))

    doc_split = {}          # docId -> split (for image discovery)
    doc_pages = {}          # docId -> [paths]  (filled during discovery)
    questions = []          # (qid, question_text)
    seen_docs = []
    for x in qa:
        did = x["docId"]
        qid = x["questionId"]
        questions.append((qid, x["question"]))
        if did not in doc_split:
            doc_split[did] = x.get("data_split", args.split or "")
            seen_docs.append(did)

    if args.max_docs:
        seen_docs = seen_docs[:args.max_docs]
        keep = set(seen_docs)
        questions = [(qid, q) for (qid, q) in questions
                     if next(z["docId"] for z in qa if z["questionId"] == qid) in keep]

    print(f"Unique documents: {len(seen_docs)} | questions: {len(questions)}")

    # ---- image discovery ----
    missing, total_pages, page_counts = [], 0, []
    for did in tqdm(seen_docs, desc="Discovering page images"):
        pages = discover_page_images(args.images_root, did, doc_split[did])
        doc_pages[did] = pages
        if not pages:
            missing.append(did)
        else:
            total_pages += len(pages)
            page_counts.append(len(pages))

    print("\n===== DISCOVERY REPORT =====")
    print(f"Docs with images    : {len(seen_docs) - len(missing)}/{len(seen_docs)}")
    print(f"Docs MISSING images : {len(missing)}")
    print(f"Total page images   : {total_pages}")
    if page_counts:
        print(f"Pages/doc           : min={min(page_counts)} "
              f"avg={sum(page_counts)/len(page_counts):.2f} max={max(page_counts)}")
    if missing[:5]:
        print(f"Example missing docIds: {missing[:5]}")
    # show a couple of resolved examples so you can eyeball the naming
    for did in seen_docs[:2]:
        ex = doc_pages.get(did, [])
        print(f"  {did}: {len(ex)} pages; first={ex[0] if ex else None}")
    print("============================\n")

    if args.dry_run:
        print("Dry run only — no embeddings written. If page counts look right, re-run without --dry_run.")
        return
    if missing and len(missing) == len(seen_docs):
        print("ERROR: no images found for ANY doc. Fix --images_root or the CANDIDATE patterns "
              "in discover_page_images(), then re-run --dry_run.")
        return

    # ---- load ColPali backbone + encoders ----
    with open(args.model_config, "r") as f:
        model_config = yaml.safe_load(f)
    config = {"model": model_config["model"]}
    device = torch.device(config["model"]["device"] if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if config["model"]["dtype"] == "bfloat16" and torch.cuda.is_available() else torch.float32
    model_name = config["model"]["name"]

    from transformers import ColPaliForRetrieval
    quantize = config["model"].get("quantize_4bit", False)
    if quantize and torch.cuda.is_available():
        from transformers import BitsAndBytesConfig
        print(f"Loading ColPali {model_name} in 4-bit...")
        qc = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                llm_int8_skip_modules=["embedding_proj_layer"])
        shared_model = ColPaliForRetrieval.from_pretrained(
            model_name, quantization_config=qc, device_map="auto", low_cpu_mem_usage=True)
    else:
        print(f"Loading ColPali {model_name} in {dtype}...")
        shared_model = ColPaliForRetrieval.from_pretrained(
            model_name, torch_dtype=dtype,
            device_map="cuda" if torch.cuda.is_available() else "cpu", low_cpu_mem_usage=True)
    for p in shared_model.parameters():
        p.requires_grad = False

    vision_encoder = ColPaliVisionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)
    question_encoder = ColPaliQuestionEncoder(model_name=model_name, device=device, dtype=dtype, shared_model=shared_model)

    # ================= VISION: encode each doc's pages once =================
    # Resume: skip docs already saved.
    docs_todo = [d for d in seen_docs
                 if doc_pages.get(d) and not os.path.exists(os.path.join(args.out_dir, f"vision_{d}.pt"))]
    print(f"\nVision: {len(docs_todo)} docs to encode "
          f"({len(seen_docs) - len(docs_todo)} already done or imageless).")

    # Flatten pages across docs so the encoder always sees full batches.
    flat = [(did, path) for did in docs_todo for path in doc_pages[did]]
    total_needed = {did: len(doc_pages[did]) for did in docs_todo}
    buf = {did: [] for did in docs_todo}   # accumulate page embeddings per doc

    for start in tqdm(range(0, len(flat), args.batch_size), desc="Vision encode"):
        chunk = flat[start:start + args.batch_size]
        imgs = [load_image(p) for (_, p) in chunk]
        with torch.no_grad():
            embs = vision_encoder(imgs).cpu()   # [chunk, num_patches, D]
        for k, (did, _) in enumerate(chunk):
            buf[did].append(embs[k])
            if len(buf[did]) == total_needed[did]:      # doc complete -> save & free
                doc_emb = torch.stack(buf[did], dim=0)  # [num_pages, num_patches, D]
                torch.save(doc_emb, os.path.join(args.out_dir, f"vision_{did}.pt"))
                buf[did] = None
        for im in imgs:
            im.close()
        del imgs, embs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ================= QUESTIONS: encode per QA sample =================
    q_todo = [(qid, q) for (qid, q) in questions
              if not os.path.exists(os.path.join(args.out_dir, f"question_{qid}.pt"))]
    print(f"\nQuestions: {len(q_todo)} to encode "
          f"({len(questions) - len(q_todo)} already done).")

    for start in tqdm(range(0, len(q_todo), args.batch_size), desc="Question encode"):
        chunk = q_todo[start:start + args.batch_size]
        texts = [q for (_, q) in chunk]
        with torch.no_grad():
            qembs = question_encoder(texts).cpu()   # [chunk, num_tokens, D]
        for k, (qid, _) in enumerate(chunk):
            torch.save(qembs[k:k + 1], os.path.join(args.out_dir, f"question_{qid}.pt"))
        del qembs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nDone. Embeddings in {args.out_dir}")
    print("  vision_<docId>.pt   [num_pages, num_patches, D]")
    print("  question_<qid>.pt   [1, num_tokens, D]")


if __name__ == "__main__":
    main()
