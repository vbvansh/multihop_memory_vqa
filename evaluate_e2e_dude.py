"""
End-to-end ANLS/EM on DUDE: page selection -> PaliGemma reader -> answer.

Compares two page selectors feeding the SAME frozen reader on the LOCALIZABLE val
subset (questions that have a gold answer page):
  - MaxSim : pure ColPali retrieval top-1   (the system's retriever)
  - Oracle : the gold answer page           (reader ceiling)

The Oracle vs MaxSim gap = retrieval error budget.
The Oracle score itself = the reader's ceiling on DUDE (how hard the answers are).

(No reranker here: page selection is ~saturated on DUDE, so we first want the
reader ceiling. A reranker can be added later if the ceiling justifies it.)

Vision embeddings are per docId; images are images/<split>/<docId>_<page>.jpg.

Usage (server):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python evaluate_e2e_dude.py --split val --max_samples 500   # quick estimate
    python evaluate_e2e_dude.py --split val                     # full localizable subset
    python evaluate_e2e_dude.py --reader_batch 2                # lower if OOM
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
from PIL import Image
from tqdm import tqdm

from datasets.dude import DUDEDataset
from measure_maxsim_dude import maxsim_page_scores, DEFAULT_GT, DEFAULT_IMAGES, DEFAULT_EMB
from models.reader.paligemma_reader import PaliGemmaReader
from utils.metrics import calculate_anls, calculate_exact_match
from utils.logger import setup_logger

Image.MAX_IMAGE_PIXELS = None


def load_image(path):
    try:
        img = Image.open(path).convert("RGB")
        if max(img.size) > 1600:
            img.thumbnail((1600, 1600), Image.BILINEAR)
        return img
    except Exception:
        return Image.new("RGB", (448, 448), color="white")


def main():
    ap = argparse.ArgumentParser(description="DUDE end-to-end ANLS: MaxSim/Oracle -> PaliGemma.")
    ap.add_argument("--gt_json", default=DEFAULT_GT)
    ap.add_argument("--images_root", default=DEFAULT_IMAGES)
    ap.add_argument("--precomputed_dir", default=DEFAULT_EMB)
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--train_config", default="./configs/train.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all localizable")
    ap.add_argument("--reader_batch", type=int, default=4)
    args = ap.parse_args()

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    config = {**model_config, **train_config}
    config["reader"]["use_lora"] = False   # zero-shot base reader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(name="E2E-DUDE", log_dir="./logs")

    reader = PaliGemmaReader(config)

    ds = DUDEDataset(args.gt_json, args.images_root, split=args.split)
    # localizable subset only (needs a gold page for MaxSim/Oracle comparison)
    samples = [s for s in ds.samples if s["gold_pages"]]
    if args.max_samples:
        samples = samples[:args.max_samples]
    logger.info(f"[{args.split}] {len(samples)} localizable QA "
                f"(of {len(ds.samples)} total; non-localizable are excluded here).")

    # ---- 1) page selection per sample ----
    choices, skipped = [], 0
    for s in samples:
        did, qid = s["doc_id"], s["question_id"]
        vpath = os.path.join(args.precomputed_dir, f"vision_{did}.pt")
        qpath = os.path.join(args.precomputed_dir, f"question_{qid}.pt")
        if not (os.path.exists(vpath) and os.path.exists(qpath)):
            skipped += 1
            continue
        page_embs = torch.load(vpath, map_location=device).float()
        query_emb = torch.load(qpath, map_location=device).float()
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)
        P = page_embs.shape[0]
        m_page = int(maxsim_page_scores(query_emb, page_embs).argmax().item())
        g_page = min(min(s["gold_pages"]), P - 1)   # a gold page (first if several)
        choices.append({
            "doc_id": did, "split": s["split"], "question": s["question"],
            "answers": s["answers"] if s["answers"] else [""],
            "m": m_page, "g": g_page,
        })
    logger.info(f"Selected pages for {len(choices)} samples (skipped {skipped} missing embeddings).")

    # ---- 2) unique (sample, page) reader jobs ----
    jobs = {}
    for i, c in enumerate(choices):
        for pg in {c["m"], c["g"]}:
            jobs[(i, pg)] = None
    keys = list(jobs.keys())
    logger.info(f"Reader will generate for {len(keys)} unique (sample,page) pairs (vs {2*len(choices)} naive).")

    # ---- 3) batched reader generation ----
    bs = args.reader_batch
    for start in tqdm(range(0, len(keys), bs), desc="Reader generate"):
        batch_keys = keys[start:start + bs]
        images, questions = [], []
        for (i, pg) in batch_keys:
            c = choices[i]
            path = ds.image_path(c["doc_id"], c["split"], pg)
            images.append(load_image(path))
            questions.append(c["question"])
        preds = reader.generate(images, questions)
        for k, p in zip(batch_keys, preds):
            jobs[k] = p

    # ---- 4) score ----
    def score(which):
        a = e = 0.0
        for i, c in enumerate(choices):
            pred = jobs[(i, c[which])]
            a += calculate_anls(pred, c["answers"])
            e += calculate_exact_match(pred, c["answers"])
        n = max(len(choices), 1)
        return 100.0 * a / n, 100.0 * e / n

    (ma, me), (ga, ge) = score("m"), score("g")
    logger.info("=" * 70)
    logger.info(f"DUDE END-TO-END on {len(choices)} localizable {args.split} samples:")
    logger.info(f"  MaxSim -> Reader : ANLS {ma:.2f} | EM {me:.2f}   (system)")
    logger.info(f"  Oracle -> Reader : ANLS {ga:.2f} | EM {ge:.2f}   (reader ceiling / gold page)")
    logger.info(f"  retrieval gap    : {ga - ma:.2f} ANLS")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
