"""
End-to-end ANLS/EM: page selection -> reader -> answer.

Compares three page selectors feeding the SAME frozen reader:
  - Reranker  : your trained MaxSim-anchored reranker  (the system)
  - MaxSim    : pure ColPali retrieval top-1           (baseline)
  - Oracle    : the gold answer page                    (reader ceiling)

Efficient: each unique (sample, page) is read by the VLM only once.

Usage (on server, after training the reranker):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python evaluate_e2e.py                       # all val samples
    python evaluate_e2e.py --max_samples 500     # quick estimate
    python evaluate_e2e.py --reader_batch 2      # lower if OOM
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch
from PIL import Image

from models.reranker.page_reranker import PageReranker
from models.reader.paligemma_reader import PaliGemmaReader
from trainers.reranker_trainer import page_features
from datasets.docvqa import DocVQADataset
from utils.metrics import calculate_anls, calculate_exact_match
from utils.logger import setup_logger


def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (448, 448), color="white")


def main():
    parser = argparse.ArgumentParser(description="End-to-end ANLS: Reranker/MaxSim/Oracle -> Reader")
    parser.add_argument("--model_config", type=str, default="./configs/model.yaml")
    parser.add_argument("--train_config", type=str, default="./configs/train.yaml")
    parser.add_argument("--reranker_ckpt", type=str, default=None,
                        help="default: <checkpoints_dir>/reranker_best.pt")
    parser.add_argument("--max_samples", type=int, default=0, help="0 = all val samples")
    parser.add_argument("--reader_batch", type=int, default=4, help="VLM generation batch size")
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    config = {**model_config, **train_config}
    config["reader"]["use_lora"] = False   # pure zero-shot base reader (LoRA didn't help)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(name="E2E", log_dir=config["paths"].get("logs_dir", "./logs"))
    precomputed_dir = config["debug"]["precomputed_dir"]

    ckpt_dir = config["paths"].get("checkpoints_dir", "./checkpoints")
    reranker_ckpt = args.reranker_ckpt or os.path.join(ckpt_dir, "reranker_best.pt")

    # ---- reranker ----
    reranker = PageReranker(config).to(device)
    reranker.load_state_dict(torch.load(reranker_ckpt, map_location=device))
    reranker.eval()
    logger.info(f"Loaded reranker from {reranker_ckpt}")

    # ---- reader (frozen PaliGemma) ----
    reader = PaliGemmaReader(config)

    # ---- val samples ----
    ds = DocVQADataset(config, split="val", debug=False)
    samples = ds.samples[:args.max_samples] if args.max_samples else ds.samples

    # ---- 1) page selection per sample (reranker / maxsim / oracle) ----
    choices = []
    skipped = 0
    for s in samples:
        qid = s["question_id"]
        vpath = os.path.join(precomputed_dir, f"vision_{qid}.pt")
        qpath = os.path.join(precomputed_dir, f"question_{qid}.pt")
        if not (os.path.exists(vpath) and os.path.exists(qpath)):
            skipped += 1
            continue
        page_embs = torch.load(vpath, map_location=device).float()
        query_emb = torch.load(qpath, map_location=device).float()
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)

        P = page_embs.shape[0]
        ms, page_feat = page_features(query_emb, page_embs)              # [P], [P,2D]
        ms_n = (ms - ms.mean()) / (ms.std(unbiased=False) + 1e-6)
        ms_n = torch.nan_to_num(ms_n)
        q_mean = query_emb.mean(dim=0)
        mask = torch.zeros(1, P, dtype=torch.bool, device=device)
        with torch.no_grad():
            logits = reranker(page_feat.unsqueeze(0), q_mean.unsqueeze(0), ms_n.unsqueeze(0), mask)
        r_page = int(logits.argmax(dim=1).item())
        m_page = int(ms.argmax().item())
        g_page = max(0, min(int(s["answer_page_idx"]), P - 1))

        choices.append({
            "image_paths": s["image_paths"],
            "question": s["question"],
            "answers": s["answers"] if s.get("answers") else [""],
            "r": r_page, "m": m_page, "g": g_page, "P": P,
        })
    logger.info(f"Selected pages for {len(choices)} samples (skipped {skipped} missing embeddings).")

    # ---- 2) unique (sample, page) reader jobs ----
    jobs = {}
    for i, c in enumerate(choices):
        for pg in {c["r"], c["m"], c["g"]}:
            jobs[(i, pg)] = None
    keys = list(jobs.keys())
    logger.info(f"Reader will generate for {len(keys)} unique (sample,page) pairs "
                f"(vs {3*len(choices)} naive).")

    # ---- 3) batched reader generation ----
    bs = args.reader_batch
    from tqdm import tqdm
    for start in tqdm(range(0, len(keys), bs), desc="Reader generate"):
        batch_keys = keys[start:start + bs]
        images, questions = [], []
        for (i, pg) in batch_keys:
            c = choices[i]
            paths = c["image_paths"]
            path = paths[min(pg, len(paths) - 1)] if paths else None
            images.append(load_image(path) if path else Image.new("RGB", (448, 448), "white"))
            questions.append(c["question"])
        preds = reader.generate(images, questions)
        for k, p in zip(batch_keys, preds):
            jobs[k] = p

    # ---- 4) score each selector ----
    def score(which):
        a = e = 0.0
        for i, c in enumerate(choices):
            pred = jobs[(i, c[which])]
            a += calculate_anls(pred, c["answers"])
            e += calculate_exact_match(pred, c["answers"])
        n = max(len(choices), 1)
        return 100.0 * a / n, 100.0 * e / n

    (ra, re_), (ma, me), (ga, ge) = score("r"), score("m"), score("g")
    logger.info("=" * 70)
    logger.info(f"END-TO-END on {len(choices)} val samples:")
    logger.info(f"  Reranker -> Reader : ANLS {ra:.2f} | EM {re_:.2f}   (your system)")
    logger.info(f"  MaxSim   -> Reader : ANLS {ma:.2f} | EM {me:.2f}   (baseline)")
    logger.info(f"  Oracle   -> Reader : ANLS {ga:.2f} | EM {ge:.2f}   (ceiling / gold page)")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
