"""
Diagnostic (read-only): pure ColPali late-interaction (MaxSim) page-selection accuracy.

This trains NOTHING. It scores each page of a document by the standard ColPali
late-interaction score:  page_score = sum_over_query_tokens( max_over_page_patches( q . d ) )
picks the highest-scoring page, and compares it to the gold answer_page_idx.

The result is directly comparable to the learned router's val page-accuracy (~47%).
Reuses the precomputed embeddings you already have (vision_<qid>.pt / question_<qid>.pt),
so it runs in minutes with no model loading.

Usage:
    python measure_maxsim.py                 # val split (matches the router eval)
    python measure_maxsim.py --split both    # val + train
    python measure_maxsim.py --topk 3
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yaml
import torch

from datasets.docvqa import DocVQADataset
from utils.logger import setup_logger


def maxsim_page_scores(query_emb, page_embs):
    """
    query_emb: [Lq, D]        (question token vectors)
    page_embs: [P, N, D]      (P pages, N patches per page)
    returns:   [P]            ColPali late-interaction score per page
    """
    # sims[p, i, j] = q_i . patch_j  of page p
    sims = torch.einsum("qd,pnd->pqn", query_emb, page_embs)   # [P, Lq, N]
    max_over_patches = sims.max(dim=2).values                  # [P, Lq]
    return max_over_patches.sum(dim=1)                         # [P]


def run_split(config, split, precomputed_dir, logger, device, topk=3):
    ds = DocVQADataset(config, split=split, debug=False)

    total = correct = topk_correct = 0
    sp_total = sp_correct = 0     # single-page docs
    mp_total = mp_correct = 0     # multi-page docs
    missing = 0

    for s in ds.samples:
        qid = s["question_id"]
        vpath = os.path.join(precomputed_dir, f"vision_{qid}.pt")
        qpath = os.path.join(precomputed_dir, f"question_{qid}.pt")
        if not (os.path.exists(vpath) and os.path.exists(qpath)):
            missing += 1
            continue

        page_embs = torch.load(vpath, map_location=device).float()   # [P, N, D]
        query_emb = torch.load(qpath, map_location=device).float()
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)                         # [Lq, D]

        P = page_embs.shape[0]
        gt = max(0, min(int(s["answer_page_idx"]), P - 1))

        scores = maxsim_page_scores(query_emb, page_embs)            # [P]
        pred = int(torch.argmax(scores).item())
        k = min(topk, P)
        topk_idx = torch.topk(scores, k).indices.tolist()

        total += 1
        correct += int(pred == gt)
        topk_correct += int(gt in topk_idx)
        if P == 1:
            sp_total += 1
            sp_correct += int(pred == gt)
        else:
            mp_total += 1
            mp_correct += int(pred == gt)

    def pct(a, b):
        return (100.0 * a / b) if b else 0.0

    logger.info("=" * 70)
    logger.info(f"[MaxSim/{split}] N={total} (skipped {missing} with missing embeddings)")
    logger.info(f"[MaxSim/{split}] Top-1 page acc : {pct(correct, total):.2f}%  ({correct}/{total})")
    logger.info(f"[MaxSim/{split}] Top-{topk} recall   : {pct(topk_correct, total):.2f}%")
    logger.info(f"[MaxSim/{split}] single-page    : {pct(sp_correct, sp_total):.2f}%  ({sp_correct}/{sp_total})")
    logger.info(f"[MaxSim/{split}] MULTI-PAGE     : {pct(mp_correct, mp_total):.2f}%  ({mp_correct}/{mp_total})   <-- the real head-to-head vs the router")
    logger.info("=" * 70)
    return pct(correct, total)


def main():
    parser = argparse.ArgumentParser(description="Pure ColPali MaxSim page-selection accuracy (read-only).")
    parser.add_argument("--model_config", type=str, default="./configs/model.yaml")
    parser.add_argument("--train_config", type=str, default="./configs/train.yaml")
    parser.add_argument("--split", type=str, default="val", choices=["val", "train", "both"])
    parser.add_argument("--topk", type=int, default=3)
    args = parser.parse_args()

    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.train_config) as f:
        train_config = yaml.safe_load(f)
    config = {**model_config, **train_config}

    precomputed_dir = config["debug"]["precomputed_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(name="MaxSim", log_dir=config["paths"].get("logs_dir", "./logs"))
    logger.info(f"Precomputed dir: {precomputed_dir} | device: {device}")

    splits = ["val", "train"] if args.split == "both" else [args.split]
    for sp in splits:
        run_split(config, sp, precomputed_dir, logger, device, topk=args.topk)


if __name__ == "__main__":
    main()
