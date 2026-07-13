"""
Diagnostic (read-only): pure ColPali MaxSim page-selection accuracy on DUDE.

Trains nothing. For each val/train question that HAS a gold page (derived from
answers_page_bounding_boxes), it scores each page by ColPali late interaction:
    page_score = sum_over_query_tokens( max_over_page_patches( q . d ) )
picks the top page(s), and checks against the gold page set.

This is THE decisive experiment: it tells us whether MaxSim is already strong on
DUDE (page selection saturated -> pivot novelty to region localization) or weak
(headroom -> page-level learned module can win).

Vision embeddings are per docId (vision_<docId>.pt); questions per qid.

Usage (server):
    python measure_maxsim_dude.py --split val
    python measure_maxsim_dude.py --split val --topk 5
    python measure_maxsim_dude.py --split train --max_samples 3000
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from datasets.dude import DUDEDataset
from utils.logger import setup_logger

DEFAULT_GT = "/c/ujjwalb/Vansh/Datasets/DUDE/data/2023-03-23_DUDE_gt_test_PUBLIC.json"
DEFAULT_IMAGES = ("/c/ujjwalb/Vansh/Datasets/DUDE/data/DUDE_train-val-test_binaries/"
                  "DUDE_train-val-test_binaries/images")
DEFAULT_EMB = "/c/ujjwalb/Vansh/Datasets/DUDE/precomputed_embeddings"


def maxsim_page_scores(query_emb, page_embs):
    """query_emb [Lq,D], page_embs [P,N,D] -> [P] late-interaction scores."""
    sims = torch.einsum("qd,pnd->pqn", query_emb, page_embs)   # [P, Lq, N]
    return sims.max(dim=2).values.sum(dim=1)                   # [P]


def main():
    ap = argparse.ArgumentParser(description="ColPali MaxSim page accuracy on DUDE (read-only).")
    ap.add_argument("--gt_json", default=DEFAULT_GT)
    ap.add_argument("--images_root", default=DEFAULT_IMAGES)
    ap.add_argument("--precomputed_dir", default=DEFAULT_EMB)
    ap.add_argument("--split", default="val", choices=["val", "train"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = setup_logger(name="MaxSimDUDE", log_dir="./logs")
    logger.info(f"GT: {args.gt_json}")
    logger.info(f"Embeddings: {args.precomputed_dir} | device: {device}")

    ds = DUDEDataset(args.gt_json, args.images_root, split=args.split)
    samples = ds.samples[:args.max_samples] if args.max_samples else ds.samples
    logger.info(f"[{args.split}] {len(samples)} QA total")

    # counters
    total = correct = topk_correct = 0
    sp_total = sp_correct = 0          # single-page documents (P==1)
    mp_total = mp_correct = mp_topk = 0  # multi-page documents (P>1)  <- the real test
    mpa_total = mpa_correct = 0        # multi-page ANSWER (gold spans >1 page)
    no_gold = missing_emb = 0

    for s in samples:
        if not s["gold_pages"]:            # unanswerable / no bbox / test -> can't score
            no_gold += 1
            continue
        did, qid = s["doc_id"], s["question_id"]
        vpath = os.path.join(args.precomputed_dir, f"vision_{did}.pt")
        qpath = os.path.join(args.precomputed_dir, f"question_{qid}.pt")
        if not (os.path.exists(vpath) and os.path.exists(qpath)):
            missing_emb += 1
            continue

        page_embs = torch.load(vpath, map_location=device).float()   # [P,N,D]
        query_emb = torch.load(qpath, map_location=device).float()
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)                         # [Lq,D]

        P = page_embs.shape[0]
        gold = {min(g, P - 1) for g in s["gold_pages"]}              # clamp to available pages
        scores = maxsim_page_scores(query_emb, page_embs)           # [P]
        pred = int(torch.argmax(scores).item())
        k = min(args.topk, P)
        topk_idx = set(torch.topk(scores, k).indices.tolist())

        hit1 = int(pred in gold)
        hitk = int(len(topk_idx & gold) > 0)

        total += 1
        correct += hit1
        topk_correct += hitk
        if P == 1:
            sp_total += 1; sp_correct += hit1
        else:
            mp_total += 1; mp_correct += hit1; mp_topk += hitk
        if len(s["gold_pages"]) > 1:      # answer text localized on multiple pages
            mpa_total += 1; mpa_correct += hit1

    def pct(a, b):
        return (100.0 * a / b) if b else 0.0

    logger.info("=" * 74)
    logger.info(f"[MaxSim-DUDE/{args.split}] scored N={total}  "
                f"(no-gold/unlabeled {no_gold}, missing-emb {missing_emb})")
    logger.info(f"  Top-1 page acc      : {pct(correct, total):.2f}%  ({correct}/{total})")
    logger.info(f"  Top-{args.topk} recall        : {pct(topk_correct, total):.2f}%")
    logger.info(f"  single-page docs    : {pct(sp_correct, sp_total):.2f}%  ({sp_correct}/{sp_total})")
    logger.info(f"  MULTI-PAGE docs     : {pct(mp_correct, mp_total):.2f}%  ({mp_correct}/{mp_total})   "
                f"top-{args.topk} {pct(mp_topk, mp_total):.2f}%   <-- headroom lives here")
    logger.info(f"  multi-page ANSWERS  : {pct(mpa_correct, mpa_total):.2f}%  ({mpa_correct}/{mpa_total})")
    logger.info("=" * 74)


if __name__ == "__main__":
    main()
