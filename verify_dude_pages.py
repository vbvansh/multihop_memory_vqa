"""
Sanity-check DUDE page alignment BEFORE trusting the MaxSim diagnostic.

The MaxSim top-1 on DUDE came out at ~13% (BELOW random ~18% for ~5.5-page docs),
which usually means a page-index misalignment, not a weak retriever. This script
rules that out by:

  1. Reporting the RAW bbox 'page' values vs the actual page count P per doc:
       - if any raw page == P  -> field is 1-indexed (max 1-index == P)
       - if any raw page == 0  -> field is 0-indexed
       - out-of-range counts   -> deeper ordering problem
  2. Computing MaxSim top-1 accuracy under BOTH hypotheses:
       - offset -1 (assume 1-indexed, current code)
       - offset  0 (assume 0-indexed)
     Whichever is dramatically higher reveals the true convention.

Read-only. Usage:
    python verify_dude_pages.py --split val
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from measure_maxsim_dude import (maxsim_page_scores, DEFAULT_GT, DEFAULT_IMAGES, DEFAULT_EMB)


def raw_pages(apbb):
    """All raw (un-offset) 'page' values in the annotation."""
    out = []
    if not apbb:
        return out
    for per_answer in apbb:
        if not per_answer:
            continue
        boxes = per_answer if isinstance(per_answer, list) else [per_answer]
        for b in boxes:
            if isinstance(b, dict) and b.get("page") is not None:
                out.append(int(b["page"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", default=DEFAULT_GT)
    ap.add_argument("--precomputed_dir", default=DEFAULT_EMB)
    ap.add_argument("--split", default="val")
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(args.gt_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data", obj) if isinstance(obj, dict) else obj
    data = [x for x in data if x.get("data_split") == args.split]
    if args.max_samples:
        data = data[:args.max_samples]

    # indexing evidence
    n_raw_eq_P = 0       # raw page == P  -> supports 1-indexed
    n_raw_eq_0 = 0       # raw page == 0  -> supports 0-indexed
    n_out_hi = 0         # raw page  > P  -> ordering / range problem
    min_raw, max_raw = 10**9, -10**9

    # accuracy under both hypotheses
    scored = 0
    hit_off1 = 0   # gold = page - 1 (current assumption: 1-indexed)
    hit_off0 = 0   # gold = page     (alt assumption: 0-indexed)

    for x in data:
        pr = raw_pages(x.get("answers_page_bounding_boxes"))
        if not pr:
            continue
        did, qid = x["docId"], x["questionId"]
        vpath = os.path.join(args.precomputed_dir, f"vision_{did}.pt")
        qpath = os.path.join(args.precomputed_dir, f"question_{qid}.pt")
        if not (os.path.exists(vpath) and os.path.exists(qpath)):
            continue

        page_embs = torch.load(vpath, map_location=device).float()
        query_emb = torch.load(qpath, map_location=device).float()
        if query_emb.dim() == 3:
            query_emb = query_emb.squeeze(0)
        P = page_embs.shape[0]

        for p in pr:
            min_raw = min(min_raw, p); max_raw = max(max_raw, p)
            if p == P: n_raw_eq_P += 1
            if p == 0: n_raw_eq_0 += 1
            if p > P:  n_out_hi += 1

        pred = int(torch.argmax(maxsim_page_scores(query_emb, page_embs)).item())
        gold1 = {min(max(p - 1, 0), P - 1) for p in pr}     # 1-indexed hypothesis
        gold0 = {min(max(p, 0), P - 1) for p in pr}          # 0-indexed hypothesis
        hit_off1 += int(pred in gold1)
        hit_off0 += int(pred in gold0)
        scored += 1

    def pct(a): return 100.0 * a / scored if scored else 0.0
    print("=" * 70)
    print(f"[verify/{args.split}] scored {scored} questions")
    print(f"raw page range: min={min_raw} max={max_raw}")
    print(f"raw page == P  (evidence of 1-indexed): {n_raw_eq_P}")
    print(f"raw page == 0  (evidence of 0-indexed): {n_raw_eq_0}")
    print(f"raw page  > P  (range/ordering problem): {n_out_hi}")
    print("-" * 70)
    print(f"MaxSim top-1 assuming 1-indexed (page-1): {pct(hit_off1):.2f}%  ({hit_off1}/{scored})")
    print(f"MaxSim top-1 assuming 0-indexed (page)  : {pct(hit_off0):.2f}%  ({hit_off0}/{scored})")
    print("=" * 70)
    print("INTERPRETATION:")
    print(" - If '== 0' count > 0 and '== P' count == 0  -> field is 0-indexed; use offset 0.")
    print(" - If '== P' count > 0 and '== 0' count == 0  -> field is 1-indexed; page-1 is correct.")
    print(" - Whichever accuracy is much higher is the right convention.")
    print(" - If BOTH stay ~random (~18%): not an offset bug -> page ORDERING during")
    print("   precompute is suspect (natural-sort of <docId>_<page>.jpg).")


if __name__ == "__main__":
    main()
