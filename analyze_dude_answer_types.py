"""
Analyze DUDE answer-type distribution (read-only).

Counts each answer_type per split, plus coarse buckets (extractive / abstractive /
list / unanswerable), and how many are LOCALIZABLE (have a gold answer page).
This quantifies the reader-side failure buckets and feeds the DUDE dataset summary.

Usage:
    python analyze_dude_answer_types.py
    python analyze_dude_answer_types.py --split val
"""
import os
import sys
import json
import argparse
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datasets.dude import _extract_gold_pages

DEFAULT_GT = "/c/ujjwalb/Vansh/Datasets/DUDE/data/2023-03-23_DUDE_gt_test_PUBLIC.json"


def coarse_bucket(atype):
    """Map a raw answer_type string (e.g. 'list/extractive') to coarse buckets it belongs to."""
    t = (atype or "").lower()
    buckets = []
    if any(k in t for k in ["not-answerable", "not answerable", "unanswerable", "non-answerable", "none"]):
        buckets.append("unanswerable")
    if "list" in t:
        buckets.append("list")
    if "abstractive" in t:
        buckets.append("abstractive")
    if "extractive" in t:
        buckets.append("extractive")
    if not buckets:
        buckets.append("other/" + (t if t else "empty"))
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", default=DEFAULT_GT)
    ap.add_argument("--split", default="all", choices=["all", "train", "val", "test"])
    args = ap.parse_args()

    with open(args.gt_json, "r", encoding="utf-8") as f:
        obj = json.load(f)
    data = obj.get("data", obj) if isinstance(obj, dict) else obj

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]

    for split in splits:
        rows = [x for x in data if x.get("data_split") == split]
        raw = Counter()
        coarse = Counter()
        coarse_localizable = defaultdict(lambda: [0, 0])   # bucket -> [localizable, total]
        n_localizable = 0

        for x in rows:
            atype = x.get("answer_type", "")
            raw[atype] += 1
            has_gold = len(_extract_gold_pages(x.get("answers_page_bounding_boxes"))) > 0
            n_localizable += int(has_gold)
            for b in coarse_bucket(atype):
                coarse[b] += 1
                coarse_localizable[b][1] += 1
                coarse_localizable[b][0] += int(has_gold)

        n = len(rows)
        print("=" * 74)
        print(f"SPLIT: {split}   ({n} QA;  localizable={n_localizable} = {100.0*n_localizable/max(n,1):.1f}%)")
        print("-" * 74)
        print("Coarse buckets (a QA can belong to several, e.g. 'list/extractive'):")
        for b, c in coarse.most_common():
            loc, tot = coarse_localizable[b]
            print(f"  {b:22s} {c:6d}  ({100.0*c/max(n,1):5.1f}%)   localizable {loc}/{tot} "
                  f"({100.0*loc/max(tot,1):.0f}%)")
        print("-" * 74)
        print("Raw answer_type strings:")
        for t, c in raw.most_common():
            print(f"  {str(t):30s} {c:6d}  ({100.0*c/max(n,1):5.1f}%)")
    print("=" * 74)


if __name__ == "__main__":
    main()
