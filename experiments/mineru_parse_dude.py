"""
Parse DUDE gold pages with MinerU -> cache full-page markdown + tables.

RUN IN THE `mineru` CONDA ENV. DUDE pages are scanned, so MinerU runs full OCR
(~1-3 min/page on GPU) -> this is the slow step. Start with a slice (--max_samples)
and let it run (resumable; skips pages already cached).

    conda activate mineru
    python experiments/mineru_parse_dude.py --split val --max_samples 150

Cache: experiments/dude_cache/<docId>_<page>.json = {markdown, tables, table_text}
Only the gold page of each (localizable) question is parsed, deduped by (doc,page).
"""
import os
import sys
import json
import glob
import shutil
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.dude import DUDEDataset

DEFAULT_GT = "/c/ujjwalb/Vansh/Datasets/DUDE/data/2023-03-23_DUDE_gt_test_PUBLIC.json"
DEFAULT_IMAGES = ("/c/ujjwalb/Vansh/Datasets/DUDE/data/DUDE_train-val-test_binaries/"
                  "DUDE_train-val-test_binaries/images")
HERE = os.path.dirname(os.path.abspath(__file__))


def read_markdown(raw_dir, stem):
    hits = glob.glob(os.path.join(raw_dir, stem, "**", "*.md"), recursive=True)
    return open(hits[0], encoding="utf-8", errors="ignore").read() if hits else ""


def read_tables(raw_dir, stem):
    cls = glob.glob(os.path.join(raw_dir, stem, "**", "*_content_list.json"), recursive=True)
    tables = []
    if cls:
        obj = json.load(open(cls[0], encoding="utf-8"))
        for b in (obj if isinstance(obj, list) else []):
            if isinstance(b, dict) and "table" in str(b.get("type", "")).lower():
                body = b.get("table_body") or b.get("html") or ""
                if body:
                    tables.append(body)
    return tables


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", default=DEFAULT_GT)
    ap.add_argument("--images_root", default=DEFAULT_IMAGES)
    ap.add_argument("--split", default="val")
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "dude_cache"))
    ap.add_argument("--raw_dir", default=os.path.join(HERE, "dude_raw"))
    ap.add_argument("--max_samples", type=int, default=150, help="cap questions (dedupes to fewer pages)")
    ap.add_argument("--keep_raw", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    ds = DUDEDataset(args.gt_json, args.images_root, split=args.split)
    localizable = [s for s in ds.samples if s["gold_pages"]]
    if args.max_samples:
        localizable = localizable[:args.max_samples]

    pages = {}   # (doc_id, page) -> split
    for s in localizable:
        pages[(s["doc_id"], min(s["gold_pages"]))] = s["split"]
    print(f"{len(localizable)} questions -> {len(pages)} unique gold pages to parse.")

    done = skip = fail = 0
    items = list(pages.items())
    for i, ((did, pg), sp) in enumerate(items):
        stem = f"{did}_{pg}"
        cache_path = os.path.join(args.cache_dir, stem + ".json")
        if os.path.exists(cache_path):
            skip += 1
            continue
        img = os.path.join(args.images_root, sp, f"{did}_{pg}.jpg")
        if not os.path.exists(img):
            print(f"  [no image] {img}")
            fail += 1
            continue
        try:
            subprocess.run(["mineru", "-p", img, "-o", args.raw_dir],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"  [mineru failed] {stem}: {e}")
            fail += 1
            continue
        md = read_markdown(args.raw_dir, stem)
        tables = read_tables(args.raw_dir, stem)
        json.dump({"markdown": md, "tables": tables, "table_text": "\n\n".join(tables)},
                  open(cache_path, "w", encoding="utf-8"))
        done += 1
        if not args.keep_raw:
            shutil.rmtree(os.path.join(args.raw_dir, stem), ignore_errors=True)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(items)}] done={done} skip={skip} fail={fail}")
    print(f"FINISHED. done={done} skip={skip} fail={fail}. cache -> {args.cache_dir}")


if __name__ == "__main__":
    main()
