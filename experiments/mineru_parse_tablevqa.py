"""
Parse TableVQA table images with MinerU -> cache markdown + tables.

RUN IN THE `mineru` CONDA ENV, AFTER dumping images with
experiments/dump_tablevqa_images.py (in the .doc env). Table renders are clean, so
MinerU is fast here. Resumable.

    conda activate mineru
    python experiments/mineru_parse_tablevqa.py

Reads experiments/tablevqa_images/*.png, writes
experiments/tablevqa_cache/<qa_id>.json = {markdown, tables, table_text}.
Uses only stdlib (no pandas), so the mineru env needs nothing extra.
"""
import os
import sys
import json
import glob
import shutil
import argparse
import subprocess

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
    ap.add_argument("--images_dir", default=os.path.join(HERE, "tablevqa_images"))
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "tablevqa_cache"))
    ap.add_argument("--raw_dir", default=os.path.join(HERE, "tablevqa_raw"))
    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--keep_raw", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    imgs = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    if args.max_images:
        imgs = imgs[:args.max_images]
    print(f"{len(imgs)} table images to parse.")

    done = skip = fail = 0
    for i, img in enumerate(imgs):
        stem = os.path.splitext(os.path.basename(img))[0]
        cache_path = os.path.join(args.cache_dir, stem + ".json")
        if os.path.exists(cache_path):
            skip += 1
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
        if (i + 1) % 25 == 0:
            print(f"[{i+1}/{len(imgs)}] done={done} skip={skip} fail={fail}")
    print(f"FINISHED. done={done} skip={skip} fail={fail}. cache -> {args.cache_dir}")


if __name__ == "__main__":
    main()
