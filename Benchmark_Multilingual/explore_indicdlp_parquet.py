"""
Explore IndicDLP parquet shards (ai4bharat/indicdlp on HF).

Actual HF schema (verified):
    image         dict    {'bytes': ..., 'path': 'relative/path.png'}
    bboxes        ndarray array of boxes for the page
    category_ids  ndarray NUMERIC layout-class ids (one per box)

So: labels are ids (not strings), and language/domain must be recovered from
image['path']. This script reports the id histogram, the path structure, page
density, and (if a category mapping is supplied) the named class distribution
and table coverage.

Usage:
    python Benchmark_Multilingual/explore_indicdlp_parquet.py --dir <shards> --max_shards 3
    # once you have the id->name mapping (json {"1": "Text", "7": "Table", ...}):
    python Benchmark_Multilingual/explore_indicdlp_parquet.py --dir <shards> --categories cats.json
"""
import os
import glob
import json
import argparse
from collections import Counter

import pandas as pd

TABLE_HINTS = ("table", "tabular", "cell")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="")
    ap.add_argument("--shards", nargs="*", default=None)
    ap.add_argument("--max_shards", type=int, default=3)
    ap.add_argument("--categories", default="", help="optional json mapping category_id -> class name")
    ap.add_argument("--table_ids", default="", help="comma-separated ids that mean TABLE (if known)")
    args = ap.parse_args()

    files = args.shards or sorted(glob.glob(os.path.join(args.dir, "**", "*.parquet"), recursive=True))
    files = files[:args.max_shards] if args.max_shards else files
    if not files:
        print("No parquet shards found.")
        return
    print(f"Reading {len(files)} shard(s)")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"rows: {len(df):,}\n")

    cats = {}
    if args.categories and os.path.exists(args.categories):
        raw = json.load(open(args.categories, encoding="utf-8"))
        if isinstance(raw, dict):
            cats = {int(k): str(v) for k, v in raw.items()}
        elif isinstance(raw, list):        # COCO "categories": [{id, name}, ...]
            cats = {int(c["id"]): str(c.get("name", c["id"])) for c in raw}
        print(f"loaded {len(cats)} category names\n")

    table_ids = set()
    if args.table_ids:
        table_ids = {int(x) for x in args.table_ids.split(",") if x.strip()}
    elif cats:
        table_ids = {i for i, n in cats.items() if any(h in n.lower() for h in TABLE_HINTS)}
    if table_ids:
        print(f"treating these ids as TABLE: {sorted(table_ids)}\n")

    # ---------- paths: where language / domain live ----------
    paths = []
    for v in df["image"].head(2000):
        p = v.get("path") if isinstance(v, dict) else None
        if p:
            paths.append(str(p))
    print("--- SAMPLE image['path'] VALUES (language/domain likely encoded here) ---")
    for p in paths[:15]:
        print("   ", p)

    depth = Counter(p.count("/") for p in paths)
    print(f"\npath depth histogram: {dict(depth)}")
    for lvl in range(0, 4):
        vals = Counter(p.split("/")[lvl] for p in paths if p.count("/") >= lvl)
        if 1 < len(vals) <= 40:
            print(f"\n--- path level {lvl} ({len(vals)} distinct) — likely language or domain ---")
            for k, c in vals.most_common(40):
                print(f"    {k:30s} {c:,}")

    # ---------- layout classes ----------
    id_counter, per_page, table_pages, pages = Counter(), [], 0, 0
    for ids in df["category_ids"]:
        try:
            lst = list(ids)
        except Exception:
            continue
        pages += 1
        per_page.append(len(lst))
        for i in lst:
            id_counter[int(i)] += 1
        if table_ids and any(int(i) in table_ids for i in lst):
            table_pages += 1

    print(f"\n--- CATEGORY IDS ({len(id_counter)} distinct across {pages:,} pages) ---")
    for i, c in sorted(id_counter.items()):
        name = cats.get(i, "")
        mark = "  <-- TABLE" if i in table_ids else ""
        print(f"    id {i:3d}  {name:28s} {c:,}{mark}")

    if per_page:
        per_page.sort()
        n = len(per_page)
        print("\n--- PAGE DENSITY (regions per page) ---")
        print(f"    mean {sum(per_page)/n:.1f} | median {per_page[n//2]} | "
              f"p90 {per_page[int(0.9*n)]} | max {per_page[-1]}")
        for thr in (10, 15, 30):
            k = sum(1 for d in per_page if d >= thr)
            print(f"    pages with >={thr:2d} regions: {k:,} ({100.0*k/n:.1f}%)")

    if table_ids:
        print(f"\n--- TABLE COVERAGE ---\n    pages with a table: {table_pages:,}/{pages:,} "
              f"({100.0*table_pages/max(pages,1):.1f}%)")
        print(f"    extrapolated over 119,806 images: ~{int(119806*table_pages/max(pages,1)):,} table pages")
    else:
        print("\n[!] No category mapping supplied — rerun with --categories cats.json "
              "(or --table_ids N) to get table coverage.")


if __name__ == "__main__":
    main()
