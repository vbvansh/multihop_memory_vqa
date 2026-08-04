"""
Explore IndicDLP parquet shards (ai4bharat/indicdlp on HF).

The HF release bundles images + COCO-style annotations inside parquet shards
(data/train-*, validation-*, test-*), so we analyse shard by shard instead of
downloading annotations separately.

Answers the questions that decide whether a dense/table benchmark is viable:
  - what columns/schema does a shard have?
  - which layout classes exist (is "Table" one of the 42)?
  - language + document-category distribution
  - how many pages contain a TABLE, and how dense are pages (regions/page)?

Uses pandas/pyarrow (NOT the `datasets` lib, which this repo's local `datasets/`
package shadows).

Usage:
    # after downloading a few shards (see fetch_indicdlp.py):
    python Benchmark_Multilingual/explore_indicdlp_parquet.py --shards /path/to/*.parquet
    python Benchmark_Multilingual/explore_indicdlp_parquet.py --dir /path/to/data --max_shards 3
"""
import os
import glob
import json
import argparse
from collections import Counter

import pandas as pd

TABLE_HINTS = ("table", "tabular", "cell")
TEXT_HINTS = ("text", "paragraph", "para", "body", "line", "list", "caption", "title", "heading")


def find_labels(obj, out):
    """Collect any label/category strings from a nested annotation object."""
    if isinstance(obj, dict):
        for k in ("label", "category", "category_name", "class", "type", "name"):
            v = obj.get(k)
            if isinstance(v, str):
                out.append(v.lower())
                break
        for v in obj.values():
            find_labels(v, out)
    elif isinstance(obj, list):
        for v in obj:
            find_labels(v, out)


def maybe_json(v):
    if isinstance(v, (dict, list)):
        return v
    if isinstance(v, str) and v.strip().startswith(("{", "[")):
        try:
            return json.loads(v)
        except Exception:
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="*", default=None, help="explicit parquet files")
    ap.add_argument("--dir", default="", help="directory containing parquet shards")
    ap.add_argument("--max_shards", type=int, default=3)
    ap.add_argument("--max_rows", type=int, default=0, help="0 = all rows in the shards read")
    args = ap.parse_args()

    files = args.shards or sorted(glob.glob(os.path.join(args.dir, "**", "*.parquet"), recursive=True))
    files = files[:args.max_shards] if args.max_shards else files
    if not files:
        print("No parquet shards found. Download a few first (fetch_indicdlp.py --pattern ...).")
        return
    print(f"Reading {len(files)} shard(s):")
    for f in files:
        print(f"  {os.path.basename(f)}  ({os.path.getsize(f)/1e6:.0f} MB)")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    if args.max_rows:
        df = df.head(args.max_rows)
    print(f"\nrows: {len(df):,}")

    print("\n--- SCHEMA ---")
    for c in df.columns:
        v = df[c].iloc[0]
        t = type(v).__name__
        prev = ""
        if isinstance(v, (str, int, float)):
            prev = str(v)[:90]
        elif isinstance(v, dict):
            prev = f"keys={list(v.keys())[:8]}"
        elif isinstance(v, (list, tuple)):
            prev = f"len={len(v)} first={str(v[0])[:60] if len(v) else ''}"
        elif hasattr(v, "shape"):
            prev = f"array shape={getattr(v,'shape',None)}"
        print(f"  {c:26s} {t:12s} {prev}")

    # ---- language / category columns (best effort) ----
    lang_col = next((c for c in df.columns if c.lower() in
                     ("language", "lang", "script", "language_name")), None)
    cat_col = next((c for c in df.columns if c.lower() in
                    ("domain", "category", "doc_type", "document_type", "type", "doc_category")), None)
    print(f"\nlanguage column: {lang_col}   category column: {cat_col}")
    if lang_col:
        print("\n--- LANGUAGES ---")
        for k, v in df[lang_col].astype(str).value_counts().head(20).items():
            print(f"  {k:20s} {v:,}")
    if cat_col:
        print("\n--- DOCUMENT CATEGORIES ---")
        for k, v in df[cat_col].astype(str).value_counts().head(20).items():
            print(f"  {k:28s} {v:,}")

    # ---- annotations: labels, density, tables ----
    ann_cols = [c for c in df.columns
                if any(h in c.lower() for h in ("annot", "object", "label", "region", "layout", "bbox"))]
    print(f"\nannotation-ish columns: {ann_cols}")

    label_counter = Counter()
    per_page_regions, table_pages = [], 0
    lang_tables, cat_tables = Counter(), Counter()

    for i, row in df.iterrows():
        labels = []
        for c in (ann_cols or df.columns):
            o = maybe_json(row[c])
            if o is not None:
                find_labels(o, labels)
            elif isinstance(row[c], (list, tuple)) and len(row[c]) and isinstance(row[c][0], (str, dict)):
                find_labels(list(row[c]), labels)
        if not labels:
            continue
        per_page_regions.append(len(labels))
        for l in labels:
            label_counter[l] += 1
        if any(any(h in l for h in TABLE_HINTS) for l in labels):
            table_pages += 1
            if lang_col:
                lang_tables[str(row[lang_col])] += 1
            if cat_col:
                cat_tables[str(row[cat_col])] += 1

    print("\n--- LAYOUT CLASSES FOUND ---")
    for l, c in label_counter.most_common(45):
        mark = "  <-- TABLE" if any(h in l for h in TABLE_HINTS) else ""
        print(f"  {l:30s} {c:,}{mark}")

    n = len(per_page_regions)
    if n:
        per_page_regions.sort()
        print("\n--- PAGE DENSITY (regions per page) ---")
        print(f"  pages analysed {n:,} | mean {sum(per_page_regions)/n:.1f} | "
              f"median {per_page_regions[n//2]} | p90 {per_page_regions[int(0.9*n)]} | "
              f"max {per_page_regions[-1]}")
        print(f"  dense (>=15 regions): {sum(1 for d in per_page_regions if d>=15):,} "
              f"({100.0*sum(1 for d in per_page_regions if d>=15)/n:.1f}%)")
        print(f"  very dense (>=30)   : {sum(1 for d in per_page_regions if d>=30):,}")

    print("\n--- TABLE COVERAGE ---")
    print(f"  pages with a table: {table_pages:,} / {n:,} "
          f"({100.0*table_pages/max(n,1):.1f}%)")
    if lang_tables:
        print("  by language:")
        for k, v in lang_tables.most_common(15):
            print(f"    {k:20s} {v:,}")
    if cat_tables:
        print("  by category:")
        for k, v in cat_tables.most_common(15):
            print(f"    {k:28s} {v:,}")

    frac = table_pages / max(n, 1)
    print("\n" + "=" * 74)
    print(f"EXTRAPOLATION over the full 119,806 images: ~{int(frac*119806):,} table pages")
    print("Viable for a 10K dense/table benchmark if this is comfortably >10K and")
    print("spread across languages (check the by-language breakdown above).")
    print("=" * 74)


if __name__ == "__main__":
    main()
