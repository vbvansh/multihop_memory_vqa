"""
Discovery pass over the IndicDLP dataset (read-only).

We don't yet know IndicDLP's exact layout/annotation format, so this script does NOT
assume one. It walks the dataset, detects the annotation style (COCO-style vs
per-image json), and answers the questions that decide whether a dense/table-heavy
multilingual benchmark is buildable:

  1. What is the directory structure / file inventory?
  2. What layout REGION LABELS exist (Table, Text, Figure, ...) and how frequent?
  3. What LANGUAGES and DOCUMENT CATEGORIES exist, and how many docs each?
  4. HOW MANY DOCUMENTS CONTAIN TABLES, per language and per category?   <-- the decider
  5. How dense are the documents (regions per page, text regions per page)?

Usage (server):
    python Benchmark_Multilingual/explore_indicdlp.py --root /path/to/IndicDLP
    python Benchmark_Multilingual/explore_indicdlp.py --root ... --dump_sample sample.json
"""
import os
import json
import argparse
from collections import Counter, defaultdict

IMG_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp")

# Region labels we treat as "table-like" / "text-like" when tallying density.
TABLE_HINTS = ("table", "tabular", "cell", "grid")
TEXT_HINTS = ("text", "paragraph", "para", "body", "line", "list", "caption", "title", "heading")


def human(n):
    return f"{n:,}"


def walk_inventory(root, max_depth=3):
    """File-type inventory + top-level directory tree."""
    ext_counts = Counter()
    dir_counts = Counter()
    json_files = []
    for dp, dns, fns in os.walk(root):
        rel = os.path.relpath(dp, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth <= max_depth:
            dir_counts[rel] += len(fns)
        for fn in fns:
            ext = os.path.splitext(fn)[1].lower()
            ext_counts[ext] += 1
            if ext == ".json" and len(json_files) < 5000:
                json_files.append(os.path.join(dp, fn))
    return ext_counts, dir_counts, json_files


def is_coco(obj):
    return isinstance(obj, dict) and "images" in obj and "annotations" in obj


def guess_language(path, fields):
    """Language may appear in the path (…/hindi/…) or in an annotation field."""
    for key in ("language", "lang", "Language", "script"):
        v = fields.get(key)
        if isinstance(v, str) and v:
            return v.lower()
    langs = ["hindi", "gujarati", "marathi", "bengali", "tamil", "telugu", "kannada",
             "malayalam", "punjabi", "odia", "oriya", "assamese", "urdu", "english",
             "hi", "gu", "mr", "bn", "ta", "te", "kn", "ml", "pa", "or", "as", "en"]
    low = path.lower().replace("\\", "/")
    for part in low.split("/"):
        if part in langs:
            return part
    for l in langs:
        if f"/{l}_" in low or f"_{l}/" in low or f"/{l}/" in low:
            return l
    return "unknown"


def guess_category(path, fields):
    for key in ("category", "doc_type", "document_type", "domain", "type", "source"):
        v = fields.get(key)
        if isinstance(v, str) and v:
            return v.lower()
    parts = os.path.normpath(path).split(os.sep)
    return parts[-2].lower() if len(parts) >= 2 else "unknown"


def analyze_coco(path, per_lang_tables, per_cat_tables, label_counter,
                 lang_docs, cat_docs, density):
    obj = json.load(open(path, "r", encoding="utf-8"))
    cats = {c["id"]: str(c.get("name", c["id"])).lower() for c in obj.get("categories", [])}
    imgs = {im["id"]: im for im in obj.get("images", [])}
    per_img_labels = defaultdict(list)
    for a in obj.get("annotations", []):
        name = cats.get(a.get("category_id"), str(a.get("category_id")))
        label_counter[name] += 1
        per_img_labels[a.get("image_id")].append(name)

    for iid, im in imgs.items():
        fn = im.get("file_name", "")
        lang = guess_language(os.path.join(path, fn), im)
        cat = guess_category(fn, im)
        lang_docs[lang] += 1
        cat_docs[cat] += 1
        labels = per_img_labels.get(iid, [])
        density.append(len(labels))
        has_table = any(any(h in l for h in TABLE_HINTS) for l in labels)
        n_text = sum(1 for l in labels if any(h in l for h in TEXT_HINTS))
        if has_table:
            per_lang_tables[lang] += 1
            per_cat_tables[cat] += 1
        # dense = many text regions on the page
        if n_text >= 15:
            per_lang_tables[lang + "  (dense-text)"] += 0  # keep key order tidy; counted below
    return len(imgs)


def analyze_per_image_json(paths, per_lang_tables, per_cat_tables, label_counter,
                           lang_docs, cat_docs, density, limit):
    n = 0
    for p in paths[:limit]:
        try:
            obj = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        labels = []

        def walk(o):
            if isinstance(o, dict):
                for k in ("label", "category", "class", "type", "BlockType", "region_type"):
                    v = o.get(k)
                    if isinstance(v, str):
                        labels.append(v.lower())
                        break
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)

        walk(obj)
        if not labels:
            continue
        n += 1
        fields = obj if isinstance(obj, dict) else {}
        lang = guess_language(p, fields)
        cat = guess_category(p, fields)
        lang_docs[lang] += 1
        cat_docs[cat] += 1
        density.append(len(labels))
        for l in labels:
            label_counter[l] += 1
        if any(any(h in l for h in TABLE_HINTS) for l in labels):
            per_lang_tables[lang] += 1
            per_cat_tables[cat] += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="IndicDLP dataset root")
    ap.add_argument("--max_json", type=int, default=3000, help="cap per-image jsons to scan")
    ap.add_argument("--dump_sample", default="", help="write one raw annotation sample here")
    args = ap.parse_args()

    print("=" * 78)
    print("INDICDLP DISCOVERY:", args.root)
    print("=" * 78)

    ext_counts, dir_counts, json_files = walk_inventory(args.root)
    print("\n--- FILE INVENTORY (by extension) ---")
    for e, c in ext_counts.most_common(15):
        print(f"  {e or '(none)':10s} {human(c)}")
    n_images = sum(c for e, c in ext_counts.items() if e in IMG_EXT)
    print(f"  images total: {human(n_images)}")

    print("\n--- DIRECTORIES (files per dir, depth<=3) ---")
    for d, c in sorted(dir_counts.items())[:40]:
        print(f"  {d:60s} {human(c)}")

    print(f"\n--- ANNOTATIONS: found {human(len(json_files))} json files ---")
    coco_files, other_json = [], []
    for p in json_files[:200]:
        try:
            obj = json.load(open(p, "r", encoding="utf-8"))
        except Exception:
            continue
        (coco_files if is_coco(obj) else other_json).append(p)
    print(f"  COCO-style: {len(coco_files)}   other/per-image: {len(other_json)} (of first 200 scanned)")

    label_counter, lang_docs, cat_docs = Counter(), Counter(), Counter()
    per_lang_tables, per_cat_tables, density = Counter(), Counter(), []

    if coco_files:
        print("\n  Parsing COCO-style annotation files...")
        total = 0
        for p in coco_files:
            try:
                total += analyze_coco(p, per_lang_tables, per_cat_tables, label_counter,
                                      lang_docs, cat_docs, density)
            except Exception as e:
                print(f"   [skip] {os.path.basename(p)}: {e}")
        print(f"  parsed {human(total)} image entries")
    else:
        print(f"\n  Parsing up to {args.max_json} per-image jsons...")
        n = analyze_per_image_json(json_files, per_lang_tables, per_cat_tables, label_counter,
                                   lang_docs, cat_docs, density, args.max_json)
        print(f"  parsed {human(n)} annotation files")

    print("\n--- REGION LABELS (what the layout annotations contain) ---")
    for l, c in label_counter.most_common(25):
        print(f"  {l:28s} {human(c)}")

    print("\n--- DOCUMENTS PER LANGUAGE ---")
    for l, c in lang_docs.most_common(20):
        t = per_lang_tables.get(l, 0)
        pct = (100.0 * t / c) if c else 0
        print(f"  {l:20s} docs {human(c):>10s}   with-table {human(t):>8s} ({pct:5.1f}%)")

    print("\n--- DOCUMENTS PER CATEGORY ---")
    for l, c in cat_docs.most_common(25):
        t = per_cat_tables.get(l, 0)
        pct = (100.0 * t / c) if c else 0
        print(f"  {l:28s} docs {human(c):>10s}   with-table {human(t):>8s} ({pct:5.1f}%)")

    if density:
        density.sort()
        n = len(density)
        print("\n--- DOCUMENT DENSITY (layout regions per page) ---")
        print(f"  mean {sum(density)/n:.1f} | median {density[n//2]} | "
              f"p90 {density[int(0.9*n)]} | max {density[-1]}")
        print(f"  pages with >=15 regions (dense): {human(sum(1 for d in density if d >= 15))} / {human(n)}")
        print(f"  pages with >=30 regions (very dense): {human(sum(1 for d in density if d >= 30))}")

    total_tables = sum(v for k, v in per_lang_tables.items() if "(dense" not in k)
    print("\n" + "=" * 78)
    print(f"VERDICT INPUT: ~{human(total_tables)} documents contain a table region.")
    print("Benchmark is viable if this is a few thousand AND spread across languages.")
    print("=" * 78)

    if args.dump_sample and json_files:
        src = (coco_files or json_files)[0]
        obj = json.load(open(src, "r", encoding="utf-8"))
        if is_coco(obj):
            obj = {"categories": obj.get("categories", []),
                   "images": obj.get("images", [])[:2],
                   "annotations": obj.get("annotations", [])[:5]}
        json.dump(obj, open(args.dump_sample, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"\nWrote a raw annotation sample to {args.dump_sample} (send me this file).")


if __name__ == "__main__":
    main()
