"""
Parse document PDFs with MinerU and cache the structured output (tables + regions).

RUN IN THE `mineru` CONDA ENV (Python 3.11, has GPU torch):
    conda activate mineru
    python experiments/mineru_parse.py --max_docs 130

Writes one small file per document:
    experiments/mineru_cache/<uid>.json = {uid, table_text, tables, regions}
The Rung-1 runner (in the .doc env) only READS these cache files, so the two
environments never mix. Re-running skips docs already cached (resumable).

By default it parses the TAT-DQA dev docs referenced by the QA json.
"""
import os
import sys
import json
import glob
import shutil
import argparse
import subprocess

DEFAULT_JSON = "/c/ujjwalb/Vansh/Datasets/TATDQA/tatdqa_dataset_dev.json"
DEFAULT_DOCS = "/c/ujjwalb/Vansh/Datasets/TATDQA/tatdqa_docs_dev/dev"
HERE = os.path.dirname(os.path.abspath(__file__))


def find_content_list(out_dir, uid):
    hits = glob.glob(os.path.join(out_dir, uid, "**", "*_content_list.json"), recursive=True)
    return hits[0] if hits else None


def extract(content_list_path):
    """Pull table HTML + a compact region list from MinerU's content_list.json."""
    obj = json.load(open(content_list_path, "r", encoding="utf-8"))
    items = obj if isinstance(obj, list) else []
    tables, regions = [], []
    for b in items:
        if not isinstance(b, dict):
            continue
        t = str(b.get("type", "")).lower()
        regions.append({"type": t, "bbox": b.get("bbox"), "page": b.get("page_idx")})
        if "table" in t:
            body = b.get("table_body") or b.get("html") or ""
            if body:
                tables.append(body)
    return "\n\n".join(tables), tables, regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--docs_dir", default=DEFAULT_DOCS)
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "mineru_cache"))
    ap.add_argument("--raw_dir", default=os.path.join(HERE, "mineru_raw"))
    ap.add_argument("--max_docs", type=int, default=0, help="0 = all docs in the json")
    ap.add_argument("--keep_raw", action="store_true", help="keep MinerU's full output (default: delete to save disk)")
    args = ap.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)
    os.makedirs(args.raw_dir, exist_ok=True)

    data = json.load(open(args.json, "r", encoding="utf-8"))
    uids, seen = [], set()
    for doc in data:
        uid = doc.get("doc", {}).get("uid")
        if uid and uid not in seen:
            seen.add(uid)
            uids.append(uid)
    if args.max_docs:
        uids = uids[:args.max_docs]
    print(f"{len(uids)} unique docs to consider.")

    done = skip = fail = 0
    for i, uid in enumerate(uids):
        cache_path = os.path.join(args.cache_dir, uid + ".json")
        if os.path.exists(cache_path):
            skip += 1
            continue

        pdf = os.path.join(args.docs_dir, uid + ".pdf")
        if not os.path.exists(pdf):
            c = glob.glob(os.path.join(args.docs_dir, uid + "*.pdf"))
            pdf = c[0] if c else None
        if not pdf:
            print(f"  [no pdf] {uid}")
            fail += 1
            continue

        try:
            subprocess.run(["mineru", "-p", pdf, "-o", args.raw_dir],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        except Exception as e:
            print(f"  [mineru failed] {uid}: {e}")
            fail += 1
            continue

        cl = find_content_list(args.raw_dir, uid)
        if not cl:
            print(f"  [no content_list] {uid}")
            fail += 1
            continue

        table_text, tables, regions = extract(cl)
        json.dump({"uid": uid, "table_text": table_text, "tables": tables, "regions": regions},
                  open(cache_path, "w", encoding="utf-8"))
        done += 1

        if not args.keep_raw:
            shutil.rmtree(os.path.join(args.raw_dir, uid), ignore_errors=True)
        if (i + 1) % 10 == 0:
            print(f"[{i+1}/{len(uids)}] done={done} skip={skip} fail={fail}")

    print(f"FINISHED. done={done} skip={skip} fail={fail}. cache -> {args.cache_dir}")


if __name__ == "__main__":
    main()
