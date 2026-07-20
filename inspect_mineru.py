"""
Inspect MinerU output: summarize detected block types and dump any tables.

Run this AFTER you have parsed some sample files with MinerU's CLI, e.g.:
    mineru -p sample.pdf -o ~/mineru_test/out
Then:
    python inspect_mineru.py ~/mineru_test/out

It is deliberately format-tolerant (MinerU's JSON layout varies by version):
it walks the output dir, finds every .json / .md, tallies block "type" fields,
and prints the reconstructed table content so you can eyeball quality.
"""
import os
import sys
import re
import json
from collections import Counter


def find_files(root, exts):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(exts):
                out.append(os.path.join(dp, fn))
    return out


def extract_blocks(obj):
    """Collect every dict that has a string 'type' field, anywhere in the JSON."""
    blocks = []

    def walk(o):
        if isinstance(o, dict):
            if isinstance(o.get("type"), str):
                blocks.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return blocks


def main():
    dirs = sys.argv[1:]
    if not dirs:
        print("usage: python inspect_mineru.py <mineru_out_dir> [<dir2> ...]")
        return

    for d in dirs:
        print("=" * 80)
        print("DIR:", d)
        if not os.path.isdir(d):
            print("  (not a directory — skipping)")
            continue

        jsons = find_files(d, (".json",))
        mds = find_files(d, (".md",))
        print(f"found {len(jsons)} json, {len(mds)} md files")

        types = Counter()
        tables = []
        for jp in jsons:
            try:
                obj = json.load(open(jp, encoding="utf-8"))
            except Exception:
                continue
            for b in extract_blocks(obj):
                t = b["type"].lower()
                types[t] += 1
                if "table" in t:
                    tables.append(b)

        print("block types:", dict(types))
        print(f"TABLE blocks found: {len(tables)}")
        for i, tb in enumerate(tables[:3]):
            print(f"--- table {i+1}: keys = {list(tb.keys())}")
            for k in ("table_body", "html", "text", "content", "table_caption", "img_path"):
                if k in tb and tb[k]:
                    print(f"    [{k}] {str(tb[k])[:900]}")

        # also scan the markdown MinerU writes, for table markup
        for mp in mds[:2]:
            txt = open(mp, encoding="utf-8", errors="ignore").read()
            html_tabs = txt.count("<table")
            md_tab_lines = sum(1 for ln in txt.splitlines() if ln.count("|") >= 2)
            print(f"MD {os.path.basename(mp)}: <table> tags={html_tabs}, "
                  f"markdown-table-ish lines={md_tab_lines}")
            m = re.search(r"<table.*?</table>", txt, re.S)
            if m:
                print("    first <table> snippet:", m.group(0)[:900])
    print("=" * 80)
    print("GO/NO-GO: we want TABLE blocks > 0 with readable cell content "
          "(rows/columns/headers), plus text/title/figure types present.")


if __name__ == "__main__":
    main()
