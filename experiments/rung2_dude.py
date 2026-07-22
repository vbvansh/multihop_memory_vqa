"""
RUNG 2 (DUDE): does structure-preserving page text help the reader on non-table,
dense-text documents?

Oracle gold page (isolates reader/evidence from retrieval), extractive/localizable
val subset. Reader = Qwen2.5-VL (frozen, zero-shot).
  Baseline : Qwen( gold page image , question )
  Ours     : Qwen( gold page image , question + MinerU page markdown )
Metric: ANLS + EM vs the full DUDE answer list.

RUN IN THE .doc ENV. Requires experiments/dude_cache populated first
(experiments/mineru_parse_dude.py in the mineru env).
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/rung2_dude.py --max_samples 150 --reader_batch 2
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from PIL import Image
from tqdm import tqdm

from datasets.dude import DUDEDataset
from utils.metrics import calculate_anls, calculate_exact_match

Image.MAX_IMAGE_PIXELS = None
DEFAULT_GT = "/c/ujjwalb/Vansh/Datasets/DUDE/data/2023-03-23_DUDE_gt_test_PUBLIC.json"
DEFAULT_IMAGES = ("/c/ujjwalb/Vansh/Datasets/DUDE/data/DUDE_train-val-test_binaries/"
                  "DUDE_train-val-test_binaries/images")
HERE = os.path.dirname(os.path.abspath(__file__))


def load_image(p):
    try:
        im = Image.open(p).convert("RGB")
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.BILINEAR)
        return im
    except Exception:
        return Image.new("RGB", (448, 448), color="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", default=DEFAULT_GT)
    ap.add_argument("--images_root", default=DEFAULT_IMAGES)
    ap.add_argument("--split", default="val")
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "dude_cache"))
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--reader", choices=["qwen", "paligemma"], default="qwen")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_samples", type=int, default=150)
    ap.add_argument("--require_table", action="store_true",
                    help="keep only questions whose gold page MinerU-parsed a table (image-table subset)")
    ap.add_argument("--reader_batch", type=int, default=2)
    ap.add_argument("--max_text_chars", type=int, default=6000)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False
    config["reader"]["max_new_tokens"] = args.max_new_tokens
    if args.reader == "qwen":
        config["reader"]["model_name"] = args.qwen_model

    def cache_path(did, pg):
        return os.path.join(args.cache_dir, f"{did}_{pg}.json")

    def page_has_table(did, pg):
        try:
            d = json.load(open(cache_path(did, pg), "r", encoding="utf-8"))
            return bool(d.get("tables"))
        except Exception:
            return False

    ds = DUDEDataset(args.gt_json, args.images_root, split=args.split)
    localizable = [s for s in ds.samples if s["gold_pages"]]
    # keep only questions whose gold page has a MinerU cache
    usable = [s for s in localizable if os.path.exists(cache_path(s["doc_id"], min(s["gold_pages"])))]
    n_cached = len(usable)
    if args.require_table:
        usable = [s for s in usable if page_has_table(s["doc_id"], min(s["gold_pages"]))]
        print(f"table-page subset: {len(usable)} of {n_cached} cached questions are on pages with a table.")
    if args.max_samples:
        usable = usable[:args.max_samples]
    print(f"{len(usable)} usable samples (of {len(localizable)} localizable; need MinerU cache).")
    if not usable:
        print("No usable samples — run experiments/mineru_parse_dude.py (mineru env) first.")
        return

    if args.reader == "qwen":
        from models.reader.qwen_reader import QwenVLReader
        reader = QwenVLReader(config)
    else:
        from models.reader.paligemma_reader import PaliGemmaReader
        reader = PaliGemmaReader(config)

    md_cache = {}

    def get_md(did, pg):
        k = (did, pg)
        if k not in md_cache:
            d = json.load(open(cache_path(did, pg), "r", encoding="utf-8"))
            md_cache[k] = d.get("markdown", "") or d.get("table_text", "") or ""
        return md_cache[k]

    imgs, base_q, ours_q, golds = [], [], [], []
    for s in usable:
        pg = min(s["gold_pages"])
        imgs.append(load_image(ds.image_path(s["doc_id"], s["split"], pg)))
        base_q.append(s["question"])
        md = get_md(s["doc_id"], pg)[:args.max_text_chars]
        ours_q.append(f"{s['question']}\nDocument text:\n{md}" if md else s["question"])
        golds.append(s["answers"] if s["answers"] else [""])

    def run(qs, tag):
        preds = []
        for i in tqdm(range(0, len(qs), args.reader_batch), desc=tag):
            b = slice(i, i + args.reader_batch)
            preds.extend(reader.generate(imgs[b], qs[b]))
        return preds

    base_preds = run(base_q, "baseline (page only)")
    ours_preds = run(ours_q, "ours (page + text)")

    def score(preds):
        a = e = 0.0
        for g, p in zip(golds, preds):
            a += calculate_anls(p, g)
            e += calculate_exact_match(p, g)
        n = max(len(golds), 1)
        return 100.0 * a / n, 100.0 * e / n

    bA, bE = score(base_preds)
    oA, oE = score(ours_preds)
    rname = args.qwen_model if args.reader == "qwen" else "PaliGemma-3B"
    print("=" * 72)
    print(f"RUNG 2  DUDE {args.split} (oracle page, localizable)  N={len(usable)}  reader={rname}")
    print(f"  Baseline (page only)      : ANLS {bA:5.2f} | EM {bE:5.2f}")
    print(f"  Ours     (page + page text): ANLS {oA:5.2f} | EM {oE:5.2f}")
    print(f"  delta                     : ANLS {oA-bA:+5.2f} | EM {oE-bE:+5.2f}")
    print("=" * 72)
    for s, bp, op in list(zip(usable, base_preds, ours_preds))[:8]:
        print(f"Q: {s['question'][:80]}")
        print(f"   gold={s['answers']}  type={s['answer_type']}")
        print(f"   base={bp!r}")
        print(f"   ours={op!r}")


if __name__ == "__main__":
    main()
