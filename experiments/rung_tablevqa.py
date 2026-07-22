"""
TableVQA-Bench: the strong second leg for the image-table structure claim.

Reader = Qwen2.5-VL (frozen). THREE conditions on the same samples:
  Baseline : image only
  Ours     : image + MinerU-reconstructed table (our pipeline)
  Oracle   : image + dataset's GOLD HTML table   (ceiling if structure were perfect)
=> shows how much of the oracle-structure gain our MinerU reconstruction recovers.

Metric: approximate EM/F1, overall and per subset. Optional --pot for arithmetic.

RUN IN THE .doc ENV. Requires:
  1) experiments/dump_tablevqa_images.py  (.doc)   -> dumps images
  2) experiments/mineru_parse_tablevqa.py (mineru)  -> fills tablevqa_cache
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/rung_tablevqa.py --subsets vwtq fintabnetqa --reader_batch 2
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from PIL import Image
from tqdm import tqdm

from datasets.tablevqa import TableVQADataset
from experiments.eval_utils import evaluate
from experiments.pot import POT_SYSTEM, parse_pot

Image.MAX_IMAGE_PIXELS = None
DEFAULT_DIR = "/c/ujjwalb/Vansh/Datasets/TableVQA-Bench/data"
HERE = os.path.dirname(os.path.abspath(__file__))


def safe(qid):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in qid)


def load_image(sample, ds):
    try:
        im = ds.image(sample)
        if max(im.size) > 1600:
            im.thumbnail((1600, 1600), Image.BILINEAR)
        return im
    except Exception:
        return Image.new("RGB", (448, 448), color="white")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DIR)
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "tablevqa_cache"))
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--subsets", nargs="*", default=["vwtq", "fintabnetqa"])
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--reader_batch", type=int, default=2)
    ap.add_argument("--max_table_chars", type=int, default=4000)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--pot", action="store_true")
    args = ap.parse_args()

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False
    config["reader"]["model_name"] = args.qwen_model
    config["reader"]["max_new_tokens"] = max(args.max_new_tokens, 256) if args.pot else args.max_new_tokens

    def cache_path(qid):
        return os.path.join(args.cache_dir, safe(qid) + ".json")

    ds = TableVQADataset(args.data_dir, subsets=args.subsets)
    usable = [s for s in ds.samples if os.path.exists(cache_path(s["qa_id"]))]
    if args.max_samples:
        usable = usable[:args.max_samples]
    print(f"{len(usable)} usable samples (of {len(ds.samples)}; need MinerU cache).")
    if not usable:
        print("No cache — run dump_tablevqa_images.py (.doc) then mineru_parse_tablevqa.py (mineru).")
        return

    from models.reader.qwen_reader import QwenVLReader
    reader = QwenVLReader(config)

    def mineru_table(qid):
        d = json.load(open(cache_path(qid), "r", encoding="utf-8"))
        return (d.get("markdown") or d.get("table_text") or "")

    imgs = [load_image(s, ds) for s in usable]
    base_q = [s["question"] for s in usable]
    ours_q, oracle_q = [], []
    for s in usable:
        mt = mineru_table(s["qa_id"])[:args.max_table_chars]
        gh = s["html"][:args.max_table_chars]
        ours_q.append(f"{s['question']}\nTable:\n{mt}" if mt else s["question"])
        oracle_q.append(f"{s['question']}\nTable:\n{gh}" if gh else s["question"])

    def run(qs, tag):
        preds = []
        for i in tqdm(range(0, len(qs), args.reader_batch), desc=tag):
            b = slice(i, i + args.reader_batch)
            if args.pot:
                out = reader.generate(imgs[b], [f"{q}\n(If a calculation is needed, end with 'ANSWER = <expression>'.)"
                                                for q in qs[b]])
                preds.extend(parse_pot(o) for o in out)
            else:
                preds.extend(reader.generate(imgs[b], qs[b]))
        return preds

    base_preds = run(base_q, "baseline (image)")
    ours_preds = run(ours_q, "ours (image+MinerU)")
    oracle_preds = run(oracle_q, "oracle (image+gold HTML)")

    def score(preds, subset=None):
        em = f1 = 0.0
        n = 0
        for s, p in zip(usable, preds):
            if subset and s["subset"] != subset:
                continue
            e, f = evaluate(p, [s["gt"]])
            em += e
            f1 += f
            n += 1
        n = max(n, 1)
        return 100.0 * em / n, 100.0 * f1 / n, n

    subsets = [None] + sorted(set(s["subset"] for s in usable))
    print("=" * 88)
    print(f"TableVQA  N={len(usable)}  reader={args.qwen_model}"
          f"{'  +PoT' if args.pot else ''}")
    for sub in subsets:
        label = sub or "ALL"
        bE, bF, n = score(base_preds, sub)
        oE, oF, _ = score(ours_preds, sub)
        gE, gF, _ = score(oracle_preds, sub)
        print(f"  [{label:12s} n={n:4d}]  base EM {bE:5.1f}/F1 {bF:5.1f} | "
              f"MinerU EM {oE:5.1f}/F1 {oF:5.1f} | oracle EM {gE:5.1f}/F1 {gF:5.1f} | "
              f"dEM(MinerU-base) {oE-bE:+5.1f}")
    print("=" * 88)
    for s, bp, op, gp in list(zip(usable, base_preds, ours_preds, oracle_preds))[:8]:
        print(f"Q[{s['subset']}]: {s['question'][:75]}")
        print(f"   gt={s['gt']}  base={bp!r}  MinerU={op!r}  oracle={gp!r}")


if __name__ == "__main__":
    main()
