"""
RUNG 1 (TAT-DQA): does structure-preserving table text help the reader?

Same frozen reader (PaliGemma-3B, zero-shot), two conditions:
  Baseline : PaliGemma( page image , question )
  Ours     : PaliGemma( page image , question + MinerU table text )

Reports approximate EM/F1 for both so we can see if structure helps.

RUN IN THE .doc ENV (torch + GPU for PaliGemma). Requires the MinerU cache to be
populated first (run experiments/mineru_parse.py in the mineru env):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/rung1_tatdqa.py --max_samples 150 --reader_batch 4
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from PIL import Image
from tqdm import tqdm

from datasets.tatdqa import TATDQADataset
from experiments.eval_utils import evaluate
from models.reader.paligemma_reader import PaliGemmaReader

Image.MAX_IMAGE_PIXELS = None
DEFAULT_JSON = "/c/ujjwalb/Vansh/Datasets/TATDQA/tatdqa_dataset_dev.json"
DEFAULT_DOCS = "/c/ujjwalb/Vansh/Datasets/TATDQA/tatdqa_docs_dev/dev"
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
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--docs_dir", default=DEFAULT_DOCS)
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "mineru_cache"))
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--max_samples", type=int, default=150)
    ap.add_argument("--reader_batch", type=int, default=4)
    ap.add_argument("--max_table_chars", type=int, default=1800)
    ap.add_argument("--max_new_tokens", type=int, default=32)
    args = ap.parse_args()

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False               # zero-shot base reader
    config["reader"]["max_new_tokens"] = args.max_new_tokens

    def cache_path(uid):
        return os.path.join(args.cache_dir, uid + ".json")

    ds = TATDQADataset(args.json, args.docs_dir)
    # keep only questions whose doc has a MinerU cache (so both conditions are comparable)
    usable = [s for s in ds.samples if s["question_id"] and os.path.exists(cache_path(s["doc_uid"]))]
    if args.max_samples:
        usable = usable[:args.max_samples]
    print(f"{len(usable)} usable samples (of {len(ds.samples)} total; need MinerU cache).")
    if not usable:
        print("No usable samples — run experiments/mineru_parse.py (mineru env) first.")
        return

    reader = PaliGemmaReader(config)

    table_text_by_uid = {}

    def get_table(uid):
        if uid not in table_text_by_uid:
            d = json.load(open(cache_path(uid), "r", encoding="utf-8"))
            table_text_by_uid[uid] = d.get("table_text", "") or ""
        return table_text_by_uid[uid]

    imgs = [load_image(s["image_path"]) for s in usable]
    base_q = [s["question"] for s in usable]
    ours_q = []
    for s in usable:
        tt = get_table(s["doc_uid"])[:args.max_table_chars]
        ours_q.append(f"{s['question']}\nTable:\n{tt}" if tt else s["question"])

    def run(qs, tag):
        preds = []
        for i in tqdm(range(0, len(qs), args.reader_batch), desc=tag):
            b = slice(i, i + args.reader_batch)
            preds.extend(reader.generate(imgs[b], qs[b]))
        return preds

    base_preds = run(base_q, "baseline (page only)")
    ours_preds = run(ours_q, "ours (page + table)")

    def score(preds):
        em = f1 = 0.0
        for s, p in zip(usable, preds):
            e, f = evaluate(p, s["answers"], s.get("scale", ""))
            em += e
            f1 += f
        n = max(len(usable), 1)
        return 100.0 * em / n, 100.0 * f1 / n

    bEM, bF1 = score(base_preds)
    oEM, oF1 = score(ours_preds)
    print("=" * 72)
    print(f"RUNG 1  TAT-DQA dev  N={len(usable)}  reader=PaliGemma-3B (zero-shot)")
    print(f"  Baseline (page only)        : EM {bEM:5.2f} | F1 {bF1:5.2f}")
    print(f"  Ours     (page + table text): EM {oEM:5.2f} | F1 {oF1:5.2f}")
    print(f"  delta                       : EM {oEM-bEM:+5.2f} | F1 {oF1-bF1:+5.2f}")
    print("=" * 72)
    print("Sample predictions (eyeball structure vs no-structure):")
    for s, bp, op in list(zip(usable, base_preds, ours_preds))[:8]:
        print(f"Q: {s['question'][:80]}")
        print(f"   gold={s['answers']} scale={s.get('scale','')!r}  type={s['answer_type']}")
        print(f"   base={bp!r}")
        print(f"   ours={op!r}")


if __name__ == "__main__":
    main()
