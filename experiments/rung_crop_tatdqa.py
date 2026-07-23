"""
TAT-DQA diagnostic: is the +26 F1 win from FOCUS/RESOLUTION or from table STRUCTURE?

Reader = Qwen2.5-VL (frozen). Same usable set (docs with a MinerU cache + crop).
FOUR conditions on the same questions:
  base      : full page @ 1600px            (what gave F1 ~8)
  hires     : full page @ 2048px            (does more resolution alone help?)
  crop      : MinerU's cropped TABLE IMAGE   (focus, no structure text)
  structure : full page + MinerU table text  (what gave the +26 win)

Reading:
  - if crop ~ structure   -> the win is FOCUS/READABILITY, not structure  -> reframe
  - if structure > crop    -> table structure adds something beyond focus
  - if hires ~ structure   -> it was just resolution (our baseline was under-resolved)

RUN IN THE .doc ENV (requires the TAT-DQA MinerU cache + crops, i.e. re-run
experiments/mineru_parse.py after the crop-saving update):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/rung_crop_tatdqa.py --max_samples 150 --reader_batch 2
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

Image.MAX_IMAGE_PIXELS = None
DEFAULT_JSON = "/c/ujjwalb/Vansh/Datasets/TATDQA/tatdqa_dataset_dev.json"
DEFAULT_DOCS = "/c/ujjwalb/Vansh/Datasets/TATDQA/tatdqa_docs_dev/dev"
HERE = os.path.dirname(os.path.abspath(__file__))


def open_img(path, maxside):
    try:
        im = Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (448, 448), color="white")
    if max(im.size) > maxside:
        im.thumbnail((maxside, maxside), Image.BILINEAR)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--docs_dir", default=DEFAULT_DOCS)
    ap.add_argument("--cache_dir", default=os.path.join(HERE, "mineru_cache"))
    ap.add_argument("--crops_dir", default=os.path.join(HERE, "tatdqa_crops"))
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_samples", type=int, default=150)
    ap.add_argument("--reader_batch", type=int, default=2)
    ap.add_argument("--max_table_chars", type=int, default=4000)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False
    config["reader"]["model_name"] = args.qwen_model
    config["reader"]["max_new_tokens"] = args.max_new_tokens

    def cache_path(uid):
        return os.path.join(args.cache_dir, uid + ".json")

    ds = TATDQADataset(args.json, args.docs_dir)
    usable = [s for s in ds.samples if s["question_id"] and os.path.exists(cache_path(s["doc_uid"]))]
    if args.max_samples:
        usable = usable[:args.max_samples]
    print(f"{len(usable)} usable samples.")

    from models.reader.qwen_reader import QwenVLReader
    reader = QwenVLReader(config)

    def table_text(uid):
        d = json.load(open(cache_path(uid), "r", encoding="utf-8"))
        return d.get("table_text", "") or ""

    def crop_path(uid):
        d = json.load(open(cache_path(uid), "r", encoding="utf-8"))
        c = d.get("crop")
        return os.path.join(args.crops_dir, c) if c else None

    qs = [s["question"] for s in usable]
    golds = [(s["answers"], s.get("scale", "")) for s in usable]
    base_imgs = [open_img(s["image_path"], 1600) for s in usable]
    hires_imgs = [open_img(s["image_path"], 2048) for s in usable]
    crop_imgs, n_crop = [], 0
    for s in usable:
        cp = crop_path(s["doc_uid"])
        if cp and os.path.exists(cp):
            crop_imgs.append(open_img(cp, 1600)); n_crop += 1
        else:
            crop_imgs.append(open_img(s["image_path"], 1600))   # fallback: full page
    struct_q = [f"{s['question']}\nTable:\n{table_text(s['doc_uid'])[:args.max_table_chars]}" for s in usable]
    print(f"table crops available for {n_crop}/{len(usable)} samples (rest fall back to full page).")

    def run(images, questions, tag):
        preds = []
        for i in tqdm(range(0, len(questions), args.reader_batch), desc=tag):
            b = slice(i, i + args.reader_batch)
            preds.extend(reader.generate(images[b], questions[b]))
        return preds

    preds = {
        "base": run(base_imgs, qs, "base (page@1600)"),
        "hires": run(hires_imgs, qs, "hires (page@2048)"),
        "crop": run(crop_imgs, qs, "crop (table image)"),
        "structure": run(base_imgs, struct_q, "structure (page+table text)"),
    }

    def score(ps):
        em = f1 = 0.0
        for (ans, scale), p in zip(golds, ps):
            e, f = evaluate(p, ans, scale)
            em += e; f1 += f
        n = max(len(golds), 1)
        return 100.0 * em / n, 100.0 * f1 / n

    print("=" * 72)
    print(f"TAT-DQA DIAGNOSTIC  dev  N={len(usable)}  reader={args.qwen_model}")
    for k in ["base", "hires", "crop", "structure"]:
        em, f1 = score(preds[k])
        print(f"  {k:10s} : EM {em:5.2f} | F1 {f1:5.2f}")
    print("=" * 72)
    for i, s in enumerate(usable[:8]):
        print(f"Q: {s['question'][:75]}  gold={s['answers']}")
        print(f"   base={preds['base'][i]!r} | crop={preds['crop'][i]!r} | struct={preds['structure'][i]!r}")


if __name__ == "__main__":
    main()
