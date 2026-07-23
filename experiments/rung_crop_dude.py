"""
DUDE focus test: does cropping to the ANSWER REGION (high-res focus) help the reader
on dense full-page documents? Uses the gold answer bounding box (oracle focus ceiling).

Reader = Qwen2.5-VL (frozen). Localizable (extractive) val subset.
  Baseline : full page image
  Crop     : image cropped to the gold answer box (+ padding), then read
Metric: ANLS + EM.

This tests the "focus/readability" hypothesis (vs table structure) on a NON-table
dataset. No MinerU needed — the box comes from the annotations.

RUN IN THE .doc ENV:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/rung_crop_dude.py --max_samples 200 --reader_batch 2 --pad 0.6
"""
import os
import sys
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


def open_page(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return Image.new("RGB", (448, 448), color="white")


def thumb(img, maxside=1600):
    im = img.copy()
    if max(im.size) > maxside:
        im.thumbnail((maxside, maxside), Image.BILINEAR)
    return im


def crop_region(img, box, pad):
    W, H = img.size
    l, t, w, h = box["left"], box["top"], box["width"], box["height"]
    px, py = w * pad + 40, h * pad + 40           # pad by ratio + a fixed margin for context
    l0, t0 = max(0, int(l - px)), max(0, int(t - py))
    l1, t1 = min(W, int(l + w + px)), min(H, int(t + h + py))
    if l1 <= l0 or t1 <= t0:
        return img
    return img.crop((l0, t0, l1, t1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt_json", default=DEFAULT_GT)
    ap.add_argument("--images_root", default=DEFAULT_IMAGES)
    ap.add_argument("--split", default="val")
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--reader_batch", type=int, default=2)
    ap.add_argument("--pad", type=float, default=0.6, help="extra context around the box (fraction of box size)")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False
    config["reader"]["model_name"] = args.qwen_model
    config["reader"]["max_new_tokens"] = args.max_new_tokens

    ds = DUDEDataset(args.gt_json, args.images_root, split=args.split)
    usable = [s for s in ds.samples if s["gold_pages"] and s["gold_boxes"]]
    if args.max_samples:
        usable = usable[:args.max_samples]
    print(f"{len(usable)} usable samples (localizable + has box).")

    from models.reader.qwen_reader import QwenVLReader
    reader = QwenVLReader(config)

    base_imgs, crop_imgs, qs, golds = [], [], [], []
    for s in usable:
        pg = min(s["gold_pages"])
        box = next((b for b in s["gold_boxes"] if b["page"] == pg), s["gold_boxes"][0])
        page = open_page(ds.image_path(s["doc_id"], s["split"], pg))
        base_imgs.append(thumb(page, 1600))
        crop_imgs.append(thumb(crop_region(page, box, args.pad), 1600))
        qs.append(s["question"])
        golds.append(s["answers"] if s["answers"] else [""])

    def run(images, tag):
        preds = []
        for i in tqdm(range(0, len(images), args.reader_batch), desc=tag):
            b = slice(i, i + args.reader_batch)
            preds.extend(reader.generate(images[b], qs[b]))
        return preds

    base_preds = run(base_imgs, "baseline (full page)")
    crop_preds = run(crop_imgs, "crop (answer region)")

    def score(preds):
        a = e = 0.0
        for g, p in zip(golds, preds):
            a += calculate_anls(p, g)
            e += calculate_exact_match(p, g)
        n = max(len(golds), 1)
        return 100.0 * a / n, 100.0 * e / n

    bA, bE = score(base_preds)
    cA, cE = score(crop_preds)
    print("=" * 72)
    print(f"DUDE FOCUS TEST  {args.split}  N={len(usable)}  reader={args.qwen_model}  pad={args.pad}")
    print(f"  Baseline (full page)     : ANLS {bA:5.2f} | EM {bE:5.2f}")
    print(f"  Crop     (answer region) : ANLS {cA:5.2f} | EM {cE:5.2f}")
    print(f"  delta                    : ANLS {cA-bA:+5.2f} | EM {cE-bE:+5.2f}")
    print("=" * 72)
    for s, bp, cp in list(zip(usable, base_preds, crop_preds))[:8]:
        print(f"Q: {s['question'][:80]}")
        print(f"   gold={s['answers']}  base={bp!r}  crop={cp!r}")


if __name__ == "__main__":
    main()
