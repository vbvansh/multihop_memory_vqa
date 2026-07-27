"""
InfographicVQA focus test: does cropping to the answer region help on dense
infographics where the reader struggles with the full image?

Reader = Qwen2.5-VL (frozen). Val subset. The answer region is located from the OCR
(oracle focus) since InfographicVQA has no gold answer box.
  Baseline : full infographic image
  Crop     : image cropped to the answer region (+ context padding)
Metric: ANLS + EM.

This is the breadth check for the "focusing helps dense documents" claim on a
NON-table, standard benchmark.

RUN IN THE .doc ENV (no MinerU needed):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/rung_crop_infographicvqa.py --max_samples 300 --reader_batch 2 --pad 1.0
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from PIL import Image
from tqdm import tqdm

from datasets.infographicvqa import InfographicVQADataset, load_ocr_boxes, answer_box
from utils.metrics import calculate_anls, calculate_exact_match

Image.MAX_IMAGE_PIXELS = None
BASE = "/c/ujjwalb/Vansh/Datasets/InfographicsVQA"
DEFAULT_QAS = os.path.join(BASE, "infographicsvqa_qas", "infographicsVQA_val_v1.0_withQT.json")
DEFAULT_IMAGES = os.path.join(BASE, "infographicsvqa_images")
DEFAULT_OCR = os.path.join(BASE, "infographicsvqa_ocr")


def open_img(path):
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
    l, t = box["left"] * W, box["top"] * H
    w, h = box["width"] * W, box["height"] * H
    px, py = w * pad + 40, h * pad + 40
    l0, t0 = max(0, int(l - px)), max(0, int(t - py))
    l1, t1 = min(W, int(l + w + px)), min(H, int(t + h + py))
    if l1 <= l0 or t1 <= t0:
        return img
    return img.crop((l0, t0, l1, t1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qas_json", default=DEFAULT_QAS)
    ap.add_argument("--images_dir", default=DEFAULT_IMAGES)
    ap.add_argument("--ocr_dir", default=DEFAULT_OCR)
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_samples", type=int, default=300)
    ap.add_argument("--reader_batch", type=int, default=2)
    ap.add_argument("--pad", type=float, default=1.0, help="context around the answer box (fraction of box size)")
    ap.add_argument("--max_new_tokens", type=int, default=48)
    args = ap.parse_args()

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False
    config["reader"]["model_name"] = args.qwen_model
    config["reader"]["max_new_tokens"] = args.max_new_tokens

    ds = InfographicVQADataset(args.qas_json, args.images_dir, args.ocr_dir)

    # keep only samples where we can locate the answer region in the OCR (so we can crop)
    usable, n_seen, n_nobox = [], 0, 0
    for s in ds.samples:
        if args.max_samples and len(usable) >= args.max_samples:
            break
        n_seen += 1
        boxes = load_ocr_boxes(s["ocr_path"])
        box = answer_box(boxes, s["answers"])
        if box is None:
            n_nobox += 1
            continue
        s["_box"] = box
        usable.append(s)
    print(f"{len(usable)} usable samples (answer located in OCR); scanned {n_seen}, no-box {n_nobox}.")
    if not usable:
        print("No answer boxes located — check OCR format / paths.")
        return

    from models.reader.qwen_reader import QwenVLReader
    reader = QwenVLReader(config)

    base_imgs, crop_imgs, qs, golds = [], [], [], []
    for s in usable:
        img = open_img(s["image_path"])
        base_imgs.append(thumb(img, 1600))
        crop_imgs.append(thumb(crop_region(img, s["_box"], args.pad), 1600))
        qs.append(s["question"])
        golds.append(s["answers"] if s["answers"] else [""])

    def run(images, tag):
        preds = []
        for i in tqdm(range(0, len(images), args.reader_batch), desc=tag):
            b = slice(i, i + args.reader_batch)
            preds.extend(reader.generate(images[b], qs[b]))
        return preds

    base_preds = run(base_imgs, "baseline (full image)")
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
    print(f"INFOGRAPHICVQA FOCUS TEST  val  N={len(usable)}  reader={args.qwen_model}  pad={args.pad}")
    print(f"  Baseline (full image)    : ANLS {bA:5.2f} | EM {bE:5.2f}")
    print(f"  Crop     (answer region) : ANLS {cA:5.2f} | EM {cE:5.2f}")
    print(f"  delta                    : ANLS {cA-bA:+5.2f} | EM {cE-bE:+5.2f}")
    print("=" * 72)
    for s, bp, cp in list(zip(usable, base_preds, crop_preds))[:8]:
        print(f"Q: {s['question'][:80]}")
        print(f"   gold={s['answers']}  base={bp!r}  crop={cp!r}")


if __name__ == "__main__":
    main()
