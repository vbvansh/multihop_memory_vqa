
"""
FET step 1: build the training set for the Failure-Aware Evidence Transformation module.

For each sample we run the frozen reader BOTH ways and record which one was correct:
    READ_FULL (0) : reader( full page image , question )
    FOCUS     (1) : reader( full page image , question + MinerU table text )
Label = the mode that scored higher (tie -> READ_FULL, the cheaper & safer default).
This is the observed-failure supervision: at train time we SEE where the reader fails.

Features saved alongside: mean-pooled ColPali question + page embeddings (cheap, and
available before the reader runs -> usable at inference).

Output: one jsonl per dataset in experiments/fet_data/<dataset>_<split>.jsonl
        {qid, dataset, q_feat[D], p_feat[D], score_full, score_focus, label}

RUN IN THE .doc ENV (needs the reader + precomputed embeddings + MinerU cache):
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/fet_make_labels.py --dataset tatdqa --split train --max_samples 800
    python experiments/fet_make_labels.py --dataset dude   --split val   --max_samples 400
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
import torch
from PIL import Image
from tqdm import tqdm

from experiments.eval_utils import evaluate
from utils.metrics import calculate_anls

Image.MAX_IMAGE_PIXELS = None
HERE = os.path.dirname(os.path.abspath(__file__))
TAT = "/c/ujjwalb/Vansh/Datasets/TATDQA"
DUDE_GT = "/c/ujjwalb/Vansh/Datasets/DUDE/data/2023-03-23_DUDE_gt_test_PUBLIC.json"
DUDE_IMG = ("/c/ujjwalb/Vansh/Datasets/DUDE/data/DUDE_train-val-test_binaries/"
            "DUDE_train-val-test_binaries/images")
DUDE_EMB = "/c/ujjwalb/Vansh/Datasets/DUDE/precomputed_embeddings"


def load_image(p, maxside=1600):
    try:
        im = Image.open(p).convert("RGB")
        if max(im.size) > maxside:
            im.thumbnail((maxside, maxside), Image.BILINEAR)
        return im
    except Exception:
        return Image.new("RGB", (448, 448), color="white")


def pooled(path):
    """Mean-pool a saved ColPali embedding [.., N, D] -> list[D]."""
    t = torch.load(path, map_location="cpu").float()
    while t.dim() > 2:
        t = t.reshape(-1, t.shape[-1])
    return t.mean(dim=0).tolist()


def collect_tatdqa(split, max_samples):
    """-> list of dicts: qid, question, image_path, gold, scale, table_text, q_emb, p_emb"""
    from datasets.tatdqa import TATDQADataset
    qa_json = os.path.join(TAT, f"tatdqa_dataset_{'test_gold' if split=='test' else split}.json")
    docs_dir = os.path.join(TAT, f"tatdqa_docs_{split}", split)
    emb_dir = os.path.join(TAT, "precomputed_embeddings")
    cache_dir = os.path.join(HERE, "mineru_cache")

    ds = TATDQADataset(qa_json, docs_dir)
    out = []
    for s in ds.samples:
        if max_samples and len(out) >= max_samples:
            break
        qid, uid = s["question_id"], s["doc_uid"]
        cache = os.path.join(cache_dir, uid + ".json")
        qe = os.path.join(emb_dir, f"question_{qid}.pt")
        pe = os.path.join(emb_dir, f"vision_{uid}.pt")
        if not (qid and os.path.exists(cache) and os.path.exists(qe) and os.path.exists(pe)):
            continue
        tt = json.load(open(cache, encoding="utf-8")).get("table_text", "") or ""
        if not tt:
            continue
        out.append({"qid": qid, "question": s["question"], "image_path": s["image_path"],
                    "gold": s["answers"], "scale": s.get("scale", ""), "table_text": tt,
                    "q_emb": pooled(qe), "p_emb": pooled(pe), "metric": "tatdqa"})
    return out


def collect_dude(split, max_samples):
    from datasets.dude import DUDEDataset
    ds = DUDEDataset(DUDE_GT, DUDE_IMG, split=split)
    cache_dir = os.path.join(HERE, "dude_cache")
    out = []
    for s in ds.samples:
        if max_samples and len(out) >= max_samples:
            break
        if not s["gold_pages"]:
            continue
        pg = min(s["gold_pages"])
        did, qid = s["doc_id"], s["question_id"]
        cache = os.path.join(cache_dir, f"{did}_{pg}.json")
        qe = os.path.join(DUDE_EMB, f"question_{qid}.pt")
        pe = os.path.join(DUDE_EMB, f"vision_{did}.pt")
        if not (os.path.exists(cache) and os.path.exists(qe) and os.path.exists(pe)):
            continue
        d = json.load(open(cache, encoding="utf-8"))
        tt = (d.get("markdown") or d.get("table_text") or "")
        if not tt:
            continue
        out.append({"qid": qid, "question": s["question"],
                    "image_path": ds.image_path(did, s["split"], pg),
                    "gold": s["answers"] or [""], "scale": "", "table_text": tt,
                    "q_emb": pooled(qe), "p_emb": pooled(pe), "metric": "anls"})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["tatdqa", "dude"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_samples", type=int, default=800)
    ap.add_argument("--reader_batch", type=int, default=2)
    ap.add_argument("--max_table_chars", type=int, default=4000)
    ap.add_argument("--max_new_tokens", type=int, default=48)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "fet_data"))
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False
    config["reader"]["model_name"] = args.qwen_model
    config["reader"]["max_new_tokens"] = args.max_new_tokens

    rows = (collect_tatdqa if args.dataset == "tatdqa" else collect_dude)(args.split, args.max_samples)
    print(f"[{args.dataset}/{args.split}] {len(rows)} samples with embeddings + MinerU cache.")
    if not rows:
        print("Nothing to do — check precomputed embeddings and the MinerU cache.")
        return

    from models.reader.qwen_reader import QwenVLReader
    reader = QwenVLReader(config)

    imgs = [load_image(r["image_path"]) for r in rows]
    q_full = [r["question"] for r in rows]
    q_focus = [f"{r['question']}\nTable:\n{r['table_text'][:args.max_table_chars]}" for r in rows]

    def run(qs, tag):
        preds = []
        for i in tqdm(range(0, len(qs), args.reader_batch), desc=tag):
            b = slice(i, i + args.reader_batch)
            preds.extend(reader.generate(imgs[b], qs[b]))
        return preds

    p_full = run(q_full, "READ_FULL")
    p_focus = run(q_focus, "FOCUS")

    def sc(r, pred):
        if r["metric"] == "tatdqa":
            em, f1 = evaluate(pred, r["gold"], r.get("scale", ""))
            return f1
        return calculate_anls(pred, r["gold"])

    out_path = os.path.join(args.out_dir, f"{args.dataset}_{args.split}.jsonl")
    n_focus = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r, pf, po in zip(rows, p_full, p_focus):
            s_full, s_focus = sc(r, pf), sc(r, po)
            label = 1 if s_focus > s_full else 0        # tie -> READ_FULL (cheaper, safer)
            n_focus += label
            f.write(json.dumps({
                "qid": r["qid"], "dataset": args.dataset, "split": args.split,
                "q_feat": r["q_emb"], "p_feat": r["p_emb"],
                "score_full": round(float(s_full), 4), "score_focus": round(float(s_focus), 4),
                "label": label}) + "\n")

    n = len(rows)
    print("=" * 70)
    print(f"Wrote {n} labelled samples -> {out_path}")
    print(f"  label=FOCUS (transformation helps): {n_focus} ({100.0*n_focus/n:.1f}%)")
    print(f"  label=READ_FULL                   : {n-n_focus} ({100.0*(n-n_focus)/n:.1f}%)")
    print(f"  mean score  READ_FULL {sum(sc(r,p) for r,p in zip(rows,p_full))/n:.4f}"
          f" | FOCUS {sum(sc(r,p) for r,p in zip(rows,p_focus))/n:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
