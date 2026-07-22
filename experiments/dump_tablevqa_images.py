"""
Dump TableVQA-Bench table images to PNG files so MinerU (separate env) can parse them.

RUN IN THE .doc ENV (has pandas). Writes experiments/tablevqa_images/<qa_id>.png.
Then run experiments/mineru_parse_tablevqa.py in the mineru env on that folder.

    python experiments/dump_tablevqa_images.py --subsets vwtq fintabnetqa
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datasets.tablevqa import TableVQADataset

DEFAULT_DIR = "/c/ujjwalb/Vansh/Datasets/TableVQA-Bench/data"
HERE = os.path.dirname(os.path.abspath(__file__))


def safe(qid):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in qid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default=DEFAULT_DIR)
    ap.add_argument("--out_dir", default=os.path.join(HERE, "tablevqa_images"))
    ap.add_argument("--subsets", nargs="*", default=["vwtq", "fintabnetqa"])
    ap.add_argument("--max_samples", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    ds = TableVQADataset(args.data_dir, subsets=args.subsets)
    samples = ds.samples[:args.max_samples] if args.max_samples else ds.samples
    print(f"{len(samples)} samples from subsets {args.subsets}")

    done = skip = 0
    for s in samples:
        p = os.path.join(args.out_dir, safe(s["qa_id"]) + ".png")
        if os.path.exists(p):
            skip += 1
            continue
        try:
            ds.image(s).save(p)
            done += 1
        except Exception as e:
            print(f"  [fail] {s['qa_id']}: {e}")
    print(f"FINISHED. wrote {done}, skipped {skip}. images -> {args.out_dir}")


if __name__ == "__main__":
    main()
