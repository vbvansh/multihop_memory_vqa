"""
Fetch IndicDLP from Hugging Face WITHOUT downloading the ~300GB of images.

Stage 1 (now)   : list the repo, then download ONLY annotation/metadata files
                  (json/jsonl/csv/parquet/txt) -> enough for the full layout analysis.
Stage 2 (later) : download images only for the documents you actually select.

Note: this uses `huggingface_hub` (not the `datasets` library) because this repo has a
local `datasets/` package that shadows it.

Usage:
    # 1) see what's in the repo and how big (no download):
    python Benchmark_Multilingual/fetch_indicdlp.py --repo <org/IndicDLP> --list

    # 2) download just the annotations:
    python Benchmark_Multilingual/fetch_indicdlp.py --repo <org/IndicDLP> \
        --out /c/ujjwalb/Vansh/Datasets/IndicDLP_meta --annotations

    # 3) later: specific image files only
    python Benchmark_Multilingual/fetch_indicdlp.py --repo <org/IndicDLP> \
        --out /c/ujjwalb/Vansh/Datasets/IndicDLP_imgs --files path/a.jpg path/b.jpg
"""
import os
import argparse
from collections import Counter

ANNOT_EXT = (".json", ".jsonl", ".csv", ".tsv", ".parquet", ".txt", ".yaml", ".yml", ".md")
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="HF repo id, e.g. 'ihub-iiith/IndicDLP'")
    ap.add_argument("--repo_type", default="dataset", choices=["dataset", "model"])
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true", help="list repo files + size summary, download nothing")
    ap.add_argument("--annotations", action="store_true", help="download only annotation/metadata files")
    ap.add_argument("--files", nargs="*", default=None, help="download these exact repo paths")
    ap.add_argument("--pattern", nargs="*", default=None, help="download files matching these glob patterns")
    args = ap.parse_args()

    from huggingface_hub import HfApi, snapshot_download, hf_hub_download

    api = HfApi()

    if args.list:
        info = api.repo_info(args.repo, repo_type=args.repo_type, files_metadata=True)
        files = info.siblings or []
        print(f"{len(files)} files in {args.repo}\n")
        ext_n, ext_bytes, top = Counter(), Counter(), Counter()
        for f in files:
            ext = os.path.splitext(f.rfilename)[1].lower()
            size = getattr(f, "size", None) or 0
            ext_n[ext] += 1
            ext_bytes[ext] += size
            top[f.rfilename.split("/")[0]] += 1
        print(f"{'ext':12s} {'files':>8s} {'size':>12s}")
        for e, n in ext_n.most_common(20):
            gb = ext_bytes[e] / 1e9
            print(f"{e or '(none)':12s} {n:>8,} {gb:>10.2f} GB")
        print("\ntop-level entries:")
        for t, n in top.most_common(25):
            print(f"  {t:50s} {n:,} files")
        annot = sum(ext_bytes[e] for e in ANNOT_EXT)
        print(f"\nANNOTATION-ONLY download would be ~{annot/1e9:.2f} GB "
              f"(vs {sum(ext_bytes.values())/1e9:.1f} GB total)")
        print("\nNext: re-run with --annotations --out <dir>")
        return

    if not args.out:
        print("--out is required for downloads")
        return
    os.makedirs(args.out, exist_ok=True)

    if args.files:
        for rp in args.files:
            p = hf_hub_download(args.repo, rp, repo_type=args.repo_type, local_dir=args.out)
            print("got", p)
        return

    if args.annotations:
        patterns = [f"**/*{e}" for e in ANNOT_EXT]
    elif args.pattern:
        patterns = args.pattern
    else:
        print("choose --list, --annotations, --pattern, or --files")
        return

    print(f"downloading patterns {patterns} -> {args.out}")
    path = snapshot_download(args.repo, repo_type=args.repo_type, local_dir=args.out,
                             allow_patterns=patterns, ignore_patterns=[f"**/*{e}" for e in IMAGE_EXT])
    print("done ->", path)


if __name__ == "__main__":
    main()
