"""
MultiHiertt (Option-C generalization test): does a structure-preserving table
representation help the reader reason, vs a flattened one?

No images (MultiHiertt is text+HTML). Reader = Qwen2.5-VL in TEXT-ONLY mode, given
the ORACLE evidence (the relevant table(s) via table_evidence + paragraphs via
text_evidence). Two conditions:
  Baseline : evidence table(s) as FLAT text (HTML stripped, structure lost)
  Ours     : evidence table(s) as structure-preserving HTML
Metric: EM / F1, reported overall and split by question_type (span vs arithmetic).

RUN IN THE .doc ENV:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python experiments/rung_multihiertt.py --max_samples 200 --reader_batch 2
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import yaml
from tqdm import tqdm

from datasets.multihiertt import MultiHierttDataset, relevant_table_indices, flatten_html
from experiments.eval_utils import evaluate_multihiertt
from experiments.pot import POT_SYSTEM, parse_pot

DEFAULT_JSON = "/c/ujjwalb/Vansh/Datasets/MultiHiertt/dev.json"


def build_evidence(s, structured, max_chars):
    """Return the prompt for one sample. structured=True -> HTML tables; False -> flat."""
    rel = relevant_table_indices(s["table_evidence"])
    if not rel:
        rel = list(range(min(2, len(s["tables"]))))   # fallback: first couple tables
    tabs = []
    for t in rel:
        if 0 <= t < len(s["tables"]):
            html = s["tables"][t]
            tabs.append(html if structured else flatten_html(html))
    table_block = "\n".join(tabs)

    paras = []
    for i in s["text_evidence"]:
        if isinstance(i, int) and 0 <= i < len(s["paragraphs"]):
            paras.append(s["paragraphs"][i])
    text_block = "\n".join(paras)

    parts = []
    if text_block:
        parts.append("Text:\n" + text_block)
    if table_block:
        parts.append("Table:\n" + table_block)
    parts.append("Question: " + s["question"])
    return ("\n\n".join(parts))[:max_chars]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=DEFAULT_JSON)
    ap.add_argument("--model_config", default="./configs/model.yaml")
    ap.add_argument("--qwen_model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max_samples", type=int, default=200)
    ap.add_argument("--reader_batch", type=int, default=2)
    ap.add_argument("--max_chars", type=int, default=7000)
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--pot", action="store_true",
                    help="Program-of-Thought: model emits an arithmetic expression, we execute it")
    args = ap.parse_args()

    with open(args.model_config) as f:
        config = yaml.safe_load(f)
    config.setdefault("reader", {})
    config["reader"]["use_lora"] = False
    config["reader"]["model_name"] = args.qwen_model
    # PoT needs room to reason before the final ANSWER = line
    config["reader"]["max_new_tokens"] = max(args.max_new_tokens, 256) if args.pot else args.max_new_tokens

    ds = MultiHierttDataset(args.json)
    usable = ds.samples[:args.max_samples] if args.max_samples else ds.samples
    print(f"{len(usable)} samples (of {len(ds.samples)}).")

    from models.reader.qwen_reader import QwenVLReader
    reader = QwenVLReader(config)

    base_p = [build_evidence(s, structured=False, max_chars=args.max_chars) for s in usable]
    ours_p = [build_evidence(s, structured=True, max_chars=args.max_chars) for s in usable]

    def run(prompts, tag):
        preds = []
        for i in tqdm(range(0, len(prompts), args.reader_batch), desc=tag):
            batch = prompts[i:i + args.reader_batch]
            if args.pot:
                out = reader.generate_text(batch, system=POT_SYSTEM)
                preds.extend(parse_pot(o) for o in out)
            else:
                preds.extend(reader.generate_text(batch))
        return preds

    base_preds = run(base_p, "baseline (flat table)")
    ours_preds = run(ours_p, "ours (HTML table)")

    def score(preds, subset=None):
        em = f1 = 0.0
        n = 0
        for s, p in zip(usable, preds):
            if subset and s["question_type"] != subset:
                continue
            e, f = evaluate_multihiertt(p, s["answer"], s["question_type"])
            em += e
            f1 += f
            n += 1
        n = max(n, 1)
        return 100.0 * em / n, 100.0 * f1 / n, n

    print("=" * 74)
    mode = "PoT (execute expression)" if args.pot else "direct answer"
    print(f"MultiHiertt dev  N={len(usable)}  reader={args.qwen_model} (text-only, oracle evidence, {mode})")
    for subset, label in [(None, "ALL"), ("span_selection", "span"), ("arithmetic", "arithmetic")]:
        bE, bF, n = score(base_preds, subset)
        oE, oF, _ = score(ours_preds, subset)
        print(f"  [{label:10s} n={n:4d}]  Baseline(flat) EM {bE:5.2f} F1 {bF:5.2f}  |  "
              f"Ours(HTML) EM {oE:5.2f} F1 {oF:5.2f}  |  dEM {oE-bE:+5.2f} dF1 {oF-bF:+5.2f}")
    print("=" * 74)
    for s, bp, op in list(zip(usable, base_preds, ours_preds))[:8]:
        print(f"Q[{s['question_type']}]: {s['question'][:80]}")
        print(f"   gold={s['answer']}  base={bp!r}  ours={op!r}")


if __name__ == "__main__":
    main()
