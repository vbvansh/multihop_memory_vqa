"""
FET step 2: train the Failure-Aware Evidence Transformation predictor and evaluate
the policies it competes against.

Because fet_make_labels.py stored BOTH scores per sample (score_full, score_focus),
we can score every policy on the held-out split without re-running the reader:

    NEVER  (always READ_FULL)   = mean(score_full)          <- plain reader baseline
    ALWAYS (always FOCUS)       = mean(score_focus)         <- TALENT-style always-transform
    ORACLE (perfect prediction) = mean(max(full, focus))    <- ceiling
    FET    (our prediction)     = mean(score chosen by FET)

Also reports FET's prediction accuracy and how often it invokes the costly transform
(the efficiency number).

    python experiments/fet_train.py --epochs 60
    python experiments/fet_train.py --files experiments/fet_data/tatdqa_train.jsonl \
                                            experiments/fet_data/dude_val.jsonl
"""
import os
import sys
import glob
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from models.router.fet import FET

HERE = os.path.dirname(os.path.abspath(__file__))


def load_rows(files):
    rows = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=None, help="jsonl files (default: all in fet_data/)")
    ap.add_argument("--out", default=os.path.join(HERE, "fet_checkpoint.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    files = args.files or sorted(glob.glob(os.path.join(HERE, "fet_data", "*.jsonl")))
    rows = load_rows(files)
    if not rows:
        print("No data — run experiments/fet_make_labels.py first.")
        return
    print(f"Loaded {len(rows)} samples from {len(files)} file(s): "
          f"{[os.path.basename(f) for f in files]}")
    by_ds = {}
    for r in rows:
        by_ds.setdefault(r["dataset"], []).append(r)
    for d, rs in by_ds.items():
        pos = sum(r["label"] for r in rs)
        print(f"  {d:10s} n={len(rs):5d}  FOCUS-helps {pos} ({100.0*pos/len(rs):.1f}%)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.shuffle(rows)
    n_val = max(1, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]
    print(f"train {len(train_rows)} | val {len(val_rows)}")

    def tensors(rs):
        q = torch.tensor([r["q_feat"] for r in rs], dtype=torch.float32)
        p = torch.tensor([r["p_feat"] for r in rs], dtype=torch.float32)
        y = torch.tensor([r["label"] for r in rs], dtype=torch.long)
        sf = torch.tensor([r["score_full"] for r in rs], dtype=torch.float32)
        so = torch.tensor([r["score_focus"] for r in rs], dtype=torch.float32)
        return q, p, y, sf, so

    qtr, ptr, ytr, _, _ = tensors(train_rows)
    qva, pva, yva, sf_va, so_va = tensors(val_rows)

    emb_dim = qtr.shape[1]
    model = FET(emb_dim=emb_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # class weights: FOCUS is usually the minority class
    cnt = torch.bincount(ytr, minlength=2).float().clamp(min=1)
    lossf = nn.CrossEntropyLoss(weight=(cnt.sum() / cnt))

    best_acc, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(ytr))
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            opt.zero_grad()
            loss = lossf(model(qtr[idx], ptr[idx]), ytr[idx])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            pred = model(qva, pva).argmax(dim=1)
        acc = (pred == yva).float().mean().item()
        if acc > best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if ep % 10 == 0 or ep == 1:
            print(f"  epoch {ep:3d}  loss {tot/max(len(ytr),1):.4f}  val-acc {100*acc:.2f}%")

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "emb_dim": emb_dim}, args.out)
    model.eval()

    with torch.no_grad():
        focus = model.decide(qva, pva, threshold=args.threshold)
    chosen = torch.where(focus, so_va, sf_va)

    never = sf_va.mean().item() * 100
    always = so_va.mean().item() * 100
    oracle = torch.maximum(sf_va, so_va).mean().item() * 100
    fet = chosen.mean().item() * 100
    invoke = focus.float().mean().item() * 100
    pred = model(qva, pva).argmax(dim=1)
    acc = (pred == yva).float().mean().item() * 100
    tp = ((pred == 1) & (yva == 1)).sum().item()
    fp = ((pred == 1) & (yva == 0)).sum().item()
    fn = ((pred == 0) & (yva == 1)).sum().item()
    prec = 100.0 * tp / max(tp + fp, 1)
    rec = 100.0 * tp / max(tp + fn, 1)

    print("=" * 74)
    print(f"FET  (held-out n={len(val_rows)}, emb_dim={emb_dim})")
    print(f"  prediction accuracy : {acc:.2f}%   (FOCUS precision {prec:.1f}% / recall {rec:.1f}%)")
    print(f"  transform invoked on: {invoke:.1f}% of questions   <- efficiency")
    print("-" * 74)
    print(f"  NEVER  transform (plain reader)      : {never:.2f}")
    print(f"  ALWAYS transform (TALENT-style)      : {always:.2f}")
    print(f"  FET    (ours)                        : {fet:.2f}   "
          f"(vs never {fet-never:+.2f}, vs always {fet-always:+.2f})")
    print(f"  ORACLE (perfect prediction, ceiling) : {oracle:.2f}")
    print("=" * 74)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
