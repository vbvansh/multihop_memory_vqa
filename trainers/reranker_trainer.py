import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from tqdm import tqdm

from models.reranker.page_reranker import PageReranker
from datasets.docvqa import DocVQADataset
from utils.logger import setup_logger


def maxsim_scores(query_emb, page_embs):
    """query_emb [Lq,D], page_embs [P,N,D] -> [P] ColPali late-interaction score."""
    sims = torch.einsum("qd,pnd->pqn", query_emb, page_embs)   # [P, Lq, N]
    return sims.max(dim=2).values.sum(dim=1)                    # [P]


class RerankerTrainer:
    """
    Trains the MaxSim-anchored PageReranker and reports its page-selection
    accuracy HEAD-TO-HEAD against the pure MaxSim baseline on the same samples.
    Features (per-page MaxSim, mean page embedding, mean query embedding) are
    computed once from the precomputed ColPali embeddings and cached in RAM,
    so training is fast (no per-epoch disk I/O, no big model).
    """

    def __init__(self, model_config_path, train_config_path):
        with open(model_config_path) as f:
            model_config = yaml.safe_load(f)
        with open(train_config_path) as f:
            train_config = yaml.safe_load(f)
        self.config = {**model_config, **train_config}

        self.rc = self.config.get("reranker_training", {}) or {}
        self.logs_dir = self.config["paths"].get("logs_dir", "./logs")
        self.checkpoints_dir = self.config["paths"].get("checkpoints_dir", "./checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        self.logger = setup_logger(name="Reranker", log_dir=self.logs_dir)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.precomputed_dir = self.config["debug"]["precomputed_dir"]
        self.D = self.config["model"]["embedding_dim"]

        self.reranker = PageReranker(self.config).to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.reranker.parameters(),
            lr=float(self.rc.get("lr", 1e-4)),
            weight_decay=float(self.rc.get("weight_decay", 0.01)),
        )
        self.grad_clip = float(self.rc.get("grad_clip", 1.0))

        self.logger.info("Precomputing + caching reranker features (one-time)...")
        self.train_feats = self._build_features("train")
        self.val_feats = self._build_features("val")

    # ---------------- feature precompute (cached) ----------------

    def _build_features(self, split):
        ds = DocVQADataset(self.config, split=split, debug=False)
        feats = []
        missing = 0
        for s in tqdm(ds.samples, desc=f"features[{split}]"):
            qid = s["question_id"]
            vpath = os.path.join(self.precomputed_dir, f"vision_{qid}.pt")
            qpath = os.path.join(self.precomputed_dir, f"question_{qid}.pt")
            if not (os.path.exists(vpath) and os.path.exists(qpath)):
                missing += 1
                continue
            page_embs = torch.load(vpath, map_location=self.device).float()   # [P,N,D]
            query_emb = torch.load(qpath, map_location=self.device).float()
            if query_emb.dim() == 3:
                query_emb = query_emb.squeeze(0)                             # [Lq,D]

            P = page_embs.shape[0]
            ms = maxsim_scores(query_emb, page_embs)                         # [P]
            ms = (ms - ms.mean()) / (ms.std() + 1e-6)                        # z-score per sample
            page_mean = page_embs.mean(dim=1)                               # [P,D]
            q_mean = query_emb.mean(dim=0)                                  # [D]
            gt = max(0, min(int(s["answer_page_idx"]), P - 1))

            feats.append({
                "page_mean": page_mean.cpu(),
                "q_mean": q_mean.cpu(),
                "maxsim": ms.cpu(),
                "gt": gt,
                "P": P,
            })
        self.logger.info(f"[{split}] cached {len(feats)} samples (skipped {missing} missing).")
        return feats

    def _make_batch(self, batch_feats):
        B = len(batch_feats)
        maxP = max(f["P"] for f in batch_feats)
        page_mean = torch.zeros(B, maxP, self.D)
        maxsim = torch.zeros(B, maxP)
        mask = torch.ones(B, maxP, dtype=torch.bool)
        q_mean = torch.zeros(B, self.D)
        gt = torch.zeros(B, dtype=torch.long)
        for i, f in enumerate(batch_feats):
            P = f["P"]
            page_mean[i, :P] = f["page_mean"]
            maxsim[i, :P] = f["maxsim"]
            mask[i, :P] = False
            q_mean[i] = f["q_mean"]
            gt[i] = f["gt"]
        return (page_mean.to(self.device), q_mean.to(self.device),
                maxsim.to(self.device), mask.to(self.device), gt.to(self.device))

    def _iter_batches(self, feats, batch_size, shuffle):
        idx = list(range(len(feats)))
        if shuffle:
            random.shuffle(idx)
        for i in range(0, len(idx), batch_size):
            yield self._make_batch([feats[j] for j in idx[i:i + batch_size]])

    # ---------------- train / eval ----------------

    def train_epoch(self, epoch):
        self.reranker.train()
        bs = int(self.rc.get("batch_size", 16))
        total_loss, n = 0.0, 0
        loop = tqdm(self._iter_batches(self.train_feats, bs, shuffle=True),
                    total=(len(self.train_feats) + bs - 1) // bs,
                    desc=f"Reranker epoch {epoch+1}")
        for page_mean, q_mean, maxsim, mask, gt in loop:
            self.optimizer.zero_grad()
            logits = self.reranker(page_mean, q_mean, maxsim, mask)
            loss = self.loss_fn(logits, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.reranker.parameters(), self.grad_clip)
            self.optimizer.step()
            total_loss += loss.item()
            n += 1
            loop.set_postfix(loss=loss.item())
        return total_loss / max(n, 1)

    @torch.no_grad()
    def evaluate(self, feats, name="Val"):
        self.reranker.eval()
        bs = int(self.rc.get("batch_size", 16))
        # counters: reranker vs maxsim, overall / single / multi
        r_ok = m_ok = tot = 0
        r_ok_mp = m_ok_mp = tot_mp = 0
        r_ok_sp = tot_sp = 0
        for page_mean, q_mean, maxsim, mask, gt in self._iter_batches(feats, bs, shuffle=False):
            logits = self.reranker(page_mean, q_mean, maxsim, mask)
            r_pred = logits.argmax(dim=1)
            m_pred = maxsim.masked_fill(mask, -1e9).argmax(dim=1)   # pure MaxSim baseline
            real_pages = (~mask).sum(dim=1)                        # [B]
            for i in range(gt.shape[0]):
                correct_r = int(r_pred[i] == gt[i])
                correct_m = int(m_pred[i] == gt[i])
                r_ok += correct_r; m_ok += correct_m; tot += 1
                if real_pages[i].item() == 1:
                    r_ok_sp += correct_r; tot_sp += 1
                else:
                    r_ok_mp += correct_r; m_ok_mp += correct_m; tot_mp += 1

        def pct(a, b):
            return 100.0 * a / b if b else 0.0

        self.logger.info("-" * 70)
        self.logger.info(f"[{name}] Reranker top-1: {pct(r_ok,tot):.2f}%   | MaxSim top-1: {pct(m_ok,tot):.2f}%   (N={tot})")
        self.logger.info(f"[{name}] MULTI-PAGE  ->  Reranker: {pct(r_ok_mp,tot_mp):.2f}%  | MaxSim: {pct(m_ok_mp,tot_mp):.2f}%  (N={tot_mp})   <-- the head-to-head")
        self.logger.info(f"[{name}] single-page ->  Reranker: {pct(r_ok_sp,tot_sp):.2f}%  (N={tot_sp})")
        self.logger.info("-" * 70)
        return pct(r_ok, tot), pct(r_ok_mp, tot_mp)

    def run(self):
        epochs = int(self.rc.get("epochs", 50))
        self.logger.info("=== Baseline (epoch 0): pure MaxSim vs untrained reranker ===")
        self.evaluate(self.val_feats, name="Val@0")

        best = -1.0
        for epoch in range(epochs):
            loss = self.train_epoch(epoch)
            self.logger.info(f"[Reranker] Epoch {epoch+1}/{epochs} | train loss {loss:.4f}")
            acc, acc_mp = self.evaluate(self.val_feats, name="Val")
            if acc > best:
                best = acc
                torch.save(self.reranker.state_dict(),
                           os.path.join(self.checkpoints_dir, "reranker_best.pt"))
                self.logger.info(f"Saved best reranker (val top-1 {acc:.2f}%).")
        self.logger.info(f"Done. Best val top-1: {best:.2f}%")
