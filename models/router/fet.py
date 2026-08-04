"""
FET — Failure-Aware Evidence Transformation.

A small predictor that answers, BEFORE the reader runs:
    "will the frozen reader fail on this (question, page) if given the raw page,
     such that transforming the evidence is worth its cost?"

Input : mean-pooled ColPali question embedding + page embedding  [2D]
Output: 2 logits -> {0: READ_FULL, 1: FOCUS (transform the evidence)}

Trained on OBSERVED reader outcomes (see experiments/fet_make_labels.py): at training
time we run both modes and record which one won; FET learns to predict that from cheap
features, so at inference no reader pass (and no expensive parse) is wasted.
"""
import torch
import torch.nn as nn


class FET(nn.Module):
    def __init__(self, emb_dim=128, hidden=256, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * emb_dim),
            nn.Linear(2 * emb_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        # start near "READ_FULL" (the safe, cheap default) so an untrained FET
        # degrades to the plain-reader baseline rather than transforming everything.
        nn.init.zeros_(self.net[-1].weight)
        with torch.no_grad():
            self.net[-1].bias.copy_(torch.tensor([0.5, -0.5]))

    def forward(self, q_feat, p_feat):
        return self.net(torch.cat([q_feat, p_feat], dim=-1))

    @torch.no_grad()
    def decide(self, q_feat, p_feat, threshold=0.5):
        """Return a bool tensor: True = FOCUS (transform), False = READ_FULL."""
        prob = torch.softmax(self(q_feat, p_feat), dim=-1)[:, 1]
        return prob > threshold
