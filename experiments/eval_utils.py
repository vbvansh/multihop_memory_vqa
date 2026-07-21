"""
Approximate EM / F1 for TAT-DQA (proof-of-concept only).

TAT-DQA's official metric is a numeracy-focused F1 with scale handling; use their
official script for final paper numbers. This lightweight version (DROP-style token
overlap) is enough to COMPARE two conditions with the same reader.
"""
import re
from collections import Counter

_ARTICLES = {"a", "an", "the", "of", "in", "and"}


def normalize(text):
    """Lowercase, strip currency/commas, drop punctuation (keep % . -), tokenize."""
    text = str(text).lower()
    text = text.replace("$", " ").replace(",", "")
    text = re.sub(r"[^\w\s.%-]", " ", text)
    toks = [t for t in text.split() if t and t not in _ARTICLES]
    return toks


def _f1(pred_toks, gold_toks):
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = Counter(pred_toks) & Counter(gold_toks)
    n = sum(common.values())
    if n == 0:
        return 0.0
    prec = n / len(pred_toks)
    rec = n / len(gold_toks)
    return 2 * prec * rec / (prec + rec)


def _to_list(x):
    """TAT-DQA 'answer' is usually a list but can be a bare number/str."""
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def evaluate(pred, gold_list, scale=""):
    """
    pred: model output string.
    gold_list: gold answer span(s) (TAT-DQA 'answer' — list, or a bare number/str).
    scale: optional unit (e.g. 'percent', 'million').
    Returns (em, f1) in [0,1].
    """
    gold_list = _to_list(gold_list)
    scale = str(scale) if scale else ""
    gold_join = " ".join([str(x) for x in gold_list] + ([scale] if scale else []))
    gold_toks = normalize(gold_join)
    pred_toks = normalize(pred)
    f1 = _f1(pred_toks, gold_toks)
    em = 1.0 if pred_toks and set(pred_toks) == set(gold_toks) else 0.0
    return em, f1
