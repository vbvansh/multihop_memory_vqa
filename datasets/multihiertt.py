"""
MultiHiertt loader (numerical reasoning over multi-hierarchical tables + text).

NOTE: MultiHiertt has NO page images — it is text + HTML tables + program
supervision. Used as an Option-C "reasoning-only" generalization test (does a
structure-preserving table representation help the reader?), not the visual pipeline.

dev.json = list of docs, each:
    uid, paragraphs[str], tables[str(HTML)], table_description{"T-R-C": str},
    qa{question, answer(str|float), table_evidence["T-R-C"], text_evidence[int],
       program, question_type("span_selection"|"arithmetic")}
One QA per doc entry.
"""
import re
import json


def relevant_table_indices(table_evidence):
    """From ['0-3-1','0-5-2','1-0-0'] -> [0, 1] (unique table indices)."""
    idxs = []
    for key in table_evidence or []:
        parts = str(key).split("-")
        if parts and parts[0].lstrip("-").isdigit():
            idxs.append(int(parts[0]))
    return sorted(set(idxs))


def flatten_html(html):
    """Structure-FREE linearization: cell values in reading order, no row/col alignment."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.S | re.I)
    out = []
    for r in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", r, re.S | re.I)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        cells = [c for c in cells if c]
        if cells:
            out.append(" ".join(cells))
    return " ; ".join(out)


class MultiHierttDataset:
    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.samples = []
        for d in data:
            qa = d.get("qa") or {}
            if not qa.get("question"):
                continue
            self.samples.append({
                "uid": d.get("uid"),
                "question": qa.get("question", ""),
                "answer": qa.get("answer"),
                "question_type": qa.get("question_type", ""),
                "program": qa.get("program", "") or "",
                "table_evidence": qa.get("table_evidence", []) or [],
                "text_evidence": qa.get("text_evidence", []) or [],
                "tables": d.get("tables", []) or [],
                "paragraphs": d.get("paragraphs", []) or [],
            })

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
