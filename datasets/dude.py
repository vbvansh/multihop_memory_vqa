"""
DUDE dataset loader (multi-page DocVQA).

Differences from MPDocVQA (datasets/docvqa.py):
  - One combined GT json for all splits: {"data": [ {QA}, ... ]}, filtered by data_split.
  - Evidence label comes from answers_page_bounding_boxes -> we derive the GOLD PAGE(S)
    (page field is 1-indexed -> 0-indexed). An answer may reference multiple pages
    (rare, 298 cases) -> gold_pages is a SET.
  - Vision embeddings are stored PER docId (vision_<docId>.pt), questions per qid.
  - Page images are flat files: images/<split>/<docId>_<page>.jpg  (page 0-indexed).

Only train/val QA have bounding boxes -> only they have a derivable gold page.
Questions with no boxes (empty annotations, unanswerable, or the test split) get
gold_pages = [] and are excluded from page-selection metrics.
"""
import os
import json


def _extract_gold_pages(apbb):
    """
    answers_page_bounding_boxes is a list (per answer) of lists (boxes) of dicts
    each with a 1-indexed 'page'. Return the sorted set of 0-indexed pages, or [].
    Robust to None / empty / missing.
    """
    pages = set()
    if not apbb:
        return []
    for per_answer in apbb:
        if not per_answer:
            continue
        # per_answer is usually a list of box dicts; tolerate a bare dict too.
        boxes = per_answer if isinstance(per_answer, list) else [per_answer]
        for box in boxes:
            if isinstance(box, dict) and "page" in box and box["page"] is not None:
                p = int(box["page"]) - 1        # 1-indexed -> 0-indexed
                if p >= 0:
                    pages.add(p)
    return sorted(pages)


class DUDEDataset:
    """
    Lightweight loader (not a torch Dataset subclass; the diagnostic reads embeddings
    directly). Each sample dict:
        question_id, doc_id, split, question, answers (list), answer_type,
        gold_pages (list[int], 0-indexed, possibly empty)
    """
    def __init__(self, gt_json, images_root, split=None):
        self.images_root = images_root
        self.split = split
        with open(gt_json, "r", encoding="utf-8") as f:
            obj = json.load(f)
        data = obj.get("data", obj) if isinstance(obj, dict) else obj

        self.samples = []
        for x in data:
            if split and x.get("data_split") != split:
                continue
            gold = _extract_gold_pages(x.get("answers_page_bounding_boxes"))
            self.samples.append({
                "question_id": x["questionId"],
                "doc_id": x["docId"],
                "split": x.get("data_split", split or ""),
                "question": x["question"],
                "answers": x.get("answers") or [""],
                "answer_type": x.get("answer_type", ""),
                "gold_pages": gold,
            })

    def image_path(self, doc_id, split, page):
        """Path to a single page image: images/<split>/<docId>_<page>.jpg (0-indexed)."""
        return os.path.join(self.images_root, split, f"{doc_id}_{page}.jpg")

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
