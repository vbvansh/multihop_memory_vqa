"""
TAT-DQA loader (financial document VQA with tables).

Structure of tatdqa_dataset_*.json: a list of documents, each:
    {"doc": {"uid", "page", "source"}, "questions": [ {question, answer(list),
             answer_type, scale, ...}, ... ]}
Files on disk are named by doc.uid:  <uid>.pdf  and  <uid>_<page>.png
(in the docs dir). One doc = one page.

Each flattened sample:
    question_id, doc_uid, page, question, answers(list), answer_type, scale,
    pdf_path, image_path
"""
import os
import json
import glob


class TATDQADataset:
    def __init__(self, json_path, docs_dir):
        self.docs_dir = docs_dir
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.samples = []
        for doc in data:
            d = doc.get("doc", {})
            uid = d.get("uid")
            page = int(d.get("page", 1))
            if not uid:
                continue
            pdf_path, img_path = self._find_files(uid, page)
            for q in doc.get("questions", []):
                self.samples.append({
                    "question_id": q.get("uid"),
                    "doc_uid": uid,
                    "page": page,
                    "question": q.get("question", ""),
                    "answers": q.get("answer") or [],
                    "answer_type": q.get("answer_type", ""),
                    "scale": q.get("scale", "") or "",
                    "pdf_path": pdf_path,
                    "image_path": img_path,
                })

    def _find_files(self, uid, page):
        """Locate the doc's PDF and page PNG, tolerant of naming."""
        pdf = os.path.join(self.docs_dir, uid + ".pdf")
        if not os.path.exists(pdf):
            c = glob.glob(os.path.join(self.docs_dir, uid + "*.pdf"))
            pdf = c[0] if c else pdf
        img = os.path.join(self.docs_dir, f"{uid}_{page}.png")
        if not os.path.exists(img):
            c = sorted(glob.glob(os.path.join(self.docs_dir, uid + "*.png")))
            img = c[0] if c else img
        return pdf, img

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
