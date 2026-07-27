"""
InfographicVQA loader (dense infographic images + QA + Amazon Textract OCR).

QA json: {"data": [ {questionId, question, answers, image_local_name,
                     ocr_output_file, answer_type?, ...} ]}
Images:  infographicsvqa_images/<image_local_name>
OCR:     infographicsvqa_ocr/<ocr_output_file>   (Textract: WORD/LINE blocks with
         normalized BoundingBox {Left, Top, Width, Height})

Also provides OCR helpers to locate the answer region (for the focus/crop test):
we match the gold answer text against OCR lines/words to get an (oracle) box.
"""
import os
import json


class InfographicVQADataset:
    def __init__(self, qas_json, images_dir, ocr_dir):
        self.images_dir = images_dir
        self.ocr_dir = ocr_dir
        obj = json.load(open(qas_json, "r", encoding="utf-8"))
        data = obj.get("data", obj) if isinstance(obj, dict) else obj
        self.samples = []
        for x in data:
            self.samples.append({
                "question_id": x.get("questionId"),
                "question": x.get("question", ""),
                "answers": x.get("answers") or [],
                "answer_type": x.get("answer_type"),
                "image_path": os.path.join(images_dir, x.get("image_local_name", "")),
                "ocr_path": os.path.join(ocr_dir, x.get("ocr_output_file", "")),
            })

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)


def load_ocr_boxes(ocr_path):
    """Robustly collect text blocks with normalized boxes from a Textract-style OCR json."""
    try:
        obj = json.load(open(ocr_path, "r", encoding="utf-8"))
    except Exception:
        return []
    out = []

    def walk(o):
        if isinstance(o, dict):
            bt = str(o.get("BlockType", "")).upper()
            txt = o.get("Text")
            geo = o.get("Geometry", {})
            bb = geo.get("BoundingBox") if isinstance(geo, dict) else None
            if txt and isinstance(bb, dict) and all(k in bb for k in ("Left", "Top", "Width", "Height")):
                out.append({
                    "text": str(txt), "type": bt or "WORD",
                    "left": float(bb["Left"]), "top": float(bb["Top"]),
                    "width": float(bb["Width"]), "height": float(bb["Height"]),
                })
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(obj)
    return out


def answer_box(boxes, answers):
    """Locate an (oracle) box for the answer: a LINE containing it, else the union of matching words."""
    ans = [a.lower().strip() for a in answers if a and str(a).strip()]
    if not ans or not boxes:
        return None
    lines = [b for b in boxes if b["type"] == "LINE"] or boxes
    for a in ans:
        for b in lines:
            if a in b["text"].lower():
                return {k: b[k] for k in ("left", "top", "width", "height")}
    toks = set(t for a in ans for t in a.split())
    matched = [b for b in boxes if b["type"] != "LINE" and b["text"].lower() in toks]
    if matched:
        l = min(b["left"] for b in matched)
        t = min(b["top"] for b in matched)
        r = max(b["left"] + b["width"] for b in matched)
        btm = max(b["top"] + b["height"] for b in matched)
        return {"left": l, "top": t, "width": r - l, "height": btm - t}
    return None
