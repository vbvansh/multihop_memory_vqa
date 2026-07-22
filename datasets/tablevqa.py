"""
TableVQA-Bench loader (tables rendered as IMAGES + question + gold HTML/Markdown).

Read with pandas (NOT the HF `datasets` lib — this repo's local `datasets/` package
shadows it). Each subset is a parquet with columns:
    qa_id, image({'bytes','path'}), question, text_markdown_table, text_html_table, gt

Subsets: fintabnetqa (financial, HTML only), vwtq / vwtq_syn (WikiTableQuestions-style,
dense reasoning), vtabfact (fact verification, gt in {0,1}).
"""
import io
import os
import glob
import pandas as pd
from PIL import Image

ALL_SUBSETS = ["fintabnetqa", "vwtq", "vwtq_syn", "vtabfact"]


class TableVQADataset:
    def __init__(self, data_dir, subsets=None):
        subsets = subsets or ALL_SUBSETS
        self.samples = []
        for sub in subsets:
            files = glob.glob(os.path.join(data_dir, f"{sub}-*.parquet"))
            if not files:
                print(f"[TableVQA] no parquet found for subset '{sub}' in {data_dir}")
                continue
            df = pd.read_parquet(files[0])
            for _, r in df.iterrows():
                img = r["image"]
                img_bytes = img["bytes"] if isinstance(img, dict) else img
                self.samples.append({
                    "qa_id": str(r["qa_id"]),
                    "subset": sub,
                    "question": str(r["question"]),
                    "gt": str(r["gt"]),
                    "html": str(r.get("text_html_table") or ""),
                    "markdown": str(r.get("text_markdown_table") or ""),
                    "image_bytes": img_bytes,
                })

    def image(self, sample):
        return Image.open(io.BytesIO(sample["image_bytes"])).convert("RGB")

    def __len__(self):
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)
