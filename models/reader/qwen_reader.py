"""
Qwen2.5-VL reader (retrieve-then-read Stage B, stronger alternative to PaliGemma).

Why: PaliGemma-3B can't read dense financial pages at 448px and can't ingest a
table as text (empty outputs). Qwen2.5-VL handles high/dynamic resolution + long
text prompts + numerical reasoning, so it can actually USE structured table text.

Same interface as PaliGemmaReader: generate(images, questions) -> list[str].
Loaded 4-bit (fits 24GB for the 7B model). Frozen (zero-shot) by default.
"""
import torch
import torch.nn as nn

SYSTEM = "You are a document understanding assistant. Answer concisely with only the final answer(s), no explanation."
TEXT_SYSTEM = ("You are a table and financial QA assistant. Use the given text and table to answer. "
               "Respond with ONLY the final answer (a number or a short phrase), no explanation.")


class QwenVLReader(nn.Module):
    def __init__(self, config):
        super().__init__()
        rc = config.get("reader", {}) or {}
        self.model_name = rc.get("model_name", "Qwen/Qwen2.5-VL-7B-Instruct")
        self.max_new_tokens = int(rc.get("max_new_tokens", 48))
        self.load_4bit = rc.get("load_4bit", True)
        self.torch_dtype = torch.bfloat16
        cuda_ok = torch.cuda.is_available()

        from transformers import AutoProcessor
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        # left padding so batched generation trims cleanly
        if hasattr(self.processor, "tokenizer"):
            self.processor.tokenizer.padding_side = "left"

        model_kwargs = dict(torch_dtype=self.torch_dtype, low_cpu_mem_usage=True)
        if self.load_4bit and cuda_ok:
            from transformers import BitsAndBytesConfig
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = {"": 0}
        else:
            model_kwargs["device_map"] = "cuda" if cuda_ok else None

        # transformers class name varies by version; try the specific ones, then Auto.
        ModelCls = None
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        except Exception:
            try:
                from transformers import Qwen2VLForConditionalGeneration as ModelCls
            except Exception:
                from transformers import AutoModelForImageTextToText as ModelCls
        self.model = ModelCls.from_pretrained(self.model_name, **model_kwargs)
        self.model.eval()

    @property
    def device(self):
        return self.model.device

    def _build_inputs(self, images, questions):
        texts = []
        for q in questions:
            messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]},
            ]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, images=list(images),
                                return_tensors="pt", padding=True)
        return inputs.to(self.device)

    @torch.no_grad()
    def generate(self, images, questions):
        """Greedy-decode answer strings for a batch (image + text). Returns list[str]."""
        inputs = self._build_inputs(images, questions)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        trimmed = gen[:, in_len:]
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [t.strip() for t in texts]

    @torch.no_grad()
    def generate_text(self, prompts, system=TEXT_SYSTEM):
        """Text-only generation (no image), for structured-table QA like MultiHiertt."""
        texts = []
        for q in prompts:
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": [{"type": "text", "text": q}]},
            ]
            texts.append(self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True))
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        gen = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        in_len = inputs["input_ids"].shape[1]
        trimmed = gen[:, in_len:]
        texts = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return [t.strip() for t in texts]
