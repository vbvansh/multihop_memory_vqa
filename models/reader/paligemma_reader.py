import torch
import torch.nn as nn


class PaliGemmaReader(nn.Module):
    """
    Retrieve-then-read READER (Stage B).

    A pretrained PaliGemma-3B VLM, loaded 4-bit (QLoRA) and adapted with LoRA,
    that reads a SINGLE routed page image + the question and generates the
    answer text. It never sees more than one page, so batches of one 448px image
    per question are uniform and stackable (no variable-page-count problem).

    Config keys (under config["reader"]):
        model_name        : HF id (default google/paligemma-3b-mix-448, VQA-tuned)
        prompt_template   : str with {question}; PaliGemma likes a task prefix
        max_new_tokens    : generation cap
        use_lora/load_4bit: toggles for QLoRA
        lora_r/alpha/dropout/target_modules
    """

    def __init__(self, config):
        super().__init__()
        reader_cfg = config.get("reader", {}) or {}
        self.model_name = reader_cfg.get("model_name", "google/paligemma-3b-mix-448")
        self.prompt_template = reader_cfg.get("prompt_template", "answer en {question}")
        self.max_new_tokens = int(reader_cfg.get("max_new_tokens", 20))
        self.use_lora = reader_cfg.get("use_lora", True)
        self.load_4bit = reader_cfg.get("load_4bit", True)
        self.torch_dtype = torch.bfloat16

        device = config["model"].get("device", "cuda")
        cuda_ok = torch.cuda.is_available()

        from transformers import PaliGemmaForConditionalGeneration, PaliGemmaProcessor
        self.processor = PaliGemmaProcessor.from_pretrained(self.model_name)

        # "eager" avoids the SDPA strict attn_mask/query dtype check that crashes
        # PaliGemma under 4-bit QLoRA (BFloat16 mask vs float32 query). "sdpa" is faster
        # but only safe once transformers fixes that mask dtype; keep eager by default.
        attn_impl = reader_cfg.get("attn_implementation", "eager")
        model_kwargs = dict(
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            attn_implementation=attn_impl,
        )
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
            model_kwargs["device_map"] = device if cuda_ok else None

        self.model = PaliGemmaForConditionalGeneration.from_pretrained(self.model_name, **model_kwargs)

        if self.use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            if self.load_4bit and cuda_ok:
                # use_reentrant=False + input-grad hook so gradients actually flow
                # through gradient checkpointing (otherwise LoRA weights never update)
                self.model = prepare_model_for_kbit_training(
                    self.model,
                    use_gradient_checkpointing=True,
                    gradient_checkpointing_kwargs={"use_reentrant": False},
                )
            lora_cfg = LoraConfig(
                r=int(reader_cfg.get("lora_r", 16)),
                lora_alpha=int(reader_cfg.get("lora_alpha", 32)),
                lora_dropout=float(reader_cfg.get("lora_dropout", 0.05)),
                target_modules=reader_cfg.get(
                    "lora_target_modules",
                    ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                ),
                task_type="CAUSAL_LM",
                bias="none",
            )
            self.model = get_peft_model(self.model, lora_cfg)
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()
            self.model.print_trainable_parameters()

    @property
    def device(self):
        return self.model.device

    def _prompts(self, questions):
        return [self.prompt_template.format(question=q) for q in questions]

    def _to_device(self, inputs):
        out = {}
        for k, v in inputs.items():
            if k == "pixel_values":
                out[k] = v.to(self.device, self.torch_dtype)
            else:
                out[k] = v.to(self.device)
        return out

    def compute_loss(self, images, questions, answers):
        """
        Teacher-forced LM loss on the answer suffix.
        The processor's `suffix` arg appends the target and builds `labels`
        that mask the image + prompt tokens (only answer tokens contribute).

        Args:
            images: list[PIL.Image] (one routed page per sample)
            questions: list[str]
            answers: list[str] (primary answer per sample)
        Returns:
            scalar loss tensor
        """
        self.processor.tokenizer.padding_side = "right"
        inputs = self.processor(
            text=self._prompts(questions),
            images=images,
            suffix=answers,
            return_tensors="pt",
            padding="longest",
        )
        inputs = self._to_device(inputs)
        return self.model(**inputs).loss

    @torch.no_grad()
    def generate(self, images, questions):
        """Greedy-decode answer strings for a batch. Returns list[str]."""
        self.processor.tokenizer.padding_side = "left"
        inputs = self.processor(
            text=self._prompts(questions),
            images=images,
            return_tensors="pt",
            padding="longest",
        )
        inputs = self._to_device(inputs)
        input_len = inputs["input_ids"].shape[-1]
        generated = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
        )
        generated = generated[:, input_len:]
        texts = self.processor.batch_decode(generated, skip_special_tokens=True)
        return [t.strip() for t in texts]

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_lora(self, path):
        """Saves LoRA adapter weights (not the frozen 4-bit base)."""
        self.model.save_pretrained(path)

    def load_lora(self, path):
        """Loads LoRA adapter weights onto the already-loaded base model."""
        self.model.load_adapter(path, adapter_name="default")
