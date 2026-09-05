"""Inference wrapper around the published weights.

Design choices, all deliberate and all different from the original ``app.py``:

- the tokenizer's own ``<|finetune_right_pad_id|>`` is used for batching, so no new pad
  token is added and the embedding matrix is never resized at inference time;
- decoding is greedy with ``max_new_tokens=8`` (the answer is five tokens long);
- the answer is parsed with an anchored regex; a non-match is reported as
  ``unparsable`` and is never silently mapped to 0;
- an optional teacher-forced score compares the logits of the ``0`` and ``1`` tokens
  after the prefix ``Logistics_Delay:`` so that a probability exists for diagnostics.
"""

from __future__ import annotations

import platform
import time
from collections.abc import Sequence

from .prompting import ANSWER_PREFIX, build_messages, parse_label

PAD_TOKEN = "<|finetune_right_pad_id|>"
EXPECTED_VOCAB = 128256


class Scorer:
    def __init__(self, model_id: str, dtype: str = "bf16", device: str | None = None, batch_size: int = 16):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_id = model_id
        self.batch_size = batch_size
        self.dtype_name = dtype
        t0 = time.time()
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if len(self.tok) != EXPECTED_VOCAB:
            raise RuntimeError(f"tokenizer has {len(self.tok)} tokens, expected {EXPECTED_VOCAB}")
        pad_id = self.tok.convert_tokens_to_ids(PAD_TOKEN)
        if not isinstance(pad_id, int) or pad_id < 0 or pad_id == self.tok.unk_token_id:
            raise RuntimeError(f"{PAD_TOKEN} missing from tokenizer")
        self.tok.pad_token = PAD_TOKEN
        self.tok.padding_side = "left"
        torch_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype).to(self.device).eval()
        self.model.config.use_cache = True
        # Decoding defaults are set in memory; the published generation_config is documented separately.
        gc = self.model.generation_config
        gc.do_sample = False
        gc.temperature = None
        gc.top_p = None
        gc.pad_token_id = pad_id
        self.pad_id = pad_id
        self.load_seconds = round(time.time() - t0, 1)

        ids0 = self.tok.encode(f"{ANSWER_PREFIX} 0", add_special_tokens=False)
        ids1 = self.tok.encode(f"{ANSWER_PREFIX} 1", add_special_tokens=False)
        self.score_available = len(ids0) == len(ids1) and ids0[:-1] == ids1[:-1]
        self.answer_prefix_ids = ids0[:-1]
        self.label_ids = {0: ids0[-1], 1: ids1[-1]}

    # ------------------------------------------------------------------ info
    def info(self) -> dict:
        import torch
        import transformers

        return {
            "model_id": self.model_id,
            "params": int(sum(p.numel() for p in self.model.parameters())),
            "dtype": self.dtype_name,
            "device": self.device,
            "cuda_device_name": torch.cuda.get_device_name(0) if self.device.startswith("cuda") else None,
            "tokenizer_len": len(self.tok),
            "pad_token": PAD_TOKEN,
            "pad_token_id": self.pad_id,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "python_version": platform.python_version(),
            "load_seconds": self.load_seconds,
            "score_available": self.score_available,
            "label_token_ids": self.label_ids,
        }

    # -------------------------------------------------------------- encoding
    def _prompts(self, user_texts: Sequence[str]) -> list[str]:
        return [
            self.tok.apply_chat_template(build_messages(u), add_generation_prompt=True, tokenize=False)
            for u in user_texts
        ]

    def _encode(self, prompts: Sequence[str]):
        # The chat template already emits <|begin_of_text|>, so no extra special tokens here.
        return self.tok(list(prompts), return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)

    # ------------------------------------------------------------ generation
    def generate(
        self,
        user_texts: Sequence[str],
        *,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        seed: int | None = None,
        max_new_tokens: int = 8,
    ) -> list[str]:
        import torch

        if seed is not None:
            torch.manual_seed(seed)
            if self.device.startswith("cuda"):
                torch.cuda.manual_seed_all(seed)
        outputs: list[str] = []
        prompts = self._prompts(user_texts)
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                enc = self._encode(prompts[start : start + self.batch_size])
                kwargs = {"max_new_tokens": max_new_tokens, "pad_token_id": self.pad_id, "do_sample": do_sample}
                if do_sample:
                    kwargs.update(temperature=temperature, top_p=top_p)
                out = self.model.generate(**enc, **kwargs)
                new = out[:, enc["input_ids"].shape[1] :]
                outputs.extend(t.strip() for t in self.tok.batch_decode(new, skip_special_tokens=True))
        return outputs

    def predict(self, user_texts: Sequence[str], **kwargs) -> tuple[list[str], list[int | None]]:
        raw = self.generate(user_texts, **kwargs)
        return raw, [parse_label(t) for t in raw]

    # --------------------------------------------------------- label scoring
    def label_scores(self, user_texts: Sequence[str]) -> list[dict]:
        """Teacher-forced logit margin ``logit(1) - logit(0)`` after ``Logistics_Delay:``.

        Diagnostic only: the model was trained to emit a hard label, so these scores
        are expected to be saturated. Returns an empty list if the prefix tokenisation
        is not shared between the two answers.
        """
        import torch

        if not self.score_available:
            return []
        prompts = self._prompts(user_texts)
        results: list[dict] = []
        prefix = torch.tensor(self.answer_prefix_ids, dtype=torch.long)
        with torch.no_grad():
            for start in range(0, len(prompts), self.batch_size):
                enc = self._encode(prompts[start : start + self.batch_size])
                bsz = enc["input_ids"].shape[0]
                ids = torch.cat([enc["input_ids"], prefix.unsqueeze(0).repeat(bsz, 1).to(self.device)], dim=1)
                mask = torch.cat(
                    [
                        enc["attention_mask"],
                        torch.ones(bsz, len(prefix), dtype=enc["attention_mask"].dtype, device=self.device),
                    ],
                    dim=1,
                )
                logits = self.model(input_ids=ids, attention_mask=mask).logits[:, -1, :].float()
                l0 = logits[:, self.label_ids[0]]
                l1 = logits[:, self.label_ids[1]]
                p1 = torch.softmax(torch.stack([l0, l1], dim=1), dim=1)[:, 1]
                top = logits.argmax(dim=1)
                for i in range(bsz):
                    results.append(
                        {
                            "margin": round(float(l1[i] - l0[i]), 4),
                            "p1": round(float(p1[i]), 6),
                            "top_token_is_label": bool(int(top[i]) in self.label_ids.values()),
                        }
                    )
        return results
