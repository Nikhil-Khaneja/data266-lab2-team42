"""Qwen2.5-VL-3B-Instruct wrapper with two interchangeable backends.

* "transformers" - Hugging Face Qwen/Qwen2.5-VL-3B-Instruct, fp16, on one CUDA GPU
                   (Kaggle T4/P100, Colab T4). This is the graded configuration.
* "mlx"          - mlx-community/Qwen2.5-VL-3B-Instruct-4bit on Apple Silicon, for
                   local development only (same model weights, 4-bit quantized).

Decoding is greedy (no sampling) so answers are reproducible.
"""
import os
import time

HF_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
MLX_MODEL = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"


def pick_backend():
    forced = os.environ.get("P1_LLM_BACKEND")
    if forced:
        return forced
    try:
        import torch
        if torch.cuda.is_available():
            return "transformers"
    except ImportError:
        pass
    try:
        import mlx.core  # noqa: F401
        return "mlx"
    except ImportError:
        return "transformers"


class VLM:
    def __init__(self, backend=None, max_pixels=1280 * 28 * 28):
        self.backend = backend or pick_backend()
        self.max_pixels = max_pixels
        self.calls, self.seconds = 0, 0.0
        if self.backend == "mlx":
            from mlx_vlm import load
            self.model, self.processor = load(MLX_MODEL)
            self.model_name = MLX_MODEL
        else:
            import torch
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
            self.torch = torch
            dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported(including_emulation=False) else torch.float16
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                HF_MODEL, torch_dtype=dtype, device_map="cuda" if torch.cuda.is_available() else "cpu",
                attn_implementation="sdpa")
            self.model.eval()
            self.processor = AutoProcessor.from_pretrained(HF_MODEL, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
            self.model_name = f"{HF_MODEL} ({str(dtype).replace('torch.', '')})"

    def chat(self, system, user, images=None, max_tokens=400):
        """images: list of file paths (PNG). Returns the generated text."""
        images = [p for p in (images or []) if p and os.path.exists(p)]
        t0 = time.time()
        if self.backend == "mlx":
            try:
                out = self._chat_mlx(system, user, images, max_tokens)
            except RuntimeError:                       # transient Metal timeout on small Macs: retry once
                import mlx.core as mx
                mx.clear_cache()
                time.sleep(5)
                out = self._chat_mlx(system, user, images, max_tokens)
        else:
            out = self._chat_hf(system, user, images, max_tokens)
        self.calls += 1
        self.seconds += time.time() - t0
        return out.strip()

    def _chat_mlx(self, system, user, images, max_tokens):
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = apply_chat_template(self.processor, self.model.config, msgs, num_images=len(images))
        # collect token ids and decode once: mlx-vlm's incremental detokenizer doubles some word endings
        toks = []
        for r in stream_generate(self.model, self.processor, prompt, image=images or None,
                                 max_tokens=max_tokens, temperature=0.0):
            if getattr(r, "is_draft", False):
                continue
            if r.token is not None:
                toks.append(int(r.token))
        tok = self.processor.tokenizer if hasattr(self.processor, "tokenizer") else self.processor
        return tok.decode(toks, skip_special_tokens=True)

    def _chat_hf(self, system, user, images, max_tokens):
        from PIL import Image
        torch = self.torch
        content = [{"type": "image", "image": p} for p in images] + [{"type": "text", "text": user}]
        msgs = [{"role": "system", "content": [{"type": "text", "text": system}]},
                {"role": "user", "content": content}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        pil = [Image.open(p).convert("RGB") for p in images]
        inputs = self.processor(text=[text], images=pil or None, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False,
                                      temperature=None, top_p=None, top_k=None)
        gen = out[:, inputs["input_ids"].shape[1]:]
        return self.processor.batch_decode(gen, skip_special_tokens=True)[0]
