import os

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# RAG_MODEL / RAG_QUANT select the model. Qwen2.5-VL-7B in 4-bit takes about 6GB, well inside
# the lab's one-16GB-GPU limit on a Kaggle T4.
MODEL_NAME = os.environ.get("RAG_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
QUANTIZATION = os.environ.get("RAG_QUANT", "4bit").strip().lower()
# Enough resolution for the small printed value labels on charts.
MAX_PIXELS = int(os.environ.get("RAG_MAX_PIXELS", str(2200 * 28 * 28)))

_model = None
_processor = None


def get_model_and_processor():
    global _model, _processor
    if _model is None:
        has_gpu = torch.cuda.is_available()
        # fp16 (not bf16): Kaggle/Colab T4 GPUs don't have native bf16 tensor cores
        kwargs = {"dtype": torch.float16 if has_gpu else torch.float32, "device_map": "cuda:0" if has_gpu else "cpu",
                  "attn_implementation": "sdpa"}
        if QUANTIZATION == "4bit" and has_gpu:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, **kwargs)
        _processor = AutoProcessor.from_pretrained(MODEL_NAME, max_pixels=MAX_PIXELS)
    return _model, _processor


def chat(system_prompt, user_prompt, images=None, max_new_tokens=512, prefill=""):
    """images: optional list of file paths or PIL images, shown to the model before the text.
    prefill: text the reply is forced to begin with (used to hold the model to a JSON shape)."""
    model, processor = get_model_and_processor()
    loaded = [Image.open(i).convert("RGB") if isinstance(i, str) else i for i in (images or [])]
    content = [{"type": "image", "image": image} for image in loaded]
    content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + prefill
    inputs = processor(text=[text], images=loaded or None, return_tensors="pt").to(model.device)
    if os.environ.get("RAG_DEBUG"):
        print("prompt_tokens", inputs["input_ids"].shape[1], flush=True)
    # Greedy decoding: the same question asked twice must get the same answer.
    with torch.inference_mode():
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    return prefill + processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
