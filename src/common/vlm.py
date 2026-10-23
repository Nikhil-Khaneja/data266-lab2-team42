import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_NAME = "Qwen/Qwen2.5-VL-3B-Instruct"

_model = None
_processor = None


def get_model_and_processor():
    global _model, _processor
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # fp16 (not bf16): Kaggle/Colab T4 GPUs don't have native bf16 tensor cores
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_NAME, torch_dtype=dtype, device_map=device
        )
        _processor = AutoProcessor.from_pretrained(MODEL_NAME)
    return _model, _processor


def chat(system_prompt, user_prompt, image=None, max_new_tokens=512):
    model, processor = get_model_and_processor()
    content = []
    if image is not None:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": user_prompt})
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images = [image] if image is not None else None
    inputs = processor(text=[text], images=images, return_tensors="pt").to(model.device)
    output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = output_ids[:, inputs["input_ids"].shape[1]:]
    output_text = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return output_text.strip()
