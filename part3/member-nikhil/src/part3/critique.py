import json
import re

import open_clip
import torch

from src.common.vlm import chat
from src.part3.safety import check_text_safety

SYSTEM_PROMPT = """You are the Critique agent. Judge whether the generated image matches the
generation prompt and looks acceptable. Respond with ONLY a JSON object:
{"judgement": "<=40 words", "meets_prompt": true or false}
"""

_clip_model = None
_clip_preprocess = None
_clip_tokenizer = None


def get_clip():
    global _clip_model, _clip_preprocess, _clip_tokenizer
    if _clip_model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _clip_model, _, _clip_preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        _clip_model = _clip_model.to(device).eval()
        _clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")
    return _clip_model, _clip_preprocess, _clip_tokenizer


def clip_similarity(image, prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = get_clip()
    image_input = preprocess(image).unsqueeze(0).to(device)
    text_input = tokenizer([prompt]).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_input)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = (image_features @ text_features.T).item()
    return similarity


def vlm_judgement(image, prompt):
    user_prompt = "Generation prompt: {}".format(prompt)
    raw = chat(SYSTEM_PROMPT, user_prompt, image=image)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return raw, False
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return raw, False
    return data.get("judgement", raw), bool(data.get("meets_prompt", False))


def critique(state, clip_threshold=0.25, max_revisions=2):
    similarity = clip_similarity(state.generated_image, state.generation_prompt)
    judgement_text, meets_prompt = vlm_judgement(state.generated_image, state.generation_prompt)

    is_safe, reason = check_text_safety(judgement_text)
    if not is_safe:
        state.critique_decision = "revise"
        state.critique_feedback = "Output flagged unsafe ({}); regenerate a safe alternative.".format(reason)
        state.critique_score = similarity
        return state

    state.critique_score = similarity
    accept = meets_prompt and similarity >= clip_threshold

    if accept or state.revision_count >= max_revisions:
        state.critique_decision = "accept"
        state.critique_feedback = judgement_text
    else:
        state.critique_decision = "revise"
        state.critique_feedback = "CLIP similarity {:.2f} below target or prompt not met: {}".format(
            similarity, judgement_text
        )
        state.revision_count += 1

    return state
