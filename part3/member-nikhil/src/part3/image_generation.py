import json
import os

import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
EDIT_MODEL = "timbrooks/instruct-pix2pix"
# The run Part 2 selected as final (recorded in its memorization report), loaded from Hugging Face.
# P3_LORA overrides it, e.g. with a local folder.
PART2_REPOS = {"run1_lora_r4": "nikhilkhaneja/sd15-naruto-lora-run1", "run2_lora_r16": "nikhilkhaneja/sd15-naruto-lora-run2"}
PART2_FINAL_RECORD = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "part2", "member-nikhil", "eval", "part2_memorization.json")


def part2_final_repo():
    try:
        with open(PART2_FINAL_RECORD, encoding="utf-8") as f:
            return PART2_REPOS[json.load(f)["model"]]
    except (OSError, KeyError, ValueError):
        return PART2_REPOS["run2_lora_r16"]


LORA_REPO = os.environ.get("P3_LORA") or part2_final_repo()

_img2img_pipe = None
_edit_pipe = None
_lora_loaded = False


def get_img2img_pipe():
    global _img2img_pipe
    if _img2img_pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    return _img2img_pipe


def get_edit_pipe():
    global _edit_pipe
    if _edit_pipe is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        _edit_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(EDIT_MODEL, torch_dtype=dtype).to(device)
    return _edit_pipe


def domain_lora_source():
    return LORA_REPO


def load_domain_lora(pipe):
    global _lora_loaded
    if not _lora_loaded:
        pipe.load_lora_weights(domain_lora_source())
        _lora_loaded = True


def generate(state, seed=None, steps=30, strength=0.6):
    # Each revision uses the next seed (0, 1, 2): a revised prompt that happens to match the
    # previous one still gives the critic a new image, never the identical image again.
    seed = state.revision_count if seed is None else seed
    generator = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(seed)

    if state.route == "edit":
        pipe = get_edit_pipe()
        result = pipe(state.generation_prompt, image=state.input_image, num_inference_steps=steps,
                      generator=generator)
        model_name = EDIT_MODEL
        mode = "edit"
    else:
        pipe = get_img2img_pipe()
        if state.route == "domain_style_generation":
            load_domain_lora(pipe)
            pipe.enable_lora()
            model_name = domain_lora_source()
            mode = "domain_style_generation"
        else:
            if _lora_loaded:
                pipe.disable_lora()
            model_name = BASE_MODEL
            mode = "style_change"
        result = pipe(prompt=state.generation_prompt, image=state.input_image, strength=strength,
                      num_inference_steps=steps, generator=generator)

    state.generated_image = result.images[0]
    # Both pipelines keep Stable Diffusion's safety checker; a hit is passed to the Critique agent.
    flags = getattr(result, "nsfw_content_detected", None)
    state.output_flagged = bool(flags and flags[0])
    state.generation_mode = mode
    state.generation_model = model_name
    state.generation_seed = seed
    state.generation_steps = steps
    return state
