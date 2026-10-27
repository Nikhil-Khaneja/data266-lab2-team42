import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
EDIT_MODEL = "timbrooks/instruct-pix2pix"
# Local LoRA weights from Part 2 (run1_lora_r4), used until the model is pushed to
# Hugging Face; LORA_REPO overrides this once a HF repo id is set (e.g. after
# huggingface-cli login + push_to_hf.py).
LORA_LOCAL_PATH = "models/part2/run1_lora_r4"
LORA_REPO = None

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
    return LORA_REPO or LORA_LOCAL_PATH


def load_domain_lora(pipe):
    global _lora_loaded
    if not _lora_loaded:
        pipe.load_lora_weights(domain_lora_source())
        _lora_loaded = True


def generate(state, seed=0, steps=30, strength=0.6):
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
    state.generation_mode = mode
    state.generation_model = model_name
    state.generation_seed = seed
    state.generation_steps = steps
    return state
