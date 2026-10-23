import torch
from diffusers import StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
EDIT_MODEL = "timbrooks/instruct-pix2pix"
LORA_REPO = None  # set once the Part 2 LoRA is pushed to Hugging Face, e.g. "team42/sd15-naruto-lora-run1"

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


def load_domain_lora(pipe):
    global _lora_loaded
    if LORA_REPO and not _lora_loaded:
        pipe.load_lora_weights(LORA_REPO)
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
            model_name = LORA_REPO or BASE_MODEL
            mode = "domain_style_generation"
        else:
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
