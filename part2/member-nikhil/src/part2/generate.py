import json
import os

import torch
from diffusers import StableDiffusionPipeline

BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SEEDS = [0, 1]
STEPS = 30
GUIDANCE_SCALE = 7.5

MODELS = {
    "base": None,
    "run1_lora_r4": "models/part2/run1_lora_r4",
    "run2_lora_r16": "models/part2/run2_lora_r16",
}


def load_pipeline(lora_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(BASE_MODEL, torch_dtype=dtype).to(device)
    if lora_path:
        pipe.load_lora_weights(lora_path)
    return pipe


def generate_for_model(model_name, lora_path, prompts, output_dir):
    """Stable Diffusion's safety checker is applied to every output. The filtered image (black
    when flagged) is what is published, shown and rated, in output_dir. The unfiltered image is
    kept only on this machine, in <root>/raw/<model>/, so that a false positive does not turn
    into a black image inside KID, CLIP and LPIPS."""
    pipe = load_pipeline(lora_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checker, pipe.safety_checker = pipe.safety_checker, None
    raw_dir = os.path.join(os.path.dirname(output_dir), "raw", model_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    records = []
    for item in prompts:
        prompt = item["prompt"]
        for seed in SEEDS:
            generator = torch.Generator(device=device).manual_seed(seed)
            raw = pipe(prompt, num_inference_steps=STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator,
                       output_type="np").images
            pipe.safety_checker = checker
            filtered, flags = pipe.run_safety_checker(raw.copy(), device, pipe.unet.dtype)
            pipe.safety_checker = None
            nsfw = bool(flags and flags[0])
            image = pipe.image_processor.numpy_to_pil(filtered)[0]
            safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt)[:50]
            file_name = "{}_seed{}.png".format(safe_prompt, seed)
            image.save(os.path.join(output_dir, file_name))
            pipe.image_processor.numpy_to_pil(raw)[0].save(os.path.join(raw_dir, file_name))
            records.append({"model": model_name, "prompt": prompt, "category": item["category"],
                             "seed": seed, "file_name": file_name, "steps": STEPS,
                             "guidance_scale": GUIDANCE_SCALE, "nsfw": nsfw})
    del pipe
    torch.cuda.empty_cache()
    return records


def generate_all(prompts_path, output_root):
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)

    all_records = []
    for model_name, lora_path in MODELS.items():
        output_dir = os.path.join(output_root, model_name)
        all_records.extend(generate_for_model(model_name, lora_path, prompts, output_dir))

    with open(os.path.join(output_root, "generation_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2)
    return all_records


if __name__ == "__main__":
    generate_all("../eval/part2_prompts.json", "data/processed/part2/generated")
