import sys

from huggingface_hub import HfApi, create_repo

HF_USERNAME = "team42"  # replace with your actual Hugging Face username/org once known

MODEL_CARD_TEMPLATE = """---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
tags:
- lora
- stable-diffusion
- text-to-image
---

# {repo_name}

LoRA fine-tune of Stable Diffusion 1.5 on the Naruto BLIP Captions dataset
(lambdalabs/naruto-blip-captions), for DATA 266 Lab 2, Team 42.

- Base model: stable-diffusion-v1-5/stable-diffusion-v1-5
- Dataset: lambdalabs/naruto-blip-captions (non-commercial research use; Naruto is a
  copyrighted character)
- LoRA rank: {rank}
- Training steps: {steps}
- Learning rate: {lr}

## Limitations

Trained on a small (~1.2k image), narrow-domain dataset. Out-of-domain prompts may be
ignored or produce stylistically inconsistent results. For research/educational use only.
"""


def push_lora(local_dir, repo_name, rank, steps, lr):
    repo_id = "{}/{}".format(HF_USERNAME, repo_name)
    create_repo(repo_id, exist_ok=True)

    with open("{}/README.md".format(local_dir), "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.format(repo_name=repo_name, rank=rank, steps=steps, lr=lr))

    api = HfApi()
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    print("Pushed to https://huggingface.co/{}".format(repo_id))


if __name__ == "__main__":
    push_lora("models/part2/run1_lora_r4", "sd15-naruto-lora-run1", rank=4, steps=1200, lr="1e-4")
    push_lora("models/part2/run2_lora_r16", "sd15-naruto-lora-run2", rank=16, steps=1200, lr="1e-4")
