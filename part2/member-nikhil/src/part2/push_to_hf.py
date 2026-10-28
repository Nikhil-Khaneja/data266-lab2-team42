import json

from huggingface_hub import HfApi, create_repo

HF_USERNAME = "nikhilkhaneja"

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
(lambdalabs/naruto-blip-captions), for DATA 266 Lab 2, Team 42.{final_note}

- Base model: stable-diffusion-v1-5/stable-diffusion-v1-5
- Dataset: lambdalabs/naruto-blip-captions (non-commercial research use; Naruto is a
  copyrighted character)
- LoRA rank: {rank}
- Training steps: {steps}
- Learning rate: {lr}
{results_section}
## Limitations

Trained on a small (~1.2k image), narrow-domain dataset. Out-of-domain prompts may be
ignored or produce stylistically inconsistent results. For research/educational use only.

## Safety

No NSFW filter is applied by this adapter itself; outputs should be generated through
a pipeline with Stable Diffusion's built-in safety checker enabled for any
externally-facing use. A memorization check (closest training image by CLIP
similarity, on 10 in-domain generations) found similarities of 0.90-0.92 for the
closest pairs -- notably similar but short of near-exact duplication.
"""

RESULTS_TEMPLATE = """
## Evaluation results (selected as final run)

| Metric | Base SD1.5 | This run |
|---|---|---|
| KID (lower is better) | 0.119 | 0.024 |
| CLIP score | 0.312 | 0.304 |
| LPIPS seed diversity | 0.687 | 0.660 |

Selected over the other LoRA run (rank 4) for its much lower KID (0.024 vs 0.050)
against the real test set, at essentially the same CLIP score.
"""


def push_lora(local_dir, repo_name, rank, steps, lr, final=False):
    repo_id = "{}/{}".format(HF_USERNAME, repo_name)
    create_repo(repo_id, exist_ok=True)

    final_note = " **This is the run selected as final.**" if final else ""
    results_section = RESULTS_TEMPLATE if final else "\n"
    with open("{}/README.md".format(local_dir), "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.format(
            repo_name=repo_name, rank=rank, steps=steps, lr=lr,
            final_note=final_note, results_section=results_section,
        ))

    api = HfApi()
    api.upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model")
    print("Pushed to https://huggingface.co/{}".format(repo_id))


def push_image_dataset(generated_dir, repo_name):
    repo_id = "{}/{}".format(HF_USERNAME, repo_name)
    create_repo(repo_id, exist_ok=True, repo_type="dataset")

    with open("{}/generation_metadata.json".format(generated_dir), "r", encoding="utf-8") as f:
        metadata = json.load(f)

    readme = (
        "# {}\n\n"
        "{} images generated for DATA 266 Lab 2 Part 2: base Stable Diffusion 1.5 and two "
        "LoRA fine-tunes (rank 4, rank 16), 20 fixed prompts x 2 seeds each.\n\n"
        "Each image's prompt, seed, model and category are in generation_metadata.json, "
        "and in the same-named subfolder per model.\n"
    ).format(repo_name, len(metadata))
    with open("{}/README.md".format(generated_dir), "w", encoding="utf-8") as f:
        f.write(readme)

    api = HfApi()
    api.upload_folder(folder_path=generated_dir, repo_id=repo_id, repo_type="dataset")
    print("Pushed to https://huggingface.co/datasets/{}".format(repo_id))


if __name__ == "__main__":
    push_lora("models/part2/run1_lora_r4", "sd15-naruto-lora-run1", rank=4, steps=1200, lr="1e-4")
    push_lora("models/part2/run2_lora_r16", "sd15-naruto-lora-run2", rank=16, steps=1200, lr="1e-4", final=True)
    push_image_dataset("data/processed/part2/generated", "sd15-naruto-lora-generated-images")
