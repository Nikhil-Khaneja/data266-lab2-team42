import json
import os

from huggingface_hub import HfApi, create_repo

HF_USERNAME = "nikhilkhaneja"
BASE_MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
DATASET = "lambdalabs/naruto-blip-captions"
REPOS = {"run1_lora_r4": "sd15-naruto-lora-run1", "run2_lora_r16": "sd15-naruto-lora-run2"}
IMAGE_DATASET_REPO = "sd15-naruto-lora-generated-images"


def model_card(run_report, metrics, data_report, memorization, final_model):
    """Model card built only from the numbers the notebook computed, never typed in."""
    by_model = {m["model"]: m for m in metrics}
    run, base = by_model[run_report["run_name"]], by_model["base"]
    is_final = run_report["run_name"] == final_model
    closest = [p["similarity"] for p in memorization["closest_pairs"]] if is_final else []
    lines = [
        "---", "license: creativeml-openrail-m", "base_model: " + BASE_MODEL,
        "datasets:", "- " + DATASET, "tags:", "- lora", "- stable-diffusion", "- text-to-image", "---", "",
        "# " + REPOS[run_report["run_name"]], "",
        "LoRA fine-tune of Stable Diffusion 1.5 on Naruto BLIP Captions, for DATA 266 Lab 2 (Team 42)."
        + (" **This is the run selected as final.**" if is_final else ""), "",
        "## Training", "",
        "| Setting | Value |", "|---|---|",
        "| Base model | `{}` |".format(BASE_MODEL),
        "| Data | `{}`: {} train / {} test images after removing {} corrupted and {} duplicate images; split 90/10 with seed 266 |".format(
            DATASET, data_report["train_size"], data_report["test_size"],
            data_report["corrupted_removed"], data_report["duplicates_removed"]),
        "| Resolution | 512 x 512 |",
        "| LoRA rank | {} |".format(run_report["rank"]),
        "| Trainable parameters | {:,} |".format(run_report["trainable_params"]),
        "| Steps | {} |".format(run_report["max_train_steps"]),
        "| Effective batch size | {} (batch 1, gradient accumulation 4) |".format(run_report["effective_batch_size"]),
        "| Learning rate | {} (constant) |".format(run_report["learning_rate"]),
        "| Precision | fp16 mixed precision, gradient checkpointing |",
        "| Seed | 266 |",
        "| Peak GPU memory | {} GB |".format(run_report["peak_gpu_memory_gb"]),
        "| Training time | {} GPU-hours on {} |".format(run_report["gpu_hours"], run_report["gpu"]), "",
        "## Results", "",
        "20 fixed prompts (10 in-domain, 5 out-of-domain, 5 compositional) x seeds 0 and 1; 30 steps, guidance 7.5.", "",
        "| Metric | Base SD 1.5 | This run |", "|---|---|---|",
        "| KID vs test images, in-domain prompts (lower is better) | {:.3f} | {:.3f} |".format(base["kid_mean"], run["kid_mean"]),
        "| CLIP score, prompt vs image | {:.3f} | {:.3f} |".format(base["clip_score_mean"], run["clip_score_mean"]),
        "| LPIPS between the two seeds (diversity) | {:.3f} | {:.3f} |".format(base["lpips_diversity_mean"], run["lpips_diversity_mean"]),
        "",
        "## Limitations", "",
        "Trained on about 1.1k images of one anime style. Out-of-domain prompts drift towards that style or are "
        "partly ignored, and BLIP captions are generic. KID on 20 images is noisy.", "",
        "## License and safety", "",
        "Naruto is a copyrighted character: the dataset and this adapter are for non-commercial research and "
        "education only. Images were generated with Stable Diffusion's built-in safety checker enabled.",
    ]
    if closest:
        lines.append("Memorization check: for 10 in-domain generations, the closest training image by CLIP "
                     "similarity scored {:.2f}-{:.2f} for the top 3 pairs.".format(min(closest), max(closest)))
    return "\n".join(lines) + "\n"


def push_lora(local_dir, card):
    run_name = os.path.basename(local_dir.rstrip("/\\"))
    repo_id = "{}/{}".format(HF_USERNAME, REPOS[run_name])
    create_repo(repo_id, exist_ok=True)
    with open(os.path.join(local_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)
    HfApi().upload_folder(folder_path=local_dir, repo_id=repo_id, repo_type="model",
                          allow_patterns=["pytorch_lora_weights.safetensors", "README.md", "run_report.json"],
                          delete_patterns=["checkpoint-*/**"])
    return "https://huggingface.co/" + repo_id


def push_image_dataset(generated_dir):
    repo_id = "{}/{}".format(HF_USERNAME, IMAGE_DATASET_REPO)
    create_repo(repo_id, exist_ok=True, repo_type="dataset")
    with open(os.path.join(generated_dir, "generation_metadata.json"), "r", encoding="utf-8") as f:
        metadata = json.load(f)
    readme = (
        "# {}\n\n"
        "{} images generated for DATA 266 Lab 2 Part 2: base Stable Diffusion 1.5 and two LoRA fine-tunes "
        "(rank 4, rank 16), 20 fixed prompts x seeds 0 and 1, 30 steps, guidance 7.5.\n\n"
        "`generation_metadata.json` lists each image's model, prompt, category, seed, steps, guidance and "
        "safety-checker flag; images are in one folder per model. Images flagged by Stable Diffusion's "
        "safety checker are published black.\n"
    ).format(IMAGE_DATASET_REPO, len(metadata))
    with open(os.path.join(generated_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    # raw/ holds the unfiltered images used only for the metrics; it is never published.
    HfApi().upload_folder(folder_path=generated_dir, repo_id=repo_id, repo_type="dataset",
                          ignore_patterns=["raw/**"], delete_patterns=["*.png", "**/*.png"])
    return "https://huggingface.co/datasets/" + repo_id
