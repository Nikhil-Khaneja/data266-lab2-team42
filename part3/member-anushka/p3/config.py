"""Run configuration for the Part 3 multi-agent system.

The same code runs on a 16 GB GPU (Colab/Kaggle T4, the graded setting) and on a
laptop CPU (smoke tests only). Device-dependent defaults are chosen here; every
value can be overridden with an environment variable of the same name in upper case
prefixed with ``P3_`` (e.g. ``P3_SD_STEPS=20``).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

import torch

SEED = 266
ROOT = Path(__file__).resolve().parent.parent


def _gpu() -> bool:
    return torch.cuda.is_available()


@dataclass
class Config:
    # ---- hardware -------------------------------------------------------
    device: str = field(default_factory=lambda: "cuda" if _gpu() else "cpu")
    dtype: str = field(default_factory=lambda: "float16" if _gpu() else "float32")

    # ---- models (all open-source, loaded from the Hugging Face Hub) -------
    # Vision-language model: Qwen2.5-VL-3B on GPU (same family as Part 1);
    # SmolVLM-500M is only a CPU fallback for local smoke tests.
    vlm_id: str = field(default_factory=lambda: "Qwen/Qwen2.5-VL-3B-Instruct" if _gpu()
                        else "HuggingFaceTB/SmolVLM-500M-Instruct")
    sd_id: str = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    pix2pix_id: str = "timbrooks/instruct-pix2pix"
    clip_id: str = "openai/clip-vit-base-patch32"
    # Part 2 final run (Naruto style, LoRA rank 16) - see README for the model card.
    lora_id: str = "Anushka1202/data266-lab2-part2-lora-rank16"
    lora_weight_name: str = "pytorch_lora_weights.safetensors"

    # ---- generation settings ---------------------------------------------
    resolution: int = field(default_factory=lambda: 512 if _gpu() else 384)
    sd_steps: int = field(default_factory=lambda: 30 if _gpu() else 12)
    pix2pix_steps: int = field(default_factory=lambda: 30 if _gpu() else 12)
    guidance_scale: float = 7.5
    image_guidance_scale: float = 1.5      # instruct-pix2pix
    style_strength: float = 0.60           # img2img strength for style change
    domain_strength: float = 0.70          # img2img strength for Naruto-style requests
    enhance_strength: float = 0.30         # img2img strength for enhancement
    lora_scale: float = 1.0
    seed: int = SEED

    # ---- vision ----------------------------------------------------------
    vlm_max_new_tokens: int = 96
    vlm_max_side: int = field(default_factory=lambda: 448 if _gpu() else 384)

    # ---- critique --------------------------------------------------------
    use_critique: bool = True
    max_revisions: int = 2
    clip_accept: float = 0.26      # CLIP(ViT-B/32) cosine image-vs-prompt needed to accept
    clip_strong: float = 0.31      # accept on CLIP alone above this, even if the VLM says no
    preserve_min: float = 0.55     # CLIP image-image similarity to input for style/edit/enhance

    # ---- input checks ----------------------------------------------------
    min_side: int = 64
    min_pixel_std: float = 4.0

    # ---- output ----------------------------------------------------------
    out_dir: str = field(default_factory=lambda: str(ROOT / "outputs"))

    def __post_init__(self) -> None:
        for f in fields(self):
            env = os.environ.get("P3_" + f.name.upper())
            if env is None:
                continue
            cur = getattr(self, f.name)
            if isinstance(cur, bool):
                val = env.lower() in {"1", "true", "yes"}
            elif isinstance(cur, int):
                val = int(env)
            elif isinstance(cur, float):
                val = float(env)
            else:
                val = env
            setattr(self, f.name, val)

    @property
    def torch_dtype(self) -> torch.dtype:
        return getattr(torch, self.dtype)

    def public(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}
