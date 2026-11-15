"""Lazy-loaded models shared by the agents.

* ``VLM``       - vision-language model (caption, objects, OCR, VQA, judgement)
* ``ClipScorer`` - CLIP ViT-B/32 for image-text and image-image similarity
* ``Diffusion`` - SD 1.5 img2img / txt2img (+ Part 2 Naruto LoRA) and InstructPix2Pix.
                  Pix2Pix re-uses SD 1.5's VAE, text encoder and safety checker
                  (InstructPix2Pix only fine-tuned the UNet), which saves ~1 GB of VRAM.
"""
from __future__ import annotations

import re
import threading
import time
from typing import Optional

import torch
from PIL import Image

from .config import Config


def fit_resolution(img: Image.Image, side: int) -> Image.Image:
    """Resize so the longer side is ``side`` and both sides are multiples of 8."""
    w, h = img.size
    s = side / max(w, h)
    nw, nh = max(64, int(round(w * s / 8)) * 8), max(64, int(round(h * s / 8)) * 8)
    return img.resize((nw, nh), Image.LANCZOS)


class VLM:
    def __init__(self, cfg: Config):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg
        self.is_qwen = "qwen" in cfg.vlm_id.lower()
        kw = {}
        if self.is_qwen:
            kw = dict(min_pixels=64 * 28 * 28, max_pixels=cfg.vlm_max_side * cfg.vlm_max_side)
        elif "smolvlm" in cfg.vlm_id.lower():
            kw = dict(do_image_splitting=False)
        self.processor = AutoProcessor.from_pretrained(cfg.vlm_id, **kw)
        self.model = AutoModelForImageTextToText.from_pretrained(
            cfg.vlm_id, dtype=cfg.torch_dtype).to(cfg.device).eval()
        self.dtype_note = str(cfg.torch_dtype)
        if cfg.device == "cuda" and cfg.torch_dtype == torch.float16 and not self._probe():
            # fp16 overflow guard: some VLMs emit garbage in fp16 on pre-Ampere GPUs -> reload in bf16
            del self.model
            torch.cuda.empty_cache()
            self.model = AutoModelForImageTextToText.from_pretrained(
                cfg.vlm_id, dtype=torch.bfloat16).to(cfg.device).eval()
            self.dtype_note = "torch.bfloat16 (fp16 probe failed)"

    def _probe(self) -> bool:
        img = Image.new("RGB", (224, 224), (220, 20, 20))
        try:
            ans = self.ask(img, "What color is this image? Answer with one word.", 8)
        except Exception:
            return False
        return "red" in ans.lower()

    @torch.inference_mode()
    def ask(self, image: Optional[Image.Image], question: str, max_new_tokens: Optional[int] = None) -> str:
        content = []
        if image is not None:
            img = image.copy()
            img.thumbnail((self.cfg.vlm_max_side, self.cfg.vlm_max_side))
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": question})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.cfg.vlm_max_new_tokens,
                                  do_sample=False)
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        text = re.sub(r"^\s*(assistant|Assistant)\s*:\s*", "", text)
        return text.strip()


class ClipScorer:
    def __init__(self, cfg: Config):
        from transformers import CLIPModel, CLIPProcessor

        self.device = cfg.device
        self.model = CLIPModel.from_pretrained(cfg.clip_id).to(cfg.device).eval()
        self.proc = CLIPProcessor.from_pretrained(cfg.clip_id)

    @torch.inference_mode()
    def _img(self, img: Image.Image) -> torch.Tensor:
        x = self.proc(images=img.convert("RGB"), return_tensors="pt").to(self.device)
        f = self.model.get_image_features(**x)
        f = getattr(f, "pooler_output", f)
        return torch.nn.functional.normalize(f, dim=-1)

    @torch.inference_mode()
    def _txt(self, text: str) -> torch.Tensor:
        x = self.proc(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77).to(self.device)
        f = self.model.get_text_features(**x)
        f = getattr(f, "pooler_output", f)
        return torch.nn.functional.normalize(f, dim=-1)

    def text_image(self, img: Image.Image, text: str) -> float:
        return float((self._img(img) @ self._txt(text).T).item())

    def image_image(self, a: Image.Image, b: Image.Image) -> float:
        return float((self._img(a) @ self._img(b).T).item())


class Diffusion:
    def __init__(self, cfg: Config):
        from diffusers import (EulerAncestralDiscreteScheduler, StableDiffusionImg2ImgPipeline,
                               StableDiffusionInstructPix2PixPipeline, StableDiffusionPipeline,
                               UNet2DConditionModel)

        self.cfg = cfg
        dt = cfg.torch_dtype
        self.img2img = StableDiffusionImg2ImgPipeline.from_pretrained(cfg.sd_id, torch_dtype=dt, variant="fp16")
        self.lora_loaded = False
        self.lora_error: Optional[str] = None
        try:
            self.img2img.load_lora_weights(cfg.lora_id, weight_name=cfg.lora_weight_name, adapter_name="naruto")
            self.img2img.disable_lora()
            self.lora_loaded = True
        except Exception as e:  # keep the system usable; domain requests will report it
            self.lora_error = f"{type(e).__name__}: {e}"
        self.txt2img = StableDiffusionPipeline(**self.img2img.components)
        try:
            p2p_unet = UNet2DConditionModel.from_pretrained(cfg.pix2pix_id, subfolder="unet", torch_dtype=dt, variant="fp16")
        except Exception:
            p2p_unet = UNet2DConditionModel.from_pretrained(cfg.pix2pix_id, subfolder="unet", torch_dtype=dt)
        c = self.img2img.components
        self.pix2pix = StableDiffusionInstructPix2PixPipeline(
            vae=c["vae"], text_encoder=c["text_encoder"], tokenizer=c["tokenizer"], unet=p2p_unet,
            scheduler=EulerAncestralDiscreteScheduler.from_pretrained(cfg.pix2pix_id, subfolder="scheduler"),
            safety_checker=c["safety_checker"], feature_extractor=c["feature_extractor"])
        for p in (self.img2img, self.txt2img, self.pix2pix):
            p.to(cfg.device)
            p.set_progress_bar_config(disable=True)

    def _gen(self, seed: int) -> torch.Generator:
        return torch.Generator(device="cpu").manual_seed(seed)

    def run(self, mode: str, prompt: str, negative_prompt: str, seed: int, image: Optional[Image.Image],
            strength: Optional[float], guidance_scale: float, image_guidance_scale: Optional[float]):
        """Returns (image, nsfw_flag, model_name, steps, resolution)."""
        cfg = self.cfg
        t0 = time.time()
        if mode in ("style", "enhance", "domain_img2img"):
            use_lora = mode == "domain_img2img"
            self._lora(use_lora)
            init = fit_resolution(image, cfg.resolution)
            out = self.img2img(prompt=prompt, negative_prompt=negative_prompt, image=init, strength=strength,
                               guidance_scale=guidance_scale, num_inference_steps=cfg.sd_steps,
                               generator=self._gen(seed))
            model, steps, res = cfg.sd_id + (" + " + cfg.lora_id if use_lora else ""), cfg.sd_steps, init.size
        elif mode == "domain_txt2img":
            self._lora(True)
            out = self.txt2img(prompt=prompt, negative_prompt=negative_prompt, height=cfg.resolution,
                               width=cfg.resolution, guidance_scale=guidance_scale,
                               num_inference_steps=cfg.sd_steps, generator=self._gen(seed))
            model, steps, res = cfg.sd_id + " + " + cfg.lora_id, cfg.sd_steps, (cfg.resolution, cfg.resolution)
        elif mode == "edit":
            init = fit_resolution(image, cfg.resolution)
            out = self.pix2pix(prompt=prompt, negative_prompt=negative_prompt, image=init,
                               guidance_scale=guidance_scale, image_guidance_scale=image_guidance_scale,
                               num_inference_steps=cfg.pix2pix_steps, generator=self._gen(seed))
            model, steps, res = cfg.pix2pix_id, cfg.pix2pix_steps, init.size
        else:
            raise ValueError(f"unknown generation mode {mode}")
        flags = getattr(out, "nsfw_content_detected", None) or [False]
        return out.images[0], bool(flags[0]), model, steps, list(res), time.time() - t0

    def _lora(self, on: bool) -> None:
        if not self.lora_loaded:
            if on:
                raise RuntimeError(f"Part 2 LoRA {self.cfg.lora_id} could not be loaded: {self.lora_error}")
            return
        if on:
            self.img2img.enable_lora()
            self.img2img.set_adapters(["naruto"], adapter_weights=[self.cfg.lora_scale])
        else:
            self.img2img.disable_lora()


class ModelHub:
    """Loads each model the first time an agent needs it."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._vlm = self._clip = self._diff = None
        # one lock per model: a background warm-up and a request must never load the same model twice
        self._locks = {"vlm": threading.Lock(), "clip": threading.Lock(), "diff": threading.Lock()}

    def _get(self, attr: str, key: str, factory):
        if getattr(self, attr) is None:
            with self._locks[key]:
                if getattr(self, attr) is None:
                    setattr(self, attr, factory(self.cfg))
        return getattr(self, attr)

    @property
    def vlm(self) -> VLM:
        return self._get("_vlm", "vlm", VLM)

    @property
    def clip(self) -> ClipScorer:
        return self._get("_clip", "clip", ClipScorer)

    @property
    def diffusion(self) -> Diffusion:
        return self._get("_diff", "diff", Diffusion)
