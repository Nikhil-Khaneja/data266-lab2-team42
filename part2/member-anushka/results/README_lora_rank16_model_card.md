---
license: creativeml-openrail-m
base_model: stable-diffusion-v1-5/stable-diffusion-v1-5
datasets:
- lambdalabs/naruto-blip-captions
library_name: diffusers
pipeline_tag: text-to-image
tags:
- text-to-image
- stable-diffusion
- lora
- diffusers
---

# DATA266 Lab 2 Part 2: Naruto-style LoRA (rank 16) - final run

LoRA adapter for the UNet of `stable-diffusion-v1-5/stable-diffusion-v1-5` that generates images in the style of the Naruto anime.
Course project (SJSU DATA 266), non-commercial research use only.

## Base model
`stable-diffusion-v1-5/stable-diffusion-v1-5` at 512x512. Only the UNet attention projections are adapted; the text encoder and VAE are frozen.

## Data
`lambdalabs/naruto-blip-captions`: 1221 anime frames with BLIP captions. 0 corrupt and
0 exact-duplicate images were removed; split 90/10 with seed 266 into
1098 training and 123 held-out test images. A perceptual-hash audit found
1 test image(s) with a near-duplicate in the training set.

## Training settings
| setting | value |
|---|---|
| method | LoRA, rank 16 (UNet only) |
| trainable parameters | 3,188,736 |
| steps | 1500 |
| batch size x gradient accumulation | 1 x 4 |
| learning rate / schedule | 0.0001 constant, no warm-up |
| precision / optimizer | fp16 mixed precision, 8-bit AdamW, gradient checkpointing |
| augmentation | center crop, random horizontal flip |
| seed | 266 |
| hardware / time | 1x Tesla T4, 0.81 GPU-hours, peak 6.17 GiB |

## Results
20 fixed prompts (10 in-domain from held-out captions, 5 out-of-domain, 5 compositional), seeds 0 and 1,
40 steps of PNDMScheduler, guidance 7.5. KID: 20 in-domain generations vs 123 test images.

| model | KID in-domain (lower better) | CLIP score, 20 prompts | CLIP score, in-domain | LPIPS seed diversity | Human win rate vs base (excl. ties) | NSFW flagged / 40 | Training GPU-hours | Peak GPU memory (GiB) | Trainable parameters |
|---|---|---|---|---|---|---|---|---|---|
| base | 0.1420 ± 0.0260 | 0.3126 | 0.2953 | 0.6902 | reference | 0 | 0.0 | - | 0 |
| lora_rank4 | 0.0336 ± 0.0155 | 0.3067 | 0.2885 | 0.6924 | not rated | 2 | 0.812 | 6.15 | 797184 |
| lora_rank16 | 0.0351 ± 0.0162 | 0.3037 | 0.2865 | 0.6761 | 42.6% | 0 | 0.808 | 6.17 | 3188736 |

## Limitations
* Strong style prior: out-of-domain prompts (photographs, paintings) are pulled towards the anime style.
* Weak on composition: counting, colour binding and spatial relations across several characters.
* BLIP captions are short and generic; caption terms are skewed towards male characters.
* KID is computed from only 20 generated images and is noisy.

## Safety
* All generations pass the Stable Diffusion safety checker before release; flagged images were removed.
  Flagged on our 40 evaluation images: 0.
* Memorization check (final run): the 10 in-domain generations were matched to their nearest training images
  by CLIP similarity; 0 of 10 are closer than the 95th percentile of
  real unseen frames, and the smallest LPIPS distance to a training image is 0.553.

## License and intended use
The adapter is derived from Stable Diffusion v1.5 and is distributed under the CreativeML OpenRAIL-M licence,
including its use restrictions. The training images show copyrighted Naruto characters: use the adapter and
its outputs for non-commercial research and teaching only.

## Use
```python
import torch
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
pipe.load_lora_weights("Anushka1202/data266-lab2-part2-lora-rank16")
image = pipe("a man with pink hair and a hat", num_inference_steps=40, guidance_scale=7.5).images[0]
```
