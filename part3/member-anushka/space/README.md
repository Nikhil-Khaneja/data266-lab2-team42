---
title: DATA266 Part 3 Multi-Agent Image System
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
short_description: Five agents that understand, restyle, edit and check images
---

# Multimodal multi-agent image system (DATA 266 Lab 2, Part 3)

Five agents handle each request. The **Planner** checks input safety and routes the request, **Vision** captions and
answers questions about the image, **Prompt Engineering** writes the generation prompt, **Image Generation** produces
the image, and **Critique** scores it and decides to accept it or revise it, up to 2 times. Each agent's output is shown in its own tab.

What you can ask:
- **Describe or ask about an image:** "Describe this image.", "How many cats are in this picture?"
- **Change the style:** "Turn this photo into a watercolor painting."
- **Edit or enhance:** "Turn the cats into tigers.", "Enhance this photo: brighten it and make it sharper."
- **Naruto style:** "Draw these two cats in Naruto anime style." This uses our Part 2 LoRA,
  [`Anushka1202/data266-lab2-part2-lora-rank16`](https://huggingface.co/Anushka1202/data266-lab2-part2-lora-rank16).
  It also works without an image: "Generate a Naruto-style ninja with a red scarf standing on a cliff at sunset."

Requests that should be refused or fail safely:
- unsafe requests
- deceptive edits of real people
- images containing instructions
- empty or broken files

**This Space runs on a free CPU**, so it uses lighter settings:
- `HuggingFaceTB/SmolVLM-500M-Instruct` instead of `Qwen/Qwen2.5-VL-3B-Instruct`
- 384 px images and 12 diffusion steps

A generation request takes one to several minutes, and the first request waits for the models to load. The reported
results were produced on a Colab T4 GPU with the full settings.

Models used:
- `stable-diffusion-v1-5/stable-diffusion-v1-5` with its safety checker
- `timbrooks/instruct-pix2pix`
- `openai/clip-vit-base-patch32`
- our Part 2 Naruto LoRA

For non-commercial research and education use. Naruto is a copyrighted franchise.
