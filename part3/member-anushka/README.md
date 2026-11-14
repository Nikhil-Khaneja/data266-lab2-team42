# DATA 266 Lab 2 - Part 3: Multimodal Multi-Agent System

Five agents understand an input image, turn a user request into a generation prompt, generate a new
image (our Part 2 Naruto LoRA for domain-style requests), check it, and revise it if needed.

```
user request + image
   │
   ▼
Planner ──(unsafe / broken / ambiguous / unsupported)──► refuse or ask, stop
   │  classify: caption_question | style_change | edit | enhance | domain_generation
   │  input safety: NSFW terms, deceptive real-person edits, OCR scan for instructions in the image
   ▼
Vision Understanding  (caption, objects, visible text, ≥2 VQA answers) ──(caption/question)──► answer, stop
   ▼
Prompt Engineering  (template prompt from request + caption; image text never copied in)
   ▼
Image Generation   style/enhance: SD1.5 img2img · edit: InstructPix2Pix · domain: SD1.5 + Part 2 LoRA (img2img or txt2img)
   ▼
Critique  CLIP(image, prompt) + VLM yes/no judgement + CLIP preservation vs input + SD safety checker
   ├─ accept ──► final image
   └─ revise (concrete parameter/prompt change) ──► Prompt Engineering   (max 2 revisions)
```

All agents share one typed pydantic `AgentState` (`p3/state.py`); the orchestrator (`p3/graph.py`) is plain
Python and writes a JSON trace per request (`outputs/traces/<request_id>.json`: agent, input, output, seconds).

## Models (all open weights from the Hugging Face Hub; no hosted APIs)

| role | model |
|---|---|
| VLM (vision, OCR, critic judgement) | `Qwen/Qwen2.5-VL-3B-Instruct` (GPU). CPU smoke tests only: `HuggingFaceTB/SmolVLM-500M-Instruct` |
| style change / enhancement / txt2img | `stable-diffusion-v1-5/stable-diffusion-v1-5` (fp16) |
| domain-style (Part 2) | `Anushka1202/data266-lab2-part2-lora-rank16` (Naruto LoRA, rank 16, final Part 2 run) |
| instruction edits | `timbrooks/instruct-pix2pix` (UNet; shares SD 1.5 VAE / text encoder / safety checker) |
| CLIP scores | `openai/clip-vit-base-patch32` |
| output safety | Stable Diffusion safety checker (`CompVis/stable-diffusion-safety-checker`, bundled with SD 1.5) |

## How to run

**Colab (free T4, the graded setting)**
1. Copy the contents of this folder (`part3/member-anushka/`) to Google Drive at
   *My Drive / Sem3 / DATA266 / lab2 / Part3*, so that `Part3/p3/`, `Part3/eval/` and `Part3/notebooks/` exist.
2. Open `Part3/notebooks/p3_agents.ipynb` in Colab and choose *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all* (~45-60 min). The first cell mounts Drive (approve the prompt), switches to
   the Part3 folder, installs `requirements-colab.txt`, and downloads/generates the test images into
   `data/images/` (not in git, per the repo policy). Results go to `Part3/outputs/` and human-eval
   templates to `Part3/human_eval/`. The last cell starts the Gradio app with a public link.
4. *File → Download .ipynb*, then commit to this folder: the executed notebook, `outputs/results.json`,
   `outputs/results_table.md`, `outputs/traces/` and the filled `human_eval/*.csv`.
   Copy `outputs/part3_eval_anushka.json` to the shared `part3/eval/` folder, next to the other member's final results.

The first cell prints an empty commit hash on Colab because the Drive copy has no `.git`; the run belongs to the commit tagged `p3-v1.0`.
Without Drive, `python tools/make_zip.py` makes `part3_colab.zip`. Upload it to `/content` and the notebook unzips it.

**Local**
```bash
pip install -r requirements.txt
python tools/make_assets.py          # test images (already in data/images)
python -m p3.app                     # Gradio demo
jupyter nbconvert --to notebook --execute notebooks/p3_agents.ipynb --inplace
```
Without a GPU the code automatically uses the small CPU settings (384 px, 12 steps, SmolVLM). This is only
for smoke tests. Settings can be overridden with environment variables, e.g. `P3_SD_STEPS=20`.

## Files

| path | content |
|---|---|
| `p3/agents.py` | the five agents |
| `p3/graph.py` | orchestrator, revise loop (max 2), failure handling, JSON traces |
| `p3/safety.py` | guardrails (NSFW, real-person deception, image-text injection) |
| `p3/models.py` | VLM, CLIP, diffusion pipelines (lazy loaded) |
| `p3/evaluate.py` | rubric checks, red-team runner, results table, human-eval agreement |
| `p3/app.py` | Gradio demo |
| `eval/scenarios.json` | 15 test scenarios with pass/fail rubrics (written before running) |
| `eval/redteam.json` | 10 red-team cases |
| `human_eval/*.csv` | human pass/fail judgements and critic-agreement labels (filled in by both members) |
| `notebooks/p3_agents.ipynb` | executed evaluation notebook |

## Evaluation (C2)

| measurement | definition | where |
|---|---|---|
| Primary | task success: scenarios passing their rubric (/15), with vs without critique | notebook §4-5, §9 |
| Secondary | critic agreement with our accept/revise labels; red-team attack success (/10) | §7-9 |
| Human | both members judge each scenario pass/fail independently; agreement + resolution | §8 |
| Efficiency | median seconds per request | §9 |

Human-evaluation numbers come only from `human_eval/*.csv`, which the team fills in by hand. Re-run §8-9 after filling them.

## System card (summary)

- **Can do:** caption an image, list its objects, read visible text, answer questions; restyle a photo (img2img),
  edit it from an instruction (InstructPix2Pix), enhance it (low-strength img2img), and render it or a text prompt
  in Naruto anime style with our Part 2 LoRA. It checks its own output and revises it up to two times.
- **Known limits:** the Planner is rule-based (English keywords), so unusual phrasings can be misrouted or
  flagged as ambiguous. OCR misses very small text, so small injected instructions can go undetected. CLIP and
  the 3B VLM judge are imperfect critics. InstructPix2Pix can change more of the image than asked. The Naruto
  LoRA pulls every image towards anime style. Generation is 512 px only.
- **Safety measures:** input checks for NSFW/gore terms (with leetspeak and spacing normalization), deceptive edits
  of real people (named public figures, face swaps, fabricated arrests/news), and instructions written inside
  images (OCR scan; image text is never copied into prompts). A negative prompt always excludes NSFW content.
  The SD safety checker runs on every output; flagged images are withheld and regenerated, or refused.
  Broken, empty, tiny or blank images get a clear error message.
