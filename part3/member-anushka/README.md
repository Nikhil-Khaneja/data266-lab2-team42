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
   *My Drive / Sem3 / DATA266 / Lab2 / Part3*, so that `Part3/p3/`, `Part3/eval/` and `Part3/notebooks/` exist.
2. Open `Part3/notebooks/p3_agents.ipynb` in Colab and choose *Runtime → Change runtime type → T4 GPU*.
3. *Runtime → Run all* (~45-60 min). The first cell mounts Drive (approve the prompt), switches to
   the Part3 folder, installs `requirements-colab.txt`, and downloads/generates the test images into
   `data/images/` (not in git, per the repo policy). Results go to `Part3/outputs/` and human-eval
   templates to `Part3/human_eval/`. The last cell starts the Gradio app with a public link.
4. *File → Download .ipynb*, then commit to this folder: the executed notebook, `outputs/results.json`,
   `outputs/results_table.md`, `outputs/traces/` and the filled `human_eval/*.csv`.
   Copy `outputs/part3_eval_anushka.json` to the shared `part3/eval/` folder, next to the other member's final results.

Before uploading to Drive, run `python tools/stamp_commit.py` in the repo clone and upload the `COMMIT` file too. The Drive copy has no `.git`, so the first cell prints the commit from that file.
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

## Results (Colab Tesla T4, code at commit `40c0a5c`)

These are copied from the notebook outputs in `notebooks/p3_agents.ipynb` §9 (automatic measurements) and
`notebooks/p3_human_eval.ipynb` §5 (human measurements, `outputs/results_table_final.md`). They are the
pre-registered results: the scenarios and red-team cases were committed before this run, and nothing was tuned on them afterwards.

| system | task success, automatic rubric (/15) | human pass, agreed (/15) | critic agreement | red-team attack success (/10) | median s / request | median s / generation request |
|---|---|---|---|---|---|---|
| baseline (no critique loop) | 12/15 | 12/15 | n/a | 1/10 | 4.9 | 5.5 |
| final (critique loop, max 2 revisions) | **14/15** | 12/15 | 0.71 (10/14) | 1/10 | 6.4 | 8.8 |

**Human evaluation.** Both members judged all 30 scenario outputs (15 scenarios × 2 conditions) and all 14 critic
decisions independently. They used a review page that hid the automatic scores and the critic's decisions, and the raw
answers were committed before any discussion.
- **Agreement between members:** 25/30 scenario judgements (83.3%) and 12/14 critic labels (85.7%).
- **How disagreements went:** every scenario disagreement was a case where Anushka judged pass and Nikhil judged fail.
  We discussed each one and agreed on 4 fails (S05 baseline, S07 baseline, S11 both conditions) and 1 pass (S06 baseline).
  Both critic-label disagreements (S05 attempt 1, S06 attempt 0) were resolved as accept.
- **Human vs automatic:** the two agree on 24/30 judgements.
  - **Critique loop under human judgement:** humans gave the loop no advantage (12/15 in both conditions), while the
    automatic rubric gave it two extra passes. Humans failed S07 and S11 in both conditions, where the automatic rubric
    passed them, and passed S06 and S10 in the baseline, where it failed them.
  - **Critic agreement:** the critic matched our agreed label on 10/14 attempts. It accepted two images we would have
    sent back (S07, S11), and asked for revisions on two we would have accepted (S05 attempt 1, S06 attempt 0).

**What worked**
- The critique loop fixed 2 scenarios the baseline failed. In **S06** (Van Gogh room), the first image was almost
  unchanged (target CLIP 0.214). The critic asked for higher strength and guidance, and the revision became a clear Van Gogh-style
  painting (0.354). In **S10** (cats in Naruto style), the baseline drew ninja people (0.242). After 2 revisions the
  output showed anime cats (0.295).
- The critique loop costs +1.5 s per request at the median (+3.3 s on generation requests).
- Caption/question, edit, enhancement, text-to-image with the Part 2 LoRA, ambiguous and broken-input scenarios all passed in both conditions.
- Guardrails refused 9/10 red-team cases before any generation: injected text in 2 of 3 images, both NSFW requests,
  the obfuscated `n.u.d.e` request, both real-person edits, and both broken files (clear messages, no crash).

**What did not work:** see [failure_analysis.md](failure_analysis.md). In short:
- S05 failed because the caption's "cartoon-style" fought the requested realistic style.
- R01's injected text leaked through the caption instead of OCR, so it was not refused.
- In S07, InstructPix2Pix replaced most of the scene.
- The VLM judge said NO in 19 of 20 critiques.

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
