# DATA 266 Lab 2 — Team 42

Pair_Programming_Team_42: Nikhil Khaneja, Anushka Khadatkar

Three parts, one shared knowledge base of tools: multimodal RAG (Part 1), diffusion
fine-tuning (Part 2), and a multimodal multi-agent system (Part 3). See the lab
requirements PDF for full task details and grading rubric.

## Hosted APIs

None. All language/vision model inference uses local open-source models
(Qwen/Qwen2.5-VL-3B-Instruct for vision understanding and answer generation).

## Repo layout

```
data266-lab2-team42/
├── README.md               how to run each part (this file)
├── requirements.txt        pinned dependency versions
├── CONTRIBUTIONS.md        who did what, per part + meeting recordings log
├── AI_USAGE.md             which AI coding assistants were used, and for what
├── links.txt               GitHub tags/commits, Hugging Face links, Kaggle notebook link
├── .gitignore              excludes data/, weights, secrets
├── notebooks/
│   ├── p1_rag.ipynb
│   ├── p2_finetune.ipynb
│   └── p3_agents.ipynb
├── src/
│   ├── part1/               ingestion, retrieval, answering engine
│   ├── part2/                data prep, training, generation, metrics
│   └── part3/                 agents, orchestration, safety checks
├── eval/
│   ├── part1_golden_set.json       25 Q&A pairs (task 1.4)
│   ├── part2_prompts.json           20 fixed generation prompts (task 2.3)
│   ├── part3_scenarios.json         15 test scenarios + rubrics (task 3.3)
│   └── part3_redteam.json           10 red-team cases (task 3.3)
└── app/
    └── gradio_app.py        Part 3 demo app
```

- `notebooks/` — the three graded notebooks, imported from `src/`, not holding business
  logic inline. Submitted **executed, with all outputs visible** (Restart & Run All) —
  a notebook without outputs earns at most 50% of the marks it supports.
- `src/part1/`, `src/part2/`, `src/part3/` — shared code imported by the notebooks. Plain
  functions/classes, no lambdas, no type annotations — keep it simple and readable.
- `eval/` — golden sets, fixed prompts, and test scenarios. These are written and
  **committed before** the corresponding tuning/training/running starts, since the lab
  checks the git timestamp to confirm they weren't adjusted after seeing results.
- `app/` — Gradio demo app for Part 3, used live in the team demo.
- `requirements.txt` — pinned dependency versions (`package==x.y.z`), updated as
  dependencies are added.

## Notebook rules (every notebook)

1. First cell prints git commit, Python/library versions, GPU name, and the fixed seed
   (`SEED = 266`) — see the lab PDF for the exact snippet. Copy it verbatim into each
   notebook's first cell.
2. Fix seeds wherever randomness is involved.
3. Before committing a "final" version of a notebook: **Restart & Run All**, save, then
   commit — never commit a notebook with stale or missing outputs.

## What never goes in this repo

Per the lab rules: no model weights (`.pt`/`.pth`/`.ckpt`/`.safetensors`), no datasets,
no secrets/API keys. Model weights and generated images go to **Hugging Face**;
`.gitignore` already excludes these by pattern — don't force-add them.

## Git workflow (both partners)

- Work directly on `main`; always `git pull` before you start and before you push.
- Commit in small, logical steps — one task/change per commit, not end-of-day dumps.
- Prefix commit messages with the part, e.g. `[Part1] add hybrid retrieval`,
  `[Part2] LoRA run 1`, `[Docs] update README`.
- No AI-assistant attribution in commit messages or PRs.
- Tag each part when it's feature-complete and ready for grading:
  `git tag p1-v1.0 && git push origin p1-v1.0` (same for `p2-v1.0`, `p3-v1.0`). Tags
  created after the Nov 17 deadline are ignored by graders, so tag early and re-tag
  (new tag, don't move it) if you keep improving.
- Add the instructor/ISA GitHub handles (posted on Canvas) as repo collaborators
  (read access) once you have them.

## Setup

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`requirements.txt` pins a CUDA 13.0 build of torch/torchvision for local dev on an
RTX 5090 (Blackwell). On Kaggle/Colab, the preinstalled torch already matches their GPU
(T4/P100) — skip those two lines or let Kaggle's environment take precedence.

## Running each part

**Part 1** needs the real lab files from Canvas (Files → Labs → Lab 2 →
DATA266_Lab2_Part1) or the Kaggle competition's Data tab: `Document.pdf`,
`Document2_Supplement_A_Outturns_2011-2013.pdf`,
`Document3_Briefing_B_Labour_Prices_Policy.pdf`, `kb_manifest.json`, `Questions.csv`,
`sample_submission.csv`, `validate_submission.py`. Place them under `data/raw/part1/`,
then run `notebooks/p1_rag.ipynb` top to bottom.

**Part 2** only needs public data (`lambdalabs/naruto-blip-captions` + Stable Diffusion
1.5) and runs end to end via `notebooks/p2_finetune.ipynb`, or the individual scripts in
`src/part2/` (`data_prep.py`, `train_runs.py`, `generate.py`, `evaluate.py`,
`human_eval.py`, `push_to_hf.py`). Pushing to Hugging Face needs `huggingface-cli login`
first.

**Part 3** needs sample input images under `data/raw/part3/` matching the file names in
`eval/part3_scenarios.json` / `eval/part3_redteam.json`, then run
`notebooks/p3_agents.ipynb` or `python app/gradio_app.py` for the interactive demo.

## Weights and datasets

Model weights and generated images are hosted on Hugging Face, not committed to this
repo. Links are in `links.txt`.
