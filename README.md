# DATA 266 Lab 2 — Team 42

Pair_Programming_Team_42: Nikhil Khaneja, Anushka Khadatkar

Three parts, one shared knowledge base of tools: multimodal RAG (Part 1), diffusion
fine-tuning (Part 2), and a multimodal multi-agent system (Part 3). See the lab
requirements PDF for full task details and grading rubric.

## Hosted APIs

None. Every model runs locally from open weights on the Hugging Face Hub:

| Part | Models |
|---|---|
| 1 | `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit, bitsandbytes) for figure descriptions and answers; `BAAI/bge-small-en-v1.5` embeddings; `BAAI/bge-reranker-v2-m3` reranker |
| 2 | `stable-diffusion-v1-5/stable-diffusion-v1-5` + our LoRA runs; `openai` CLIP ViT-B-32 (open_clip) and LPIPS (AlexNet) for evaluation |
| 3 (`member-nikhil`) | `Qwen/Qwen2.5-VL-3B-Instruct` (fp16); SD 1.5 img2img; `timbrooks/instruct-pix2pix`; our Part 2 LoRA `nikhilkhaneja/sd15-naruto-lora-run2`; CLIP ViT-B-32 |
| 3 (`member-anushka`) | See `part3/member-anushka/README.md` |

## Repo layout

One repo, three parts. Inside each part, one directory per team member, so both
partners' work lives side by side without overwriting each other:

```
data266-lab2-team42/
├── README.md               how to run each part (this file)
├── requirements.txt        pinned dependency versions (shared .venv for the whole repo)
├── CONTRIBUTIONS.md        who did what, per part + meeting recordings log
├── AI_USAGE.md             which AI coding assistants were used, and for what
├── links.txt               GitHub tags/commits, Hugging Face links, Kaggle notebook link
├── .gitignore              excludes data/, weights, secrets
├── part1/
│   ├── eval/
│   │   └── part1_golden_set.json   25 Q&A pairs (task 1.4) — shared, same for both members
│   ├── member-nikhil/
│   │   ├── src/
│   │   │   ├── common/          shared VLM chat helper
│   │   │   └── part1/           ingestion, retrieval, answering engine
│   │   └── notebooks/
│   │       └── p1_rag.ipynb
│   └── member-anushka/          Anushka's own Part 1 implementation
├── part2/
│   ├── eval/
│   │   └── part2_prompts.json      20 fixed generation prompts (task 2.3) — shared
│   ├── member-nikhil/
│   │   ├── src/part2/           data prep, training, generation, metrics
│   │   └── notebooks/
│   │       └── p2_finetune.ipynb
│   └── member-anushka/          Anushka's own Part 2 implementation
└── part3/
    ├── eval/
    │   ├── part3_scenarios.json     15 test scenarios + rubrics (task 3.3) — shared
    │   └── part3_redteam.json       10 red-team cases (task 3.3) — shared
    ├── member-nikhil/
    │   ├── src/
    │   │   ├── common/          shared VLM chat helper
    │   │   └── part3/           agents, orchestration, safety checks
    │   ├── eval/
    │   │   ├── part3_eval_report.json   this member's own results against the shared tests
    │   │   └── part3_traces/            per-request JSON traces from this member's eval runs
    │   ├── notebooks/
    │   │   └── p3_agents.ipynb
    │   └── app/
    │       └── gradio_app.py    Part 3 demo app
    └── member-anushka/          Anushka's own Part 3 implementation
```

- `part{N}/eval/` — the golden set, fixed prompts, test scenarios, and red-team cases:
  the fixed test definitions, identical for both members, written and **committed
  before** the corresponding tuning/training/running starts (the lab checks the git
  timestamp to confirm they weren't adjusted after seeing results). Only one copy per
  part — never duplicated into a member folder, so it can't drift between members.
- Each `member-<name>/` folder is otherwise self-contained: its own `src/`, `notebooks/`,
  and (locally, gitignored) `data/`/`models/`, plus its own `eval/` for that member's
  *results* (scores, reports, traces) from running their implementation against the
  shared tests above. Scripts are run with the member folder as the working directory —
  see "Running each part" below.
- `member-nikhil/` currently holds the one working implementation built so far for each
  part. `member-anushka/` is scaffolded with a README stub — add your own implementation
  there, following the same internal layout.
- `src/common/` (inside `member-nikhil/` for Part 1 and Part 3) is a small shared VLM
  chat helper, duplicated rather than centralized, so each member's folder stays
  self-contained and runnable on its own.
- `notebooks/` — the graded notebook for each part, imported from that member's `src/`,
  not holding business logic inline. Submitted **executed, with all outputs visible**
  (Restart & Run All) — a notebook without outputs earns at most 50% of the marks it
  supports.
- `app/` — Gradio demo app for Part 3, used live in the team demo.
- `requirements.txt` — pinned dependency versions (`package==x.y.z`), one shared `.venv`
  for the whole repo across both members and all three parts.

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
  `[Part2] LoRA run 1`, `[Docs] update README`. Only touch your own
  `member-<name>/` folder within a part — that's what keeps both partners' work
  side by side without merge conflicts.
- No AI-assistant attribution in commit messages or PRs.
- Tag each part when it's feature-complete and ready for grading:
  `git tag p1-v1.0 && git push origin p1-v1.0` (same for `p2-v1.0`, `p3-v1.0`). Tags
  created after the Nov 17 deadline are ignored by graders, so tag early and re-tag
  (new tag, don't move it) if you keep improving.
- Add the instructor/ISA GitHub handles (posted on Canvas) as repo collaborators
  (read access) once you have them.

## Setup

One shared virtual environment at the repo root, used by both members for every part:

```
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

`requirements.txt` pins the versions the notebooks were executed with, including the CUDA 12.8
build of torch/torchvision (RTX 4090). On Kaggle/Colab, keep the preinstalled torch: Part 1's
Kaggle notebook installs only `part1/member-nikhil/kaggle_requirements.txt`.

## Running each part

Each part's scripts expect to be run **with that member's folder as the working
directory** (e.g. `part3/member-nikhil/`), since `src/`, `eval/`, `data/`, and `models/`
all sit inside it. From the repo root:

```
cd part1\member-nikhil      # or part2\..., part3\...
..\..\.venv\Scripts\python.exe -m src.part1.score_golden_set   # example
```

**Part 1** needs the real lab files from Canvas (Files → Labs → Lab 2 →
DATA266_Lab2_Part1) or the Kaggle competition's Data tab: `Document.pdf`,
`Document2_Supplement_A_Outturns_2011-2013.pdf`,
`Document3_Briefing_B_Labour_Prices_Policy.pdf`, `kb_manifest.json`, `Questions.csv`,
`sample_submission.csv`, `validate_submission.py`. Place them under
`part1/member-nikhil/data/raw/part1/`, then run
`part1/member-nikhil/notebooks/p1_rag.ipynb` top to bottom. On Kaggle the same notebook runs
with the competition data and the code attached as a dataset; it reads `QUESTIONS_PATH`
(default `Questions.csv`) and writes `submission.csv` with no manual step.

**Part 2** only needs public data (`lambdalabs/naruto-blip-captions` + Stable Diffusion
1.5) and runs end to end via `part2/member-nikhil/notebooks/p2_finetune.ipynb`, or the
individual scripts in `part2/member-nikhil/src/part2/` (`data_prep.py`, `train_runs.py`,
`generate.py`, `evaluate.py`, `human_eval.py`, `push_to_hf.py`), run with
`part2/member-nikhil/` as the working directory (the notebook switches there itself). It
trains both runs, generates 120 images, evaluates them and builds the blinded rater page; the
last cell pushes both adapters and the images to Hugging Face when `hf auth login` has been run.

**Part 3** (`member-nikhil`): `part3/member-nikhil/notebooks/p3_agents.ipynb` makes its own test
images (`src/part3/make_test_images.py`), runs the 15 scenarios with and without the critique loop
and the 10 red-team cases, saves a JSON trace per request plus the input and output images, and
writes the human-evaluation sheets. Interactive demo, from `part3/member-nikhil/`:
`python app/gradio_app.py`. Anushka's implementation has its own instructions in
`part3/member-anushka/README.md`.

## Weights and datasets

Model weights and generated images are hosted on Hugging Face, not committed to this
repo. Links are in `links.txt`.
