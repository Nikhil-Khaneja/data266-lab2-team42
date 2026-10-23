# DATA 266 Lab 2 — Team 42

Pair_Programming_Team_42: Nikhil Khaneja, Anushka Khadatkar

Three parts, one shared knowledge base of tools: multimodal RAG (Part 1), diffusion
fine-tuning (Part 2), and a multimodal multi-agent system (Part 3). See the lab
requirements PDF for full task details and grading rubric.

## Hosted APIs

None. All language/vision model inference uses local open-source models
(Qwen/Qwen2.5-VL-3B-Instruct for vision understanding and answer generation).

## Repo layout

- `notebooks/` — the three graded notebooks (`p1_rag.ipynb`, `p2_finetune.ipynb`,
  `p3_agents.ipynb`), submitted executed with outputs.
- `src/part1/`, `src/part2/`, `src/part3/` — shared code imported by the notebooks.
- `eval/` — golden sets, fixed prompts, and test scenarios, committed before tuning/running.
- `app/` — Gradio demo app for Part 3.
- `requirements.txt` — pinned dependency versions.

## Setup

```
pip install -r requirements.txt
```

## Running each part

Instructions added as each part is implemented.

## Weights and datasets

Model weights and generated images are hosted on Hugging Face, not committed to this
repo. Links are in `links.txt`.
