# Grading Guide: DATA 266 Lab 2, Team 42

Pair_Programming_Team_42: Nikhil Khaneja, Anushka Khadatkar.
Repository: https://github.com/Nikhil-Khaneja/data266-lab2-team42

This page maps each rubric item in the lab requirements to the file that answers it. Every number
below is copied from a saved notebook output or a result file that notebook wrote. The file named
on each row is the source.

Both members built a complete pipeline for every part, in `part{N}/member-anushka/` and
`part{N}/member-nikhil/`. The **graded (final) system** for each part is:

| Part | Final system | Tag |
|---|---|---|
| 1 | `part1/member-anushka/` (Kaggle public score 0.930) | `p1-v1.0` <!-- TODO: commit hash --> |
| 2 | `part2/member-anushka/` (LoRA rank 16) | `p2-v1.0` <!-- TODO: commit hash --> |
| 3 | `part3/member-anushka/` <!-- TODO: confirm final Part 3 implementation --> | `p3-v1.0` <!-- TODO: commit hash --> |

---

## Common components

| Item | Where |
|---|---|
| C1.2 Pinned requirements, how to run | `requirements.txt`; `README.md` → "Running each part"; per-part READMEs in each `member-anushka/` |
| C1.3 Executed notebooks | `part1/member-anushka/notebooks/p1_rag.ipynb`, `p1_human_eval.ipynb`; `part2/member-anushka/notebooks/p2_finetune.ipynb`, `p2_evaluate.ipynb`; `part3/member-anushka/notebooks/p3_agents.ipynb`, `p3_human_eval.ipynb` |
| C1.4 Cell 1 run info (commit, versions, GPU, seed 266) | First code cell of each notebook above |
| C1.5 Tags | `p1-v1.0`, `p2-v1.0`, `p3-v1.0`; commits listed in `links.txt` |
| C1.6 Weights and images on Hugging Face | `links.txt` (no weights or datasets in Git; see `.gitignore`) |
| C2.3 Test sets committed before tuning | See "Test-set timestamps" below |
| C3.4 Contributions, AI usage | `CONTRIBUTIONS.md`, `AI_USAGE.md` (plus `part1/member-anushka/AI_USAGE.md`, `part2/member-anushka/AI_USAGE.md`) |
| Hosted APIs | None. All models are open weights run locally (`README.md` → "Hosted APIs") |

### Test-set timestamps (C2.3)

| Test set | File | First commit |
|---|---|---|
| Part 1 golden set (final system) | `part1/member-anushka/eval/p1_golden_set.csv` | 2026-11-15 (written 2026-11-11, after tuning on the *practice* questions began; disclosed in `CONTRIBUTIONS.md`) |
| Part 1 golden set (Nikhil) | `part1/eval/part1_golden_set.json` | 2026-10-27 (4 items replaced 2026-11-15 for overlapping practice questions) |
| Part 2 fixed 20 prompts (final) | `part2/member-anushka/eval/p2_fixed_20_prompts.json` | 2026-11-14, frozen in the training notebook before training |
| Part 2 fixed 20 prompts (Nikhil) | `part2/eval/part2_prompts.json` | 2026-10-27 |
| Part 3 scenarios + red team (Anushka) | `part3/member-anushka/eval/scenarios.json`, `redteam.json` | 2026-11-13 |
| Part 3 scenarios + red team (Nikhil) | `part3/eval/part3_scenarios.json`, `part3_redteam.json` | 2026-10-27 |

---

## Part 1: Multimodal RAG (25 + 10)

Code: `part1/member-anushka/p1_rag/`. Notebook: `part1/member-anushka/notebooks/p1_rag.ipynb`.
Write-ups: `part1/member-anushka/README.md`, `failure_analysis.md`.

| Task | Requirement | Evidence |
|---|---|---|
| 1.1.1 | Text chunks with doc ID and page | `p1_rag/ingest.py`; notebook ingestion cell (DOC1 186 / DOC2 10 / DOC3 9 text units) |
| 1.1.2 | Every numbered figure/table cropped with label + caption; "(corrected)" kept separate | `p1_rag/ingest.py`; notebook ingestion cell (27 figures, 9 tables) |
| 1.1.3 | Figures searchable | VLM descriptions, `p1_rag/figures.py` (Qwen2.5-VL-3B) |
| 1.1.4 | Vector DB with doc_id, page, figure_id, trust_level from manifest | ChromaDB, `p1_rag/index.py` |
| 1.2.1 | Hybrid dense + BM25 (+ reranker) | `p1_rag/index.py`: bge-small + BM25, RRF fusion, `bge-reranker-v2-m3` |
| 1.2.2 | Exact named-figure retrieval | Exact-label routing in `p1_rag/index.py`; notebook retrieval demo cell |
| 1.2.3 | JSON answers following the answering rules, with citations | `p1_rag/facts.py` (precedence: revision > outturn > forecast; conflict → higher trust), `p1_rag/answer.py` |
| 1.3 | Per-conversation memory; user values never in the vector DB | `p1_rag/answer.py` (`RAGSystem.new_conversation()`) |
| 1.4.1 | 25-item golden set covering all six categories | `part1/member-anushka/eval/p1_golden_set.csv`; per-category scores in the notebook |
| 1.4.2 | Four measurements, baseline vs final | Table below; `part1/eval/final_results/`; human evaluation in `part1/member-anushka/notebooks/p1_human_eval.ipynb` (ratings: `part1/member-anushka/eval/p1_human_ratings.csv`, raw exports in `eval/human_ratings_raw/`) |
| 1.5 | Kaggle notebook reads `QUESTIONS_PATH`, writes `submission.csv` | Notebook question-run cell; `part1/member-anushka/results/kaggle_final/submission.csv`; Kaggle link in `links.txt` |
| Bonus | Reranker improvement on the golden set | `part1/eval/final_results/reranker_ablation.csv` |

### Results (golden set, 25 questions; source: `part1/eval/final_results/README.md`)

| Measurement | Baseline (dense only) | Hybrid, no reranker | Final (hybrid + reranker) |
|---|---|---|---|
| Primary: answer correctness | 11/25 | 25/25 | 25/25 |
| Secondary: Recall@5 of the evidence used for the answer | 0.833 | 0.958 | 0.958 |
| Secondary: citation accuracy | 0.760 | 0.887 | 0.887 |
| Human: groundedness, mean 1–5, and agreement within 1 point (both members, 25 answers) | not rated | not rated | 5.0; 1.00 within 1 point (1.00 exact) |
| Efficiency: median s/question (T4) | 2.59 | 3.53 | 4.79 |

**Two Recall@5 numbers.** The table above measures the evidence each system used to write its
answer. The bonus reranker ablation (`reranker_ablation.csv`) measures the ranked retrieval list
alone, where the reranker lifts Recall@5 from 0.958 to 1.000 and MRR@10 from 0.931 to 0.951.

**Kaggle.** Public score 0.930 (rank 6). No question IDs or question text appear in the source
code. The only question IDs in the repository are in a printed `submission.csv` preview in the
notebook output.

**Safety (C4).** Appendix C units (`DOC3|Figure C-1`, `DOC3|p7|c0`, `DOC3|p7|c1`) are excluded from
evidence, and no golden-set answer cites them (last notebook cell). The instruction inside the Figure
C-1 image was not followed (G25), and a false premise was corrected (G24).

---

## Part 2: Diffusion fine-tuning (27)

Notebooks: `part2/member-anushka/notebooks/p2_finetune.ipynb` (data, training, generation) and
`p2_evaluate.ipynb` (metrics, safety, human evaluation, Hugging Face release).
Summary file: `part2/eval/part2_final_results_anushka.json`.

| Task | Requirement | Evidence |
|---|---|---|
| 2.1 | Clean data, report removals, 90/10 split with seed 266, 512×512, two biases | 1,221 images, 0 corrupt and 0 duplicates removed, 1,098 train / 123 test; `eval/part2_data_and_decision.json`; `README.md` |
| 2.2.1–2 | Two runs, same budget; params, LR, steps, batch, peak memory, time | LoRA rank 4 vs rank 16, 1,500 steps, lr 1e-4, batch 1 × accumulation 4; table in `p2_finetune.ipynb` |
| 2.2.3 | Both models on Hugging Face, model card for the final run | `links.txt` (`Anushka1202/data266-lab2-part2-lora-rank4`, `-rank16`) |
| 2.3 | 20 fixed prompts, seeds 0/1, 40 images per model, grids for 8 prompts, image dataset with metadata | `eval/p2_fixed_20_prompts.json`; grids in `p2_finetune.ipynb`; dataset link in `links.txt` |
| 2.4 | Metrics, failure modes, final-run choice, safety | Table below; `failure_analysis.md` |

### Results (source: `part2/eval/part2_final_results_anushka.json`)

| Measurement | Base SD 1.5 | LoRA rank 4 | LoRA rank 16 (final) |
|---|---|---|---|
| Primary: KID, in-domain (mean ± std over 100 subsets) | 0.142 ± 0.026 | 0.034 ± 0.015 | 0.035 ± 0.016 |
| Primary: CLIP score (20 prompts) | 0.313 | 0.307 | 0.304 |
| Secondary: LPIPS seed diversity | 0.690 | 0.692 | 0.676 |
| Human: final-run win rate vs base (25 pairs, 3 raters incl. 1 outside, ties excluded) | reference | not rated | 42.6% (Fleiss' κ 0.086) |
| Efficiency: GPU-hours / peak memory (T4) | – | 0.81 h / 6.15 GiB | 0.81 h / 6.17 GiB |

**Safety (C4).** Naruto images are copyrighted, so both models are for non-commercial research only.
The Stable Diffusion safety checker flagged 2 of 40 rank-4 outputs (false positives) and 0 of 40
rank-16 outputs. Flagged images are excluded from the released dataset. Memorization check: the
highest CLIP similarity between a final-run generation and a training image is 0.928, below the
95th percentile for real held-out images (0.943).

---

## Part 3: Multimodal multi-agent system (28)

Code: `part3/member-anushka/p3/`. Notebooks: `notebooks/p3_agents.ipynb`, `notebooks/p3_human_eval.ipynb`.
Architecture diagram: `part3/member-anushka/docs/p3_architecture.png`. System card: `part3/member-anushka/README.md` → "System card".

| Task | Requirement | Evidence |
|---|---|---|
| 3.1 | Planner, Vision, Prompt Engineering, Image Generation (img2img, InstructPix2Pix edit, Part 2 LoRA), Critique (CLIP + VLM + safety) | `p3/` agents; `README.md` architecture section |
| 3.2.1 | One shared typed state | `p3/state.py` (pydantic `AgentState`), `p3/graph.py` |
| 3.2.2 | At most 2 revisions; JSON trace per request | `p3/graph.py`; `outputs/traces/*.json` |
| 3.2.3 | Clear messages for ambiguous or broken input | Broken-input scenarios and red-team cases in `outputs/results.json` |
| 3.2.4 | Gradio app | `python -m p3.app` (see README); Space build in `space/` |
| 3.3 | 15 scenarios, 10 red-team cases, with and without critique | `eval/scenarios.json`, `eval/redteam.json`; `outputs/results_table_final.md` |

### Results (source: `part3/member-anushka/outputs/results_table_final.md`)

| Measurement | Baseline (no critique loop) | Final (critique loop, ≤2 revisions) |
|---|---|---|
| Primary: task success | 12/15 | 14/15 |
| Secondary: critic agreement | n/a | 0.71 (n = 14) |
| Secondary: red-team attack success | 1/10 | 1/10 |
| Human: member agreement on pass/fail | 83.3% (25/30) | 83.3% (25/30) |
| Efficiency: median s/request | 4.9 | 6.4 |

Disagreements between members and how they were resolved: `part3/member-anushka/human_eval/scenario_judgements.csv`.
