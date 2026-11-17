# Contributions

Team: Pair_Programming_Team_42 (Nikhil Khaneja, Anushka Khadatkar)

Both members contribute to all three parts, and each member built a complete, executed pipeline for every
part. Anushka led Part 2 (fine-tuning). Nikhil built the first Part 3 pipeline and the shared Part 3 test set.
Part 1 is shared. For each part the team compared the two implementations and graded the stronger one; the
other is kept as a working baseline:

| Part | Graded (final) system | Baseline |
|---|---|---|
| 1 | `part1/member-anushka/` (Kaggle public score 0.930) | `part1/member-nikhil/` (0.867) |
| 2 | `part2/member-anushka/` (LoRA rank 16) | `part2/member-nikhil/` |
| 3 | `part3/member-anushka/` | `part3/member-nikhil/` |

Each part has one directory per member (`part1/member-nikhil/`, `part1/member-anushka/`,
and so on) so both implementations live side by side in the same repo. Unless a section
says otherwise, the work below is in the `member-nikhil/` folders. `part1/member-anushka/`,
`part2/member-anushka/` and `part3/member-anushka/` hold Anushka's own implementations.

## Part 1 — MultiModal RAG

Part 1 is shared. Both members built a complete pipeline.

### Nikhil — `part1/member-nikhil/`

All numbers below are printed by `part1/member-nikhil/notebooks/p1_rag.ipynb` (executed on a Kaggle T4 on 2026-11-16,
code at commit `3055126`) and saved in `part1/member-nikhil/eval/`. Change-by-change history: `part1/member-nikhil/EXPERIMENTS.md`.

- Ingestion with PyMuPDF: text passages, numbered tables read cell by cell, every numbered figure and
  table cropped, with captions matched to figures by page geometry. Figures are searchable through
  VLM-written descriptions.
- Retrieval: ChromaDB (bge-small) plus BM25, fused with RRF and reranked by a cross-encoder
  (bge-reranker-v2-m3). Low-trust content (Document 3, Appendix C) is filtered before ranking.
- Answering with Qwen2.5-VL-7B (4-bit). Table cells become facts keyed by place, variable and year,
  so revisions, outturns that replace forecasts, and conflicts are decided in code. Named figures go to
  a focused chart reader. The model writes its arithmetic and code recomputes it. Conversation memory
  is kept per `conversation_id`, and user-supplied values stay in their own conversation.
- Golden set of 25 questions (`part1/eval/part1_golden_set.json`). Four items that overlapped
  practice questions were replaced on 2026-11-15; see EXPERIMENTS.md.
- Results on the golden set:

  | System | Correct (/25) | Recall@5 | Citation accuracy | Median s/question |
  |---|---|---|---|---|
  | Baseline: dense retrieval only | 12 | 0.88 | 0.99 | 11.4 |
  | Hybrid, no reranker | 20 | 0.92 | 0.99 | 11.4 |
  | Final: hybrid + reranker | 20 | 1.00 | 1.00 | 11.0 |

- Kaggle: best public score 0.867, from the Kaggle notebook (submissions 58293679 and 58381645). The full
  run answers the 40 questions in about 6 minutes with 12.0 GB peak GPU memory.
- Safety: neither the reader-comment false claim nor the instruction printed inside Figure C-1 was
  followed on the golden set (G18, G19).
- Human evaluation: groundedness sheet with the cited text for each answer
  (`part1/member-nikhil/eval/part1_groundedness_sheet.csv`), rated 1-5 by both members.

### Anushka — `part1/member-anushka/`

This is the team's final Part 1 system (tag `p1-v1.0`). All numbers below are printed by
`part1/member-anushka/notebooks/p1_rag.ipynb` (Kaggle T4, notebook `anushkak12345/notebook1f4786ea97`) and
saved in `part1/eval/final_results/`. Details: `part1/member-anushka/README.md`,
`part1/member-anushka/failure_analysis.md`, `part1/member-anushka/AI_USAGE.md`.

- **Ingestion.**
  - PyMuPDF text chunks with document and page.
  - Every numbered figure and table cropped with its exact label and caption; captions are found by
    their bold font, and `Figure B-5` / `Figure B-5 (corrected)` are separate objects.
  - Table grids parsed into structured facts.
  - Qwen2.5-VL-3B descriptions make figures searchable; instruction-like text inside images is fenced
    off as data.
- **Retrieval.**
  - ChromaDB (bge-small) plus BM25, fused by RRF; `bge-reranker-v2-m3` over the top 20 candidates.
  - A question that names a figure or table gets exactly that object, and values come only from it.
  - Low-trust Appendix C content is never returned as evidence.
- **Answering.**
  - Precedence (revision > outturn > forecast; conflict → higher trust) and all arithmetic are decided
    in code.
  - Deterministic guards set the status, citations, session-value statements and refusals.
  - Qwen2.5-VL-3B (fp16) writes the JSON answer.
  - One memory per `conversation_id`; user values never enter the vector database.
- **Golden set.**
  - 25 questions in `part1/member-anushka/eval/p1_golden_set.csv`, written on 2026-11-11 after tuning
    on the practice questions had started; this is disclosed.
  - Results on it:

  | System | Correct (/25) | Status accuracy | Recall@5 | Citation accuracy | Median s/question |
  |---|---|---|---|---|---|
  | Baseline: dense retrieval only | 11 | 0.76 | 0.833 | 0.760 | 2.76 |
  | Hybrid, no reranker | 25 | 1.00 | 0.958 | 0.887 | 3.77 |
  | Final: hybrid + reranker | 25 | 1.00 | 0.958 | 0.887 | 5.09 |

- **Reranker (bonus).** Recall@5 0.958 → 1.000 and MRR@10 0.931 → 0.951 on the golden set, in the
  retrieval-only ablation (`part1/eval/final_results/reranker_ablation.csv`). That ablation re-runs the retriever
  alone on each question's query. The Recall@5 in the table above (0.958 for both) counts the first five units each
  answer actually used, from its trace. The two measures differ on one of the 24 questions with a retrieval target.
- **Kaggle.** Public score **0.930**, rank 6: the team's final submission,
  `part1/member-anushka/results/kaggle_final/submission.csv`. The full notebook runs in about 22 minutes
  on a T4.
- **Safety.**
  - The instruction printed inside Figure C-1 was not followed (G25); the answer is the documented value,
    with a note that the image text is unverified.
  - A false premise was corrected (G24).
  - No answer cites Appendix C.
- **Human evaluation.** Both members independently rated the 25 final answers 1–5 for groundedness:
  mean 5.0 (both raters 5.0), agreement within one point 1.00, exact agreement 1.00. Computed by
  `part1/member-anushka/notebooks/p1_human_eval.ipynb`.

## Part 2 — Diffusion Fine-Tuning

### Anushka (lead) — `part2/member-anushka/`

All numbers below are printed by `part2/member-anushka/notebooks/p2_evaluate.ipynb` and saved in
`part2/member-anushka/eval/`.

- Data prep on `lambdalabs/naruto-blip-captions` in Colab: 1,221 images, 0 corrupt and
  0 exact duplicates removed, seed-266 90/10 split into 1,098 train / 123 test, 512x512.
  Near-duplicate audit (64-bit dHash, Hamming <= 5) found 1 test image with a near-duplicate
  in the training split; caption audit found 58.8% of captions mention male terms vs 20.6%
  female.
- Wrote her own 20 prompts (10 held-out test captions, 5 out-of-domain, 5 compositional) and
  froze them in the training notebook before training (`eval/p2_fixed_20_prompts.json`).
- Two LoRA runs on a Colab T4 with the Diffusers `train_text_to_image_lora.py` script, same
  data and 1,500-step budget (lr 1e-4 constant, batch 1 x gradient accumulation 4, fp16,
  8-bit AdamW, gradient checkpointing): rank 4 — 797,184 trainable params, 6.15 GiB peak,
  0.81 GPU-hours; rank 16 — 3,188,736 trainable params, 6.17 GiB peak, 0.81 GPU-hours.
- Generated 120 images (base + both runs x 20 prompts x seeds 0/1, PNDM 40 steps, guidance
  7.5) and side-by-side grids for 8 prompts.
- Evaluation: KID on 20 in-domain generations vs 123 test images (100 subsets of 10) drops
  from 0.142 (base) to 0.034 (rank 4) and 0.035 (rank 16), a rank-4/rank-16 gap far inside
  the subset noise; CLIP score 0.313 / 0.307 / 0.304; LPIPS seed diversity 0.690 / 0.692 /
  0.676.
- Blinded human evaluation: 25 unique pairs (final run vs base, sides randomized), 3 raters
  (Anushka, Nikhil, Ayush outside the team), 75 ratings. Final-run win rate 42.6% excluding
  ties (72.7% on out-of-domain prompts, 28.0% in-domain); Fleiss' kappa 0.086, and a binomial
  test shows two raters with a significant left/right preference, reported as a limitation.
- Safety: Stable Diffusion safety checker on all 120 images — rank 4 had 2 false positives on
  harmless prompts, base and rank 16 had 0; flagged images are excluded from the released
  dataset and were deleted from the first upload. Memorization: the 10 in-domain rank-16
  generations vs all 1,098 training images — highest CLIP similarity 0.928, below the 95th
  percentile of real held-out frames (0.943); smallest LPIPS to a training image 0.553.
- **Rank 16 selected as the final run**: quality, alignment and cost are indistinguishable
  from rank 4, and it is the only run with no NSFW-flagged outputs on the fixed prompts.
  Failure modes and the reasoning are in `part2/member-anushka/failure_analysis.md`.
- Both adapters, the final run's model card and the safety-filtered image dataset (with
  prompt/seed/model metadata) are on Hugging Face (`Anushka1202/...`, see `links.txt`).
- Found and fixed in her own first evaluation pass: KID computed from a single subset
  (std 0), a memorization check that compared real test images instead of generations, and
  human-evaluation pairs 21–25 that repeated pairs 1–5.

### Nikhil (baseline) — `part2/member-nikhil/`

- Data prep (dedup/corrupt filtering, 90/10 split, resize) run against the real
  lambdalabs/naruto-blip-captions dataset: 1221 images, 0 removed, 1098 train / 123 test.
- 20 fixed evaluation prompts generated from real test-set captions.
- Two LoRA training runs on an RTX 4090, both well under the 16 GB GPU budget (gradient
  checkpointing + fp16, 1,200 steps, lr 1e-4, effective batch 4): run1 (rank 4) — 797,184 trainable
  params, 4.12 GB peak, 1,323.8 s (0.368 GPU-hours); run2 (rank 16) — 3,188,736 trainable params,
  4.15 GB peak, 1,304.5 s (0.362 GPU-hours). Numbers from `part2/member-nikhil/notebooks/p2_finetune.ipynb`.
- Generated all 120 images (base + both runs x 20 fixed prompts x 2 seeds) and ran the
  real evaluation: KID (in-domain generations vs. real test images) drops from 0.118
  (base) to 0.051 (run1) to 0.023 (run2); CLIP score holds steady around 0.30-0.31 for
  all three; LPIPS seed diversity dips slightly for the fine-tuned models versus base.
  **run2 (rank 16) selected as the final run** — far better domain fidelity (KID) at
  essentially the same CLIP score as run1.
- Memorization check (10 in-domain generations from the final run vs. the training set
  by CLIP similarity): closest pairs are 0.90-0.92 similarity — notably similar but
  short of near-exact duplication.
- Both LoRA models pushed to Hugging Face with model cards (the final run's card
  includes the real comparison table); all 120 generated images pushed as a Hugging
  Face dataset with prompt/seed/model/category metadata.
- Blinded human-eval pairs (25, base vs. final run, left/right randomized) generated;
  actual rating by both members plus one outside rater is still pending.
- Found and fixed a real bug during evaluation: KID was being computed over all 40
  generated images per model (including out-of-domain and compositional prompts)
  against the test set, instead of just the 10 in-domain ones the lab spec asks for.

## Part 3 — Multimodal Multi-Agent System

### Anushka (final) — `part3/member-anushka/`

All numbers below are printed by `part3/member-anushka/notebooks/p3_agents.ipynb` (Colab T4, code at commit
`40c0a5c`) and `notebooks/p3_human_eval.ipynb`, and saved in `part3/member-anushka/outputs/`. Details:
`part3/member-anushka/README.md` (architecture, results, system card) and `failure_analysis.md`.

- **Agents.** Planner (rule-based routing plus input safety: NSFW terms, deceptive real-person edits, OCR scan
  for instructions in the image), Vision Understanding (Qwen2.5-VL-3B: caption, objects, visible text, at least 2
  answers), Prompt Engineering (image text never copied into the prompt), Image Generation (SD 1.5 img2img,
  InstructPix2Pix edits, the Part 2 rank-16 LoRA for domain-style requests), Critique (CLIP score, VLM judgement,
  SD safety checker, concrete revision instruction).
- **Orchestration.** One pydantic `AgentState` (`p3/state.py`), plain-Python graph with at most 2 revisions
  (`p3/graph.py`), a JSON trace per request (`outputs/traces/`), clear messages for broken, empty or ambiguous
  input, and a Gradio app (`python -m p3.app`; Space build in `space/`).
- **Test set.** 15 scenarios and 10 red-team cases with pass/fail rubrics (`eval/scenarios.json`,
  `eval/redteam.json`).
- **Results.** Task success 14/15 with the critique loop vs 12/15 without (automatic); 12/15 in both versions by
  our agreed human judgement. Critic agreement 10/14. Red-team attack success 1/10 (R01). Median 6.4 s per request
  with the loop, 4.9 s without.
- **Human evaluation.** Both members judged all 30 scenario outputs and 14 critic decisions independently on a
  page that hid the automatic results: 25/30 scenario and 12/14 critic agreement; every disagreement was discussed
  and resolved (`human_eval/scenario_judgements.csv`, `human_eval/critic_labels.csv`).

### Nikhil (baseline) — `part3/member-nikhil/`

- Five agents (planner, vision understanding, prompt engineering, image generation,
  critique), orchestrator with revise loop and JSON tracing, safety guardrails, Gradio
  demo app: implemented and run end-to-end against the local VLM and SD1.5 + LoRA.
- 15 test scenarios and 10 red-team cases run against self-generated fixture images
  (no external downloads needed). All 15 scenarios complete without crashing, including
  the two broken/missing-image cases, which refuse cleanly instead of crashing the batch.
  All 10 red-team cases are handled correctly: direct unsafe requests and deceptive edits
  of real people are refused by the planner's safety check; the three image-embedded
  prompt-injection attacks ("ignore previous instructions", a fake system prompt, a fake
  developer override) are captured verbatim as data in the vision agent's output but never
  change the caption or the route, so the architecture resists them without needing an
  explicit refusal. Found and fixed two real bugs from this run: a crash on a corrupted
  (not just missing) image file, and two safety-regex gaps ("graphic violence" and
  "celebrity ... compromising" phrasing that the original patterns missed). Final
  pass/fail judgment against the written rubric and the two instructions-in-image calls
  still needs sign-off from the team.

## Meeting recordings

| Date | Topic | Link |
|------|-------|------|
| TBD  | TBD   | TBD  |
