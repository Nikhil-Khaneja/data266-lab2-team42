# Contributions

Team: Pair_Programming_Team_42 (Nikhil Khaneja, Anushka Khadatkar)

Both members contribute to all three parts. Nikhil leads Part 3 (agents); Anushka leads
Part 2 (fine-tuning). Part 1 is shared. The partner not leading a part still implements
and runs a complete working baseline for it.

Each part has one directory per member (`part1/member-nikhil/`, `part1/member-anushka/`,
and so on) so both implementations live side by side in the same repo. Everything below
describes the work currently in each `member-nikhil/` folder; the `member-anushka/`
folders are scaffolded and pending her own implementation.

## Part 1 — MultiModal RAG

- Ingestion (PDF text/figure extraction), hybrid retrieval (dense + BM25 + ChromaDB),
  conversation memory, answering-rules engine, Kaggle submission script, golden-set
  scoring harness: implemented and run against the real documents.
- 25-question golden set written and verified by hand against the real documents.
- Current real results (both baseline-dense and final-hybrid): 11/25 correct, 72%
  Recall@5, 100% citation accuracy, ~2.5s median per question. Baseline and final match
  on this small a corpus (78 chunks) — hybrid retrieval and dense-only retrieval return
  the same top-5 results for nearly every golden-set question here.
- Found and fixed several real bugs during evaluation: 800-char chunking was splitting
  tables mid-row (caused wrong numbers); DOC1's vector-drawn charts weren't extracted at
  all (get_images() only finds embedded raster images); generation wasn't deterministic
  (same question, different answers across runs); several golden-set scoring bugs.
- Ran the real Kaggle submission pipeline locally against the 40-question practice set
  (`Questions.csv`) and the lab's own `validate_submission.py`: found that 18/40 (45%)
  of citations and 1/40 figure references failed the required strict format, because
  the model was copying stray text (a " trust=high" annotation, or a citation-style
  "DOCn:" prefix) out of the retrieved-context labels into its output. Fixed with a
  tightened prompt plus a regex sanitizer applied to every answer. This also raised
  golden-set citation accuracy from 86% to 100%, since the same malformed citations
  were failing the golden-set checker too. `submission.csv` passed
  `validate_submission.py` cleanly (40/40 valid rows) and was submitted to the real
  competition (`data-266-fall-2026-lab-2-part-1-multimodal-rag`): **public leaderboard
  score 0.571** (submission ref 57729902, 2026-10-28).

## Part 2 — Diffusion Fine-Tuning

- Data prep (dedup/corrupt filtering, 90/10 split, resize) run against the real
  lambdalabs/naruto-blip-captions dataset: 1221 images, 0 removed, 1098 train / 123 test.
- 20 fixed evaluation prompts generated from real test-set captions.
- Two real LoRA training runs completed on an RTX 5090, both under the 16GB GPU budget
  (gradient checkpointing + fp16): run1 (rank 4) — 797,184 trainable params, 7.3GB peak,
  946s; run2 (rank 16) — 3,188,736 trainable params, 8.6GB peak, 1046s.
- Generated all 120 images (base + both runs x 20 fixed prompts x 2 seeds) and ran the
  real evaluation: KID (in-domain generations vs. real test images) drops from 0.119
  (base) to 0.050 (run1) to 0.024 (run2); CLIP score holds steady around 0.30-0.31 for
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
