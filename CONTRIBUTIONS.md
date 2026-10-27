# Contributions

Team: Pair_Programming_Team_42 (Nikhil Khaneja, Anushka Khadatkar)

Both members contribute to all three parts. Nikhil leads Part 3 (agents); Anushka leads
Part 2 (fine-tuning). Part 1 is shared. The partner not leading a part still implements
and runs a complete working baseline for it.

## Part 1 — MultiModal RAG

- Ingestion (PDF text/figure extraction), hybrid retrieval (dense + BM25 + ChromaDB),
  conversation memory, answering-rules engine, Kaggle submission script, golden-set
  scoring harness: implemented and run against the real documents.
- 25-question golden set written and verified by hand against the real documents.
- Current real results (both baseline-dense and final-hybrid): 11/25 correct, 72%
  Recall@5, 86% citation accuracy, ~1.9s median per question. Baseline and final match
  on this small a corpus (78 chunks) — hybrid retrieval and dense-only retrieval return
  the same top-5 results for nearly every golden-set question here.
- Found and fixed several real bugs during evaluation: 800-char chunking was splitting
  tables mid-row (caused wrong numbers); DOC1's vector-drawn charts weren't extracted at
  all (get_images() only finds embedded raster images); generation wasn't deterministic
  (same question, different answers across runs); several golden-set scoring bugs.

## Part 2 — Diffusion Fine-Tuning

- Data prep (dedup/corrupt filtering, 90/10 split, resize) run against the real
  lambdalabs/naruto-blip-captions dataset: 1221 images, 0 removed, 1098 train / 123 test.
- 20 fixed evaluation prompts generated from real test-set captions.
- Two real LoRA training runs completed on an RTX 5090, both under the 16GB GPU budget
  (gradient checkpointing + fp16): run1 (rank 4) — 797,184 trainable params, 7.3GB peak,
  946s; run2 (rank 16) — 3,188,736 trainable params, 8.6GB peak, 1046s.
- Generation script, evaluation metrics (KID/CLIP/LPIPS, memorization check), blinded
  human-eval tooling, Hugging Face push script: implemented, not yet run.

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
