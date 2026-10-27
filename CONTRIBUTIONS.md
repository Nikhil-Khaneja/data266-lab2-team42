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
- LoRA training launcher, generation script, evaluation metrics (KID/CLIP/LPIPS,
  memorization check), blinded human-eval tooling, Hugging Face push script: implemented.

## Part 3 — Multimodal Multi-Agent System

- Five agents (planner, vision understanding, prompt engineering, image generation,
  critique), orchestrator with revise loop and JSON tracing, safety guardrails, Gradio
  demo app: implemented and smoke-tested against the local VLM.
- 15 test scenarios and 10 red-team cases drafted; pending real sample images to run.

## Meeting recordings

| Date | Topic | Link |
|------|-------|------|
| TBD  | TBD   | TBD  |
