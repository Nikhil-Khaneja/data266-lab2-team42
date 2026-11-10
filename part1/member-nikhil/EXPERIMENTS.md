# Part 1 experiment log

Score per question = 0.40 answer + 0.20 status + 0.20 figure_references + 0.20 citations.
The public leaderboard scores only part of the 40 questions, and the hidden set rewords them, so every
change below is a general rule (by question shape, source type or arithmetic), never a rule for one question.

Settings are environment variables read in `src/part1/config.py` (`RAG_TOP_K`, `RAG_NARRATIVE_TOP_K`,
`RAG_MIN_PER_DOC`, `RAG_MAX_IMAGES`, `RAG_MAX_CITATIONS`, `RAG_FACT_TOP_K`, `RAG_MAX_CONTEXT_CHARS`,
`RAG_EMBEDDER`, `RAG_RERANKER`, `RAG_MODEL`, `RAG_QUANT`). Each run writes `run_manifest.json`.

| Public score | Change |
|---|---|
| 0.555 / 0.531 | Earlier pipeline (BM25 fix; scoring-harness fixes) |
| 0.629 | Rebuilt ingestion: captions matched to figures by page geometry, structured tables read cell by cell, figures cropped and shown to the VLM; Qwen2.5-VL-7B 4-bit; citations built from the source ids the model picks |
| 0.788 | Table cells as atomic facts keyed by (place, variable, year); revision, outturn-over-forecast and conflict decided in code; focused chart reader; YES/NO verifiers for not_found and false_premise |
| 0.836 | Separate path for explanatory questions (own prompt and passages) |
| 0.839 | Same pipeline, produced end to end by the Kaggle notebook on a T4 (about 32 min including golden-set evals, 11.9GB peak) |
| 0.867 | The model writes its arithmetic in `evidence` and code recomputes it; a difference states its inputs; change questions get both years' cells; a figure or table named in an answer on a cited page is referenced; the chart reader reports a value for every item; currency units no source prints are removed |

Known weak spot: applying a rule stated in prose (for example a slope and a threshold growth rate)
to a value is still unreliable with the 7B model.

Unit tests for the deterministic layer: `python -m pytest tests -q` from this folder.
