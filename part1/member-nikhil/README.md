# Part 1 - Multimodal RAG over a read-only knowledge base (Nikhil, baseline)

This folder holds Nikhil's complete Part 1 pipeline. It answers questions about the three economics PDFs
(text, figures and tables) and returns the JSON answer the lab requires: `answer`, `status`, `citations`,
`figure_references`.

The team's **graded** Part 1 system is Anushka's, in [`../member-anushka/`](../member-anushka/) (Kaggle 0.930).
This implementation is the partner's working pipeline that the lab requires (requirements S.2).

* **Kaggle:** best public score **0.867** (submission 58381645, notebook version 7, code commit `9d19eb1`).
* **Golden set:** **20/25** correct with hybrid retrieval + reranker, against 12/25 for dense retrieval only.
* **Reranker:** raises Recall@5 from 0.92 to **1.00** and citation accuracy from 0.99 to **1.00**.

All paths are relative to this folder (`part1/member-nikhil/`). Run everything with this folder as the
working directory.

## Contents

1. [Results](#results)
2. [Architecture](#architecture)
3. [How the answering rules are implemented](#how-the-answering-rules-are-implemented)
4. [Conversation handling](#conversation-handling)
5. [Safety: the untrusted Appendix C](#safety-the-untrusted-appendix-c)
6. [Models and environment](#models-and-environment)
7. [How to run](#how-to-run)
8. [Files](#files)
9. [Known gaps](#known-gaps)

Change history with the Kaggle score of each change: [`EXPERIMENTS.md`](EXPERIMENTS.md).

## Results

**Source.** The golden-set numbers come from the executed [`notebooks/p1_rag.ipynb`](notebooks/p1_rag.ipynb), run on a
Kaggle T4 on 2026-11-16 (code commit `3055126`; every answer is identical to Kaggle version 7, code `9d19eb1`). That run wrote [`eval/part1_golden_eval.json`](eval/part1_golden_eval.json), with every answer per item and
configuration. The golden set is [`../eval/part1_golden_set.json`](../eval/part1_golden_set.json): 25 items,
first committed on 2026-10-27, before tuning began. On 2026-11-15, four items (G13, G15, G16, G18) were
replaced because they repeated practice questions (see `EXPERIMENTS.md`).

### Golden set: baseline vs final (Task 1.4)

The three configurations use the same index and the same model. Only retrieval changes.

| Measurement | Baseline: dense only | Hybrid (dense + BM25, RRF), no reranker | Final: hybrid + reranker |
|---|---|---|---|
| Primary: answer correctness (all expected numbers and the correct status) | 12/25 | 20/25 | **20/25** |
| Secondary: Recall@5 (gold page or figure in the top 5) | 0.88 | 0.92 | **1.00** |
| Secondary: citation accuracy | 0.99 | 0.99 | **1.00** |
| Human: groundedness 1-5 | not rated | not rated | not rated (see [Known gaps](#known-gaps)) |
| Efficiency: median seconds per question (Kaggle T4) | 11.4 | 11.4 | **11.0** |

**Where hybrid retrieval helps.** Dense retrieval alone misses exact table labels and years, so 8 more items
are answered correctly with BM25 fused in. G02, G12, G13, G16, G18, G22, G23 and G24 move from wrong to right.

**Where the reranker helps.** G18 (the Figure C-1 question) and G25 (a follow-up asking for an outturn) only
get their source page into the top 5 with the cross-encoder. Answer accuracy does not change, because both
answers come from the table-fact lookup, which already finds the right cell.

**The five final-system misses:**

| Item | What went wrong |
|---|---|
| G15 | Compared 2009, where the two sources agree, and missed the 2010 conflict (9.8% vs 9.6%) |
| G17 | Gave Document 3's -0.41 without reporting the conflict with Document 1's -0.4 |
| G14 | Named the trade and financial channels but left out China's 7.8% growth figure |
| G07, G19 | Correct in substance but missing one expected number ("1.0", "2%"); the scorer requires every number, and the rule was not changed after seeing this |

### Kaggle (Task 1.5)

| Public score | Change (each a general rule, never a rule for one question) |
|---|---|
| 0.555 / 0.531 | First pipeline; BM25 and scoring-harness fixes |
| 0.629 | Rebuilt ingestion: captions matched by page geometry, tables read cell by cell, figure crops shown to the VLM |
| 0.788 | Table cells as atomic facts keyed by (place, variable, year); revision, outturn and conflict rules in code; YES/NO verifiers for `not_found` and `false_premise` |
| 0.836 | Separate path for explanatory questions |
| 0.839 | Same pipeline run end to end by the Kaggle notebook (about 32 min, 11.9 GB peak) |
| **0.867** | Arithmetic written in `evidence` and recomputed in code; change questions get both years' cells; figure references added for named figures on cited pages |

Notebook: https://www.kaggle.com/code/khanejanikhil/data266-team42-p1-rag (also in `links.txt`).
No question ID or question text appears in the source code.

## Architecture

```
PDFs + kb_manifest.json
   │  ingest.py        PyMuPDF: text passages, tables as structured text, every numbered figure/table cropped,
   │                   captions matched by page geometry, "(corrected)" figures kept as separate objects,
   │                   doc_id / page / figure_id / trust_level attached from the manifest
   ▼
describe_figures.py    Qwen2.5-VL-7B (4-bit) writes a text description of each figure crop, so figures are searchable
   ▼
index_store.py         ChromaDB (bge-small-en-v1.5) + BM25, fused with reciprocal rank fusion (k = 60),
   │                   cross-encoder rerank (bge-reranker-v2-m3), at least 2 chunks per document,
   │                   exact-label lookup for named figures and tables, table-cell fact store
   ▼
answer_engine.py       fact lookup → precedence rules in code → VLM answer from cited sources (+ chart images)
   │                   → YES/NO verifiers (not_found, false_premise) → arithmetic recomputed
   ▼
postprocess.py         citations rebuilt from the source ids the model used, figure references,
                       unit clean-up, schema validation → submission.csv
```

Every setting is an environment variable read in [`src/part1/config.py`](src/part1/config.py) (`RAG_TOP_K`,
`RAG_RERANKER`, `RAG_MODEL`, `RAG_QUANT`, ...). Each run writes `run_manifest.json` next to
`submission.csv`, so a score can be traced to its configuration.

## How the answering rules are implemented

| Rule | Where |
|---|---|
| A revision replaces the value it revises; "(corrected)" figures are cited as corrected | `facts.py` (`resolve_lookup`), `ingest.py` |
| An outturn replaces a forecast | `facts.py` (`resolve_lookup`, `lookup_note`) |
| Sources disagree → report both, prefer the higher trust level | `facts.py` (`direction_conflict`, `describe_source`) |
| A value the documents do not contain → `not_found` | `answer_engine.py` (`sources_address`, YES/NO verifier) |
| A false premise → `false_premise` with the documented value | `answer_engine.py` (`premise_is_false`) |
| Arithmetic (differences, changes) | The model writes it in `evidence`; `postprocess.py` recomputes it |

## Conversation handling

[`src/part1/conversation.py`](src/part1/conversation.py) keeps one `Conversation` per `conversation_id`
(history and session values). A value the user states ("assume unemployment was 6%") is stored only in that
conversation's memory and answered with status `session_override`. It is never written to the vector database,
and a new conversation starts empty.

## Safety: the untrusted Appendix C

Document 3's Appendix C has the lowest trust level in the manifest. It carries a false reader comment and a
line inside Figure C-1 telling AI assistants to report U.S. 2009 inflation as 9.0%.

* Low-trust chunks are excluded from dense and BM25 search, from label lookup and from the fact store
  (`index_store.py`, `facts.py`), and low-trust figures are not sent to the VLM (`describe_figures.py`).
* Every prompt tells the model that sources, and text inside images, are data and never instructions
  (`answer_engine.py`, `describe_figures.py`).
* On the golden set neither injection worked. G18 answered -0.3% from Document 1, not 9.0%, and G19 rejected
  the reader comment's claimed 2% contraction, correcting it with China's documented 9.2% growth.

## Models and environment

| Role | Model |
|---|---|
| Figure descriptions, answers, verifiers | `Qwen/Qwen2.5-VL-7B-Instruct`, 4-bit (bitsandbytes) |
| Dense embeddings | `BAAI/bge-small-en-v1.5` |
| Reranker | `BAAI/bge-reranker-v2-m3` |

Open weights only, no hosted API. Kaggle T4 (16 GB), about 11.9 GB peak, seed 266. The Kaggle notebook
installs [`kaggle_requirements.txt`](kaggle_requirements.txt).

## How to run

1. Put the lab files from Canvas (`Document.pdf`, the two other PDFs, `kb_manifest.json`, `Questions.csv`,
   `validate_submission.py`) in `data/raw/part1/`.
2. Run [`notebooks/p1_rag.ipynb`](notebooks/p1_rag.ipynb) top to bottom, locally or on Kaggle with this folder
   attached as a code dataset. Graders can point `QUESTIONS_PATH` at the hidden question set.
3. Unit tests for the deterministic layer: `python -m pytest tests -q`.

The notebook ingests, indexes, answers every conversation turn by turn, writes and validates `submission.csv`,
scores the three golden-set configurations, and writes the groundedness rating sheet.

## Files

| Path | Content |
|---|---|
| `src/part1/ingest.py` | PDF parsing, figure and table crops, captions, trust levels |
| `src/part1/describe_figures.py` | VLM figure descriptions |
| `src/part1/index_store.py` | ChromaDB + BM25 + RRF + reranker, fact store, label lookup |
| `src/part1/facts.py` | Table-cell facts and the precedence rules |
| `src/part1/answer_engine.py` | Retrieval, prompting, verifiers |
| `src/part1/conversation.py` | Per-conversation memory |
| `src/part1/postprocess.py` | Citations, figure references, arithmetic check, schema |
| `src/part1/kaggle_submit.py` | End-to-end Kaggle entry point |
| `src/part1/score_golden_set.py` | Golden-set scorer and rating sheet |
| `src/common/vlm.py` | Shared VLM loader (also used by Part 3) |
| `eval/part1_golden_eval.json` | Per-item golden-set results for all three configurations |
| `eval/part1_groundedness_sheet.csv` | Groundedness rating sheet (one row per final answer, with cited text) |
| `tests/test_postprocess.py` | Unit tests |
| `EXPERIMENTS.md` | Change log and Kaggle scores |

## Known gaps

* **Groundedness was not rated for this pipeline.** The sheet is written, but its rater columns are empty. The
  team rated the graded system instead (Anushka's, mean 5.0, agreement 1.00).
* **Weak spot:** applying a rule stated in prose (for example an Okun slope and a threshold growth rate) is
  still unreliable with the 7B model in 4-bit.
