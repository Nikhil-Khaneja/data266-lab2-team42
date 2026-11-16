# Part 1 - Multimodal RAG (Anushka) - final Part 1 system for `p1-v1.0`

Hybrid dense + BM25 retrieval with a cross-encoder reranker, exact figure/table routing, a structured fact
store with precedence rules (revision > outturn > forecast; conflicts prefer higher trust), per-conversation
memory and deterministic answer guards. Public Kaggle score 0.930 (rank 4 at the time of writing).

All paths below are relative to this folder (`part1/member-anushka/`).

## Models (open weights, no hosted API)

| Use | Model (version) | Where it runs |
|---|---|---|
| Figure descriptions + answers | `Qwen/Qwen2.5-VL-3B-Instruct` (Hugging Face), fp16 | one Kaggle T4/P100 GPU |
| Text embeddings | `BAAI/bge-small-en-v1.5` | same GPU / CPU |
| Reranker (bonus) | `BAAI/bge-reranker-v2-m3` cross-encoder, fp16 on GPU | same GPU / CPU |
| Local development only | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | Apple Silicon (not used for graded numbers) |

`requirements.txt` pins the local development environment. The graded Kaggle run keeps the Kaggle image's
preinstalled torch/transformers (printed by Cell 1 and the next cell of the notebook) and installs only the
pinned extras listed in the notebook's install cell.

## How to run

**On Kaggle (graded run).** Open the competition, create a notebook from `notebooks/p1_rag.ipynb` at tag `p1-v1.0`,
attach the competition data, set *Accelerator = GPU T4 x1 (or P100)* and *Internet = on* (to download the three
models), then *Run All*. The notebook finds the data under `/kaggle/input/`, reads questions from `QUESTIONS_PATH`
(default `Questions.csv`, falling back to the competition copy), writes `submission.csv` and runs
`validate_submission.py`. Expected runtime is well under the 90-minute limit.

**Locally.**
```bash
python3.11 -m venv .venv && .venv/bin/pip install -r requirements.txt
# put the Part 1 files (3 PDFs, kb_manifest.json, Questions.csv, validate_submission.py) in data/
.venv/bin/python scripts/p1_run.py              # ingestion -> index -> submission -> golden-set evaluation
.venv/bin/python scripts/build_p1_notebook.py   # regenerate notebooks/p1_rag.ipynb from p1_rag/
```

## Layout

| Path | What |
|---|---|
| `p1_rag/ingest.py` | PDF text chunks, Figure/Table crops with labels + captions, table grid -> structured facts (1.1) |
| `p1_rag/figures.py` | VLM text descriptions of every figure; embedded instructions fenced off (1.1.3) |
| `p1_rag/index.py` | ChromaDB + bge-small dense, BM25, RRF fusion, exact-label routing, low-trust filter, cross-encoder reranker over the top 20 hybrid candidates (1.1.4, 1.2, bonus) |
| `p1_rag/facts.py` | precedence rules: revision > outturn > forecast; conflicts prefer higher trust (1.2.3) |
| `p1_rag/answer.py` | `RAGSystem` / `Conversation`: JSON answers, per-conversation memory, session overrides (1.2, 1.3) |
| `p1_rag/evaluate.py` | submission writer, golden-set metrics, human-evaluation summary, reranker ablation (Recall@1/3/5, MRR@10) (1.4, 1.5, bonus) |
| `eval/p1_golden_set.csv` | our 25-question golden set, written from the three PDFs on 2026-11-11 - after tuning on the public practice questions had started (disclosed in the report); three questions reworded later so none resembles a practice question, answers unchanged |
| `eval/p1_human_ratings.csv` | groundedness ratings by both members (1-5) |
| `scripts/p1_run.py` | local end-to-end run (ingestion, index, submission, golden-set evaluation, reranker ablation) |
| `scripts/build_p1_notebook.py` | builds `notebooks/p1_rag.ipynb` from `p1_rag/` and `eval/`; `--commit <hash>` sets the commit Cell 1 prints on Kaggle |
