# Part 1 - final results (Anushka's system, the team's final Part 1)

**Where these come from.** The graded Kaggle run (`part1/member-anushka/notebooks/p1_rag.ipynb`, Kaggle
notebook `anushkak12345/notebook1e60e8258f`, Tesla T4, seed 266). The CSV files are the run's own
outputs, copied unchanged.

**Leaderboard.** Public score **0.930**, rank **4**. The submitted file is
`part1/member-anushka/results/kaggle_final/submission.csv`.

## Golden set (25 questions): baseline vs final

| measurement | baseline (dense only) | hybrid, no reranker | final (hybrid + reranker) |
|---|---|---|---|
| answer correctness (numbers ±1% and right status) | 11/25 | 25/25 | **25/25** |
| status accuracy | 0.76 | 1.00 | **1.00** |
| Recall@5 (24 questions with a retrieval target) | 0.833 | 0.958 | **0.958** |
| citation accuracy (gold citations ÷ all citations) | 0.760 (38/50) | 0.887 (55/62) | **0.887 (55/62)** |
| median seconds per question | 2.59 | 3.53 | **4.79** |

**Human evaluation.** Both members rated the 25 final answers 1–5 for groundedness: mean 5.0, agreement
within one point 1.00, exact agreement 1.00 (`part1/member-anushka/notebooks/p1_human_eval.ipynb`,
ratings in `part1/member-anushka/eval/p1_human_ratings.csv`).

## Bonus: reranker ablation (retrieval only)

| mode | Recall@1 | Recall@3 | Recall@5 | MRR@10 | median retrieval ms |
|---|---|---|---|---|---|
| dense only | 0.583 | 0.833 | 0.833 | 0.709 | 16.0 |
| hybrid (dense + BM25, RRF) | 0.917 | 0.958 | 0.958 | 0.931 | 17.7 |
| hybrid + `bge-reranker-v2-m3` | 0.917 | 1.000 | 1.000 | 0.951 | 353.8 |

## Files

| file | content |
|---|---|
| `golden_scored_final.csv` | per question, final system: status, correctness, missing numbers, Recall@5, citations and how many are supported, seconds, answer |
| `golden_scored_hybrid_no_reranker.csv` | the same for the hybrid system without the reranker |
| `golden_scored_baseline.csv` | the same for the dense-only baseline |
| `reranker_ablation.csv` | the retrieval ablation table above |

The golden questions themselves are in `part1/member-anushka/eval/p1_golden_set.csv`.
