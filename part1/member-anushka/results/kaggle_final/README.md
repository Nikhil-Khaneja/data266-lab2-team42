# Final Kaggle submission (Part 1)

`submission.csv` is the file selected as the team's final submission on the leaderboard.

| | |
|---|---|
| Public score | 0.930 (rank 6) |
| Kaggle notebook | https://www.kaggle.com/code/anushkak12345/notebook1f4786ea97 |
| Hardware | Kaggle Tesla T4, about 22 minutes for the full notebook |
| System | hybrid dense + BM25, `BAAI/bge-reranker-v2-m3` reranker, `Qwen/Qwen2.5-VL-3B-Instruct` fp16 |
| Validation | passes `validate_submission.py` (40 rows) |

The graded run uses exactly the `p1_rag` code in the repository (commit `921bfb7`); Cell 1 prints
`code@921bfb7`. Its 40 answers are identical to the earlier 0.930 run (`anushkak12345/notebook1e60e8258f`),
which was built from a slightly older copy of the same code (one comment and the wording of one Okun's-law
sentence differed).
