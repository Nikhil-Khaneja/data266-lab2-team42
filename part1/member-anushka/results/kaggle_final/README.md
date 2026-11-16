# Final Kaggle submission (Part 1)

`submission.csv` is the file selected as the team's final submission on the leaderboard.

| | |
|---|---|
| Public score | 0.930 (rank 4 at submission time) |
| Kaggle notebook | https://www.kaggle.com/code/anushkak12345/notebook1e60e8258f |
| Hardware | Kaggle Tesla T4, about 21 minutes for the full notebook |
| System | hybrid dense + BM25, `BAAI/bge-reranker-v2-m3` reranker, `Qwen/Qwen2.5-VL-3B-Instruct` fp16 |
| Validation | passes `validate_submission.py` (40 rows) |

The run used the `p1_rag` answering code committed in `921bfb7`. Changes made after the run do not affect these
40 answers: the notebook's retrieval demo queries, comment wording, the Cell 1 commit fallback, the wording
of an Okun's-law slope sentence (used only when a question asks for the slope or coefficient), and three
reworded golden-set questions (answers unchanged).
