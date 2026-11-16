# AI usage - Part 1 (Anushka)

Rule S.6.3: which AI coding assistant was used, for what, and how its output was checked.

| tool | model | used for |
|---|---|---|
| Claude Code (desktop app) | Claude Opus 5.5 | implementing the `p1_rag` package and the notebook builder, local evaluation runs, analysing Kaggle outputs, a first draft of the golden-set questions, documentation drafts |

**No AI service is called by the submitted code.** Every model the notebook runs is an open-weight model
from the Hugging Face Hub: Qwen2.5-VL-3B-Instruct, bge-small-en-v1.5 and bge-reranker-v2-m3.

**My part, throughout:**
* **Design and direction:** the system design choices listed below and the direction of every iteration.
* **Kaggle:** every run and every submission, and the choice of the final one.
* **Golden set:** checking every answer against the PDFs.
* **Human evaluation:** the human ratings.
* **Review:** reviewing all code and outputs.

## Contribution by stage

| stage | my work | where AI assistance was used | how it was checked |
|---|---|---|---|
| 1.1 Ingestion | choice of PyMuPDF, ~140-word chunks, VLM text descriptions (rather than CLIP image embeddings) as the figure-search method, ChromaDB metadata | code for caption detection by bold font, figure/table cropping, table-grid parsing into facts, the VLM description prompt and the second "printed data labels" pass | figure crops inspected visually; units and facts checked against the PDFs (`Figure B-5` and `Figure B-5 (corrected)` present as two objects; Appendix C units marked `low`); trust levels read from `kb_manifest.json`, not typed in |
| 1.2 Retrieval | hybrid dense + BM25 with RRF; exact Figure/Table routing; keeping low-trust units in the index but out of results | `KBIndex` (dense, BM25, RRF k = 60, label routing, low-trust filter) | retrieval traces of every golden question inspected; Recall@5 and MRR computed by the notebook |
| 1.2 Answers | precedence rules (revision > outturn > forecast; conflict → higher trust); deterministic status and citation rules | `FactStore`, prompts, JSON parsing, the guards in `answer.py` | golden set (25 questions, automatic scoring) after every change; the full 40-question practice file run locally after every change; Kaggle public score after each submission |
| 1.3 Conversations | one memory per `conversation_id`; user values never in the vector DB; refusing permanent changes | `Conversation` class, follow-up rewriting, override extraction | golden conversations GA and GB (5 turns) and the multi-turn practice conversations, read turn by turn |
| 1.4 Evaluation | metric definitions from the lab; checking the 25 golden answers against the PDFs; human ratings | the scoring code (`evaluate.py`), the first draft of the golden-set questions, the human-evaluation notebook | metric values recomputed by hand from the per-question CSVs (e.g. citation accuracy 55/62 = 0.887) |
| Bonus reranker | adding `bge-reranker-v2-m3` over the top 20 hybrid candidates, keeping named objects and fact tables pinned first | `enable_reranker`, `rerank`, the retrieval ablation (Recall@1/3/5, MRR@10) | ablation printed by the notebook; both reranker and no-reranker submissions scored on Kaggle (0.881 each), and the reranker kept |
| Kaggle | every run, every submission, choosing the final 0.930 submission and its notebook | analysing the downloaded outputs row by row; comparing the reranker and no-reranker files | validator passes; the code inside the Kaggle notebook was compared byte for byte with the repository code before results were reported |

## How the system was improved, iteration by iteration

**How each round worked.**
* **The loop:** I ran a version on Kaggle and read the score. Claude Code compared the downloaded answers
  and traces with the previous run.
* **Where a fix could go:** each failure was traced to its stage (retrieval, fact lookup, prompt, guard)
  and fixed there with a general rule.
* **What was ruled out:** a rule for any specific question. The lab forbids it.

Main changes (all general):

| problem found in the outputs | change |
|---|---|
| Everyday wording ("out of work", "jobless") missed the documents' terms | Variable synonyms in `vocab.py`. The retrieval query is expanded with the documents' term **only** when the question does not already contain it; the first version always expanded and pulled the wrong pages. |
| Yes/no questions got `false_premise` | `false_premise` only when the question presupposes something ("why / how come / given that…") that the resolved facts contradict. |
| A figure question answered with a table's outturn instead of the chart's own value | Looked-up facts and arithmetic are restricted to the object the question names. |
| The model wrote "A% is higher than B%" with A < B | Code checks the comparison word against the two numbers in the sentence. Tested on every earlier output: it changed exactly one sentence. |
| Model sentences restated looked-up values with wrong years or extra numbers | Values come from the fact store. A model sentence is kept only if it adds a computed number and no unexplained one. |
| "Always / never" yes-no questions answered "Yes" | A second, focused call asks the model for the condition or exception in the evidence. Growth-vs-unemployment questions use Okun's law with the slope and threshold read from Document 1's text. |
| The sentence splitter cut "The U.S." off a sentence | Abbreviation-aware sentence splitting. |
| The cross-encoder changed which text passages were cited on open questions | Named objects and fact tables stay pinned first; the reranker orders only the remaining candidates. |

## Bugs and regressions caught before submission

* **Over-strict number filter.** The first version of the number filter dropped a correct sentence (a
  world-output share and output per person). The fact lookup had resolved the wrong variable, so the
  correct numbers looked "unexplained". The rule was narrowed to drop only sentences that mix a looked-up
  value with an unexplained number. The golden set returned to 25/25.
* **Comparison guard false positives.** It was tested on differences ("2 percentage points higher than…")
  and negations ("not higher than"); both are left unchanged.
* **Same-year "change" sentence.** A "From 2013 to 2013 it rose by …" sentence appeared when two lookups
  had the same entity and year but different variables. The change sentence now requires the same
  variable.
* **Misleading ablation.** The retrieval ablation first made the reranker look worse, because it ran
  without the fact-table pins. It now uses the same standalone query and pins as the answering step.

## Checks for hard-coding (S.3)

* **Automatic scan.** Before each commit, a script searched every committed file (code, notebook source,
  README) for the practice question IDs and for any 5–7-word sequence from the practice questions.
* **What was replaced.** Two demo queries in the notebook builder and one phrase in an answer template
  were too close to practice wording and were replaced with golden-set questions and neutral wording.
* **What is left.** There are no question IDs, no answer values and no rule that refers to a specific
  question.

## Verification of the reported numbers

* **Source of every number.** Every result in the README and the report is printed by the executed
  Kaggle notebook.
* **Recomputed by hand from the per-question CSVs in `../eval/final_results/`:**
  * correct counts;
  * status accuracy (19/25 baseline, 25/25 final);
  * Recall@5 (20/24, 23/24);
  * citation accuracy (38/50, 55/62);
  * medians.
* **Local runs.** The MLX 4-bit model on Apple Silicon was used only for development. No local number is
  reported.

## Not done with AI

* Running Kaggle notebooks, submitting, and choosing the final submission and notebook.
* Checking each golden-set answer, status and source against the three PDFs.
* The human groundedness ratings (both members, independently).
* Keeping the reranker in the final system after the tie on the public leaderboard.
* Disclosing the golden-set timing.
* The IEEE report text.

I can explain every module and every guard in the demo (S.2).
