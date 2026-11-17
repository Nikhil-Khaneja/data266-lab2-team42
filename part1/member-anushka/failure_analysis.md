# Part 1 - Failure analysis (Anushka)

What failed, why, what was changed, and what still fails.

**Sources.**
* **Golden-set rows:** printed by the graded Kaggle run (`notebooks/p1_rag.ipynb`, Tesla T4, seed 266);
  the per-question files are in `../eval/final_results/`.
* **Public leaderboard:** behaviour on the practice questions is described by type only, without their
  text or IDs.

## 1. Why the baseline fails (14 of 25 golden questions)

**What the baseline is.** Dense retrieval only (bge-small, top 5). The model decides the answer, the
status and the citations.

| failure | golden questions | mechanism |
|---|---|---|
| Wrong status in conversations | G15, G17, G18 | Returns `answered` where a user-supplied value or a refused permanent change requires `session_override`. Nothing outside the model tracks the user's value. |
| Lost conversation context | G16 | "What figure do the documents actually give for that year?" is not rewritten into a standalone question. Dense retrieval returns an unrelated page, and the answer gives 5.1% instead of 7.4%. |
| Missed conflict | G20 | Retrieves only DOC3 Table B-2 (medium trust) and reports a 0.5-point rise. The high-trust DOC1 value (9.6) and the conflict status are missing. |
| Accepted a false premise | G24 | Explains why unemployment "kept rising" in 2012. It fell from 8.9% to 8.1%. |
| Unanswerable answered | G23 | The model writes "not_found" inside the answer but the status stays `answered`, and it cites an unrelated table. |
| Wrong chart value | G06, G07 | Near-identical index charts (Figures A-4 and A-5, different base dates) are read wrongly: 1.0 and 2.00 instead of 1.22 and 1.32. |
| Forecast instead of outturn | G13, G22 | Reads Document 1's forecast column (9.0, 1.5) where Document 2 reports the outturn (7.7, 1.3). |
| Two-source arithmetic | G12, G21 | Picks a single source, or does the subtraction itself: 4.0 and 0.9 instead of 1.9 and 0.8. |
| Value missing | G25 | Answers in words ("slightly negative") without the number. |

**What fixed these in the final system.**
* A structured fact store with precedence rules.
* Deterministic status rules.
* Follow-up rewriting.
* Arithmetic done in code.
* Exact figure routing.

## 2. Failures found and fixed during development

### 2.1 A table value leaking into a figure question

* **Symptom.** A question naming the September-2011-vintage chart (forecasts for 2011–2012) got the
  2012 *outturn* from a supplement table, not the chart's own projected value.
* **Cause.**
  * The fact lookup ran on every question.
  * Its "verified data" line for the same entity, variable and year was put into the prompt next to the
    chart.
  * The 3B model preferred the text line over reading the image.
* **Fix.** When a question names a figure or table, looked-up facts, boosts and arithmetic are limited to
  that object.
* **Effect.** The figure question returns the chart's value, consistently across runs.

### 2.2 Model sentences that contradict the looked-up values

* **Symptom.** In answers that combine several looked-up values, the model's own sentence restated them
  with wrong years or invented numbers, e.g. "remained high at 9.3% in 2010" when 9.3% is the 2009 value.
* **Fix.** The values are stated from the fact store. A model sentence is kept only if it adds a computed
  result (a difference or forecast error that the code computed) and contains no unexplained number.
* **Regression it caused.**
  * The first version also dropped correct sentences whose numbers the lookup had not produced.
  * G02 (China's share of world output, 9.3%, and output per person, $4,300) fell to incorrect.
  * The rule was narrowed: a sentence is dropped only when it mixes a looked-up value with an unexplained
    number. G02 returned to correct.

### 2.3 Wrong comparison direction

* **Symptom.** "…was 7.3%, which is higher than … 7.7%." The two numbers were right but the
  comparative was wrong. This happened on a session-value question.
* **Fix.**
  * A guard parses "A% … higher/lower/larger/smaller/greater/less/faster/slower/above/below … B%" within
    one sentence and flips the word if it contradicts A vs B.
  * It skips differences ("2 percentage points higher") and negations.
  * Session-value comparisons also state the computed difference.
* **Test.** Run over every earlier output, the guard changed exactly one sentence.

### 2.4 "Always / never" yes-no questions

* **Symptom.** Yes/no questions asking whether a relation *always* holds were answered "Yes". The
  documents state a condition: unemployment falls only when growth exceeds about 3% (Okun's law).
* **Cause.** The 3B model answers the general tendency and ignores the qualifier.
* **Fix.**
  * Questions of the form "does/is/are … always/never/necessarily …" get a second, focused call that asks
    for the condition or exception stated in the evidence.
  * A hedged "Yes, but…" becomes "No, not always."
  * When the question relates output growth and unemployment, the answer uses Okun's law with the slope
    (−0.4) and threshold (about 3%) parsed from Document 1's text, cited as DOC1 p.29.
* **Status side effect.** One such question had been `not_found` without the reranker. The focused call
  now runs after a model-chosen `not_found` too, and accepts the result only if it does not say the
  evidence lacks the information.

### 2.5 Over-used `false_premise`

* **Symptom.** Plain yes/no questions sometimes came back `false_premise`.
* **Fix.** `false_premise` is kept only when the question presupposes something ("why / how come /
  given that / since …") and the resolved facts contradict it, for example a contraction in a year with
  positive growth. A model-chosen `false_premise` without a presupposition becomes `answered`.

### 2.6 Query expansion pulling the wrong pages

* **Symptom.** After everyday-word synonyms were added ("out of work" → unemployment rate), the public
  score dropped (rank 11 → 16). The expansion was one of the causes; the others were the over-used
  `false_premise` in 2.5 and a prompt rule about "always / never" that changed unrelated answers, since
  replaced by the focused call in 2.4.
* **Cause.** The retrieval query was expanded even when it already contained the documents' own term,
  which re-weighted BM25 towards the wrong pages.
* **Fix.** Expand only when no canonical word for the variable is present.

### 2.7 Sentence splitting at "U.S."

* **Symptom.** An answer ended with the fragment "The U.S."
* **Cause.** The splitter cut after "U.S." and the filter dropped the rest of the sentence.
* **Fix.** Splitting no longer happens after an initialism.

## 3. The reranker trade-off

* **Retrieval.** On the golden set the reranker improves retrieval: Recall@5 0.958 → 1.000, MRR@10
  0.931 → 0.951.
* **Public leaderboard.** An earlier version scored **0.881 with and without** the reranker. The two files
  differed on 16 of 40 rows:

  | effect of the reranker | type of question | what changed |
  |---|---|---|
  | Better | CPI vs GDP deflator | it ranked the passage on CPI–deflator divergence first, so the answer gave the 1979–1980 exception |
  | Better | China during the 2008–2009 crisis | the fiscal-expansion passage instead of a general growth passage |
  | Worse | a short definition question | a passage with a looser definition ("a decrease in real output") outranked the one with the formal two-consecutive-quarters definition |
  | Worse | some open questions | one supporting page dropped from the citations |

* **The two figure-value differences between the files** were not retrieval failures. They came from the
  leakage in 2.1: the reranker changed the context order, and the model copied a different line. After
  the fix in 2.1, both configurations read the chart.

## 4. What still fails in the final system

| issue | evidence | cause | possible fix (not done) |
|---|---|---|---|
| Citation accuracy 0.887 (55/62 cited sources are gold) | 4 of the 7 unsupported citations are in G02. G10, G15 and G24 add one non-gold but relevant source each (DOC1 p.29 for Okun's law; DOC1 Table 1-2, the forecast table whose value was replaced). | **G02:** the word "output" in "share of world output" made the lookup resolve *output growth*, so the answer and citations include Tables 1-1 and 1-4 next to the correct Figure 1-6 values. | Treat "share of …" and "output per person" as their own variables; only cite a fact table when its value appears in the final answer. |
| Short definition questions | With the reranker, a definition question cites a looser passage. | Cross-encoder relevance favours a short definitional sentence over the formal definition. | Boost passages containing defining patterns ("is defined as", "two consecutive quarters"). |
| Explanations vary between runs | On a "why" question the Kaggle fp16 run gave the documents' explanation (labour-market rigidities, generous unemployment insurance). The local 4-bit runs blamed the crisis itself. | The 3B model's prose depends on numerical precision and context order. Greedy decoding is deterministic only for a fixed backend. | A larger model (7B in 4-bit) for the explanation call only, or extractive explanations quoted from the top passage. |
| Latency | Median 5.09 s per question (final) vs 2.76 s (baseline). | The cross-encoder adds about 0.35 s. Explanation, rewrite and "always" calls add extra model calls. | Cache the reranker scores per query; skip the rewrite call on first turns. |

## 5. Summary

* **Biggest gains.** They came from moving decisions out of the 3B model and into code:
  * which value is current (precedence rules);
  * which status applies;
  * which object a value comes from;
  * every arithmetic result.
* **The model's job now.** Reading charts, and writing short explanations where the documents give one.
* **Remaining errors.** Almost all are in model-written text (explanations, definitions) and in variable
  resolution for unusual phrasings. The status, figure and citation rules are general and do not depend
  on any particular question.
