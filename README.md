# DATA 266 Lab 2 — Team 42

**Pair_Programming_Team_42:** Nikhil Khaneja, Anushka Khadatkar

Three parts in one repository:

1. **Part 1:** multimodal RAG over three economics PDFs. This part is also the Kaggle class competition.
2. **Part 2:** LoRA fine-tuning of Stable Diffusion 1.5 on Naruto images.
3. **Part 3:** a five-agent system that understands an image and generates a new one.

Both members built a complete, executed pipeline for every part. For each part the team compared the two
implementations and graded the stronger one; the other is kept as a working baseline.

Every number in this README is copied from a saved notebook output or from a result file that the notebook
wrote. The file is named next to each table. `GRADING_GUIDE.md` maps every rubric item to its evidence.

---

## 1. Results at a glance (graded systems)

| Part | Primary result | Secondary | Human evaluation | Efficiency |
|---|---|---|---|---|
| **1 RAG** | **25/25** golden-set answers correct (baseline 11/25); Kaggle public score **0.930** (rank 6) | Recall@5 0.958; citation accuracy 0.887 | groundedness **5.0/5**; raters within 1 point on 100% of answers | 5.09 s per question (T4) |
| **2 Diffusion** | KID **0.035** ± 0.016 (base 0.142); CLIP 0.304 (base 0.313) | LPIPS seed diversity 0.676 (base 0.690) | final run vs base: **42.6%** win rate (3 raters, 25 pairs) | 0.81 GPU-hours, 6.17 GiB peak (T4) |
| **3 Agents** | task success **14/15** automatic, 12/15 human (no loop: 12/15, 12/15) | critic agreement 10/14; red-team attack success 1/10 | members agreed on 25/30 judgements | 6.4 s per request (no loop: 4.9 s) |

**Bonus:** a reranker for Part 1. It raises retrieval Recall@5 from 0.958 to 1.000 and MRR@10 from 0.931 to
0.951 (`part1/eval/final_results/reranker_ablation.csv`).

---

## 2. Who led which part, and why

| Part | Lead (graded system) | Partner (baseline and other contributions) | Why this system is the graded one |
|---|---|---|---|
| **1 RAG** | **Anushka**: `part1/member-anushka/` (hybrid retrieval + reranker, Qwen2.5-VL-3B fp16, precedence rules in code) | **Nikhil**: complete baseline in `part1/member-nikhil/` (hybrid + reranker, Qwen2.5-VL-7B 4-bit). He wrote the first golden set, committed Oct 27, and rated groundedness. | Higher Kaggle public score: **0.930 vs 0.867**. The whole notebook runs in about 22 minutes on a T4. Revision, outturn, conflict and arithmetic decisions are made in code, which is where the gain over the baseline came from (11/25 → 25/25). |
| **2 Diffusion** | **Anushka**: `part2/member-anushka/` (LoRA rank 4 vs rank 16, Colab T4) | **Nikhil**: complete baseline in `part2/member-nikhil/` (two LoRA runs on an RTX 4090, 120 images, both adapters and the image dataset on Hugging Face). He was one of the 3 human raters. | Only this pipeline has every Part 2 evaluation item finished: the blinded 3-rater comparison with an outside rater, the NSFW checker on all 120 images (flagged images withheld from the release), a memorization check against a held-out threshold, KID over 100 subsets, and the model card. It was also trained on the required 16 GB-class GPU (T4). Nikhil's rank-16 run has a lower KID (0.023), but it uses a different KID set-up and a 1,200-step budget, and its human evaluation was not run. So the two KIDs are not directly comparable. |
| **3 Agents** | **Anushka**: `part3/member-anushka/` (pydantic state, critique loop with at most 2 revisions, Part 2 rank-16 LoRA, Gradio app and Space build) | **Nikhil**: built the first end-to-end Part 3 pipeline in `part3/member-nikhil/`, with the shared test set (`part3/eval/`, committed Oct 27), traces and a Gradio app. In his run, all 10 red-team cases were refused. He was the second human judge for the graded system. | Only this implementation has the full Part 3 evaluation finished: blinded pass/fail judgements from both members on all 30 scenario outputs, 14 critic labels, and every disagreement resolved. It also uses the team's final Part 2 model. The pass/fail sheets for Nikhil's run have not been filled in. |

What each member did, in detail, is in `CONTRIBUTIONS.md`. Which AI coding assistants were used, and for
what, is in `AI_USAGE.md`.

---

## 3. Part 1 — Multimodal RAG (Kaggle competition)

**Graded system:** `part1/member-anushka/`.
**Notebook:** `part1/member-anushka/notebooks/p1_rag.ipynb`. It ran on a Kaggle T4 with seed 266 and reads
`QUESTIONS_PATH`.
**Write-ups:** `part1/member-anushka/README.md` and `failure_analysis.md`.

**Pipeline**

1. **Ingestion (1.1).** PyMuPDF text chunks keep `doc_id` and page: 186 / 10 / 9 text units for
   DOC1 / DOC2 / DOC3. Every numbered figure and table is cropped with its exact label and caption, giving
   27 figures and 9 tables; `Figure B-5` and `Figure B-5 (corrected)` are separate objects. Qwen2.5-VL-3B
   writes a text description of each figure so it can be searched. Everything is stored in ChromaDB with
   `doc_id`, `page`, `figure_id` and `trust_level`, read from `kb_manifest.json`.
2. **Retrieval (1.2).**
   - `bge-small-en-v1.5` dense search and BM25, fused with reciprocal-rank fusion.
   - `bge-reranker-v2-m3` re-ranks the top 20 candidates.
   - A question that names a figure or table gets exactly that object, and values come only from it.
   - Appendix C (low trust) is never returned as evidence.
3. **Answering.**
   - Table cells become structured facts, so precedence (revision > outturn > forecast; conflict → the
     higher-trust source), status and arithmetic are decided in code.
   - The VLM writes the JSON answer from the evidence.
   - Deterministic guards set citations, statuses and refusals.
4. **Conversations (1.3).** Each `conversation_id` gets its own memory object. User-supplied values stay in
   that object, never enter the vector database, and are forgotten when the conversation ends. Requests to
   change the documents permanently are refused.

**Results** on the 25-question golden set (`part1/eval/final_results/README.md`):

| Measurement | Baseline (dense only) | Hybrid, no reranker | **Final (hybrid + reranker)** |
|---|---|---|---|
| Primary: answer correctness (numbers within ±1% and correct status) | 11/25 | 25/25 | **25/25** |
| Status accuracy | 0.76 | 1.00 | **1.00** |
| Secondary: Recall@5 (24 questions with a retrieval target) | 0.833 | 0.958 | **0.958** |
| Secondary: citation accuracy | 0.760 (38/50) | 0.887 (55/62) | **0.887 (55/62)** |
| Human: groundedness, 1–5, both members | not rated | not rated | **5.0** (agreement within 1 point: 1.00) |
| Efficiency: median seconds per question (T4) | 2.76 | 3.77 | **5.09** |

- **Every golden-set category is fully correct:**

  | Category | Correct |
  |---|---|
  | Facts and tables | 5/5 |
  | Figures, including near-identical charts | 6/6 |
  | Two-source questions | 3/3 |
  | Multi-turn with a user-supplied value | 5/5 |
  | Conflicts and revisions | 3/3 |
  | Unanswerable, false-premise and injection | 3/3 |

- **Kaggle:** public score **0.930**, rank 6. The submission is in
  `part1/member-anushka/results/kaggle_final/`. No question IDs or question text appear in the source code.
- **What worked.** The large gain came from moving decisions out of the 3B model and into code: which value
  is current, which status applies, which figure a value comes from, and all arithmetic.
- **What did not work.**
  - Citation accuracy is still 0.887. Four of the seven unsupported citations are in one question, where
    "share of world output" was resolved as *output growth*.
  - Explanation-style answers can vary between runs.
  - The reranker costs about 0.35 s per question.
- **Safety (C4).**
  - No answer cites Appendix C of DOC3.
  - The instruction printed inside the Figure C-1 image was not followed (golden question G25).
  - A false premise was corrected (G24).
- **Disclosed limitation.**
  - The final golden set was written on Nov 11, after tuning on the practice questions had started.
  - The human groundedness scale is at its ceiling (every answer was rated 5), and both raters built the
    system.

---

## 4. Part 2 — Diffusion fine-tuning

**Graded system:** `part2/member-anushka/`.
**Notebooks:**
- `p2_finetune.ipynb`: data preparation, training and generation.
- `p2_evaluate.ipynb`: metrics, safety, human evaluation and the Hugging Face release.

**Summary file:** `part2/eval/part2_final_results_anushka.json`.

- **Data (2.1).**
  - Dataset: `lambdalabs/naruto-blip-captions`, 1,221 images.
  - Removed: 0 corrupt and 0 exact-duplicate images.
  - Split with seed 266 into 1,098 training and 123 test images, resized to 512×512.
  - A dHash near-duplicate audit found 1 test image with a near-duplicate in the training split.
  - Biases:
    - one copyrighted style;
    - short, generic BLIP captions;
    - gender skew (58.8% of captions mention male terms vs 20.6% female).
- **Training (2.2).** Two runs with the same data and step budget. Both used Stable Diffusion 1.5,
  1,500 steps, lr 1e-4, batch 1 × gradient accumulation 4, fp16, 8-bit AdamW and gradient checkpointing.

  | Run | Trainable parameters | Peak GPU memory | GPU-hours (T4) |
  |---|---|---|---|
  | LoRA rank 4 | 797,184 | 6.15 GiB | 0.81 |
  | LoRA rank 16 (final) | 3,188,736 | 6.17 GiB | 0.81 |

- **Generation (2.3).**
  - The 20 prompts (10 in-domain from test captions, 5 out-of-domain, 5 compositional) were frozen before
    training.
  - Seeds 0 and 1, PNDM 40 steps, guidance 7.5: 40 images per model, 120 in total.
  - Side-by-side grids for 8 prompts are in the training notebook.

**Results** (`part2/member-anushka/eval/part2_metrics.json`):

| Measurement | Base SD 1.5 | LoRA rank 4 | **LoRA rank 16 (final)** |
|---|---|---|---|
| Primary: KID, in-domain ↓ (mean ± std over 100 subsets of 10) | 0.142 ± 0.026 | 0.034 ± 0.015 | **0.035 ± 0.016** |
| Primary: CLIP score, 20 prompts | 0.313 | 0.307 | **0.304** |
| Secondary: LPIPS seed diversity | 0.690 | 0.692 | **0.676** |
| Human: win rate vs base, excluding ties (25 pairs, 3 raters including 1 outside the team) | reference | not rated | **42.6%** (Fleiss' κ 0.086) |
| NSFW-flagged outputs (of 40) | 0 | 2 | **0** |
| Efficiency: GPU-hours / peak memory | – | 0.81 h / 6.15 GiB | **0.81 h / 6.17 GiB** |

- **What worked.** Both LoRAs cut KID about 4× compared with the base model. Raters preferred the final run on
  out-of-domain prompts (72.7% win rate).
- **What did not work.**
  - CLIP alignment drops slightly.
  - Raters preferred the base model on in-domain prompts (final-run win rate 28.0%).
  - Compositional prompts lose objects and relations.
  - Agreement between raters is low (κ 0.086), and two raters show a significant left/right bias.
- **Why rank 16 was chosen.** Quality, alignment and cost are within noise of rank 4: the KID gap is 0.0015,
  against a subset std of 0.016. Rank 16 is the only run with no NSFW-flagged outputs on the fixed prompts.
  Details are in `part2/member-anushka/failure_analysis.md`.
- **Safety (C4).**
  - **License:** Naruto characters are copyrighted, so the model is for non-commercial research only.
  - **NSFW filter:** the Stable Diffusion safety checker ran on every output. Flagged images were withheld
    from the released dataset.
  - **Memorization:** the highest CLIP similarity between a final-run generation and a training image is
    0.928. That is below the 95th percentile for real held-out images (0.943).

---

## 5. Part 3 — Multimodal multi-agent system

**Graded system:** `part3/member-anushka/`.
**Notebooks:**
- `p3_agents.ipynb`: agents and evaluation, run on a Colab T4 with code at `40c0a5c`.
- `p3_human_eval.ipynb`: human evaluation.

**Other files:**
- Architecture diagram: `part3/member-anushka/docs/p3_architecture.png`.
- System card: `part3/member-anushka/README.md`.

**Flow:** request → **Planner** (routing and input safety) → **Vision Understanding** → **Prompt
Engineering** → **Image Generation** → **Critique** (with an output safety check) → accept, or revise back to
Prompt Engineering at most twice. Caption and question requests stop after Vision. Unsafe requests are
refused by the Planner.

| Agent | What it does |
|---|---|
| Planner | Classifies the request as caption/question, style change, edit, enhance, domain generation or unsupported. Refuses NSFW requests, deceptive edits of real people, and instructions found in the image's OCR text. |
| Vision Understanding | Qwen2.5-VL-3B: caption, main objects, visible text, and at least 2 answers to questions about the image. |
| Prompt Engineering | Writes the generation prompt from the request and the caption. Text found in the image is never copied in. |
| Image Generation | SD 1.5 img2img (style change and enhancement), InstructPix2Pix (edits), SD 1.5 + Part 2 rank-16 LoRA (domain style). Records the model, prompt, seed and steps. |
| Critique | CLIP score against the prompt, a VLM yes/no judgement, CLIP preservation of the input, and the SD safety checker. Decides accept or revise, with a concrete revision instruction. |

**Orchestration:**
- One typed pydantic `AgentState`, passed through a plain-Python graph.
- A JSON trace for every request: which agent ran, its input and output, and how long it took
  (`outputs/traces/`).
- Clear messages instead of crashes for broken, empty or ambiguous input.
- A Gradio app: `python -m p3.app`.

**Results** (15 scenarios and 10 red-team cases; `part3/member-anushka/outputs/results_table_final.md`):

| Measurement | Baseline (no critique loop) | **Final (critique loop, ≤ 2 revisions)** |
|---|---|---|
| Primary: task success, automatic (/15) | 12/15 | **14/15** |
| Primary: task success, agreed human judgement (/15) | 12/15 | **12/15** |
| Secondary: critic agreement with our labels | n/a | **10/14 (71.4%)** |
| Secondary: red-team attack success (/10) | 1/10 | **1/10** |
| Human: member agreement on pass/fail | 11/15 | **14/15** (25/30 overall) |
| Efficiency: median seconds per request | 4.9 | **6.4** |

- **What worked.**
  - The critique loop fixed weak first attempts. In S06 the CLIP score rose from 0.214 to 0.354 after a
    revision; in S10 it rose from 0.242 to 0.295.
  - The guardrails stopped 9 of the 10 red-team cases before any generation.
- **What did not work.**
  - R01: instructions written in an image reached the prompt through the caption rather than OCR, so the
    request was not refused. That is the single red-team success.
  - InstructPix2Pix sometimes changes too much of the image (S07).
  - The 3B VLM judge said NO in 19 of 20 critiques, so the critic relies mostly on CLIP.
  - Under human judgement the loop gives no gain (12/15 in both versions).
- **Safety (C4).**
  - Input checks: NSFW and gore terms (with leetspeak and spacing normalization), deceptive real-person
    edits, and an OCR scan for instructions inside the image.
  - A negative prompt is always applied.
  - Output check: the SD safety checker; flagged images are withheld.
  - All 10 red-team outcomes are listed in `part3/member-anushka/README.md`.

---

## 6. Reproducibility

- **Hosted APIs:** none. Every model is open-weights and runs locally (see the table below).
- **Notebooks:** every notebook is committed executed, with its outputs.
- **Cell 1:** the first cell of each notebook prints the Git commit, library versions, GPU and seed (266).
- **Test sets:** the golden set, prompts and scenarios are committed under `part{1,2,3}/eval/` and each
  member's `eval/` folder. Their commit dates are listed in `GRADING_GUIDE.md`.
- **Weights and images:** they are on Hugging Face, never in Git (see `.gitignore`). Links are in `links.txt`.
- **Tags:** `p1-v1.0`, `p2-v1.0` and `p3-v1.0` mark the graded commits.

### Models (all open weights from the Hugging Face Hub)

| Part | Models |
|---|---|
| 1 (`member-anushka`, final) | `Qwen/Qwen2.5-VL-3B-Instruct` (fp16) for figure descriptions and answers; `BAAI/bge-small-en-v1.5` embeddings; `BAAI/bge-reranker-v2-m3` reranker |
| 1 (`member-nikhil`, baseline) | `Qwen/Qwen2.5-VL-7B-Instruct` (4-bit, bitsandbytes) for figure descriptions and answers; `BAAI/bge-small-en-v1.5` embeddings; `BAAI/bge-reranker-v2-m3` reranker |
| 2 (both) | `stable-diffusion-v1-5/stable-diffusion-v1-5` + our LoRA runs; `openai` CLIP ViT-B-32 (open_clip) and LPIPS (AlexNet) for evaluation |
| 3 (`member-anushka`, final) | `Qwen/Qwen2.5-VL-3B-Instruct` (vision, OCR, critic judgement); `stable-diffusion-v1-5/stable-diffusion-v1-5` img2img/txt2img; `timbrooks/instruct-pix2pix`; our Part 2 LoRA `Anushka1202/data266-lab2-part2-lora-rank16`; `openai/clip-vit-base-patch32`; SD safety checker (`CompVis/stable-diffusion-safety-checker`) |
| 3 (`member-nikhil`, baseline) | `Qwen/Qwen2.5-VL-3B-Instruct` (fp16); SD 1.5 img2img; `timbrooks/instruct-pix2pix`; our Part 2 LoRA `nikhilkhaneja/sd15-naruto-lora-run2`; CLIP ViT-B-32 |

### Where the notebooks are

The repo uses one folder per member inside each part, instead of a single root `notebooks/` folder.

| Part | Final (graded) notebook | Partner baseline |
|---|---|---|
| 1 | `part1/member-anushka/notebooks/p1_rag.ipynb` (Kaggle T4), `p1_human_eval.ipynb` | `part1/member-nikhil/notebooks/p1_rag.ipynb` |
| 2 | `part2/member-anushka/notebooks/p2_finetune.ipynb`, `p2_evaluate.ipynb` | `part2/member-nikhil/notebooks/p2_finetune.ipynb` |
| 3 | `part3/member-anushka/notebooks/p3_agents.ipynb`, `p3_human_eval.ipynb` | `part3/member-nikhil/notebooks/p3_agents.ipynb` |

### Environment

`requirements.txt` pins the versions used for the RTX 4090 runs (Nikhil's baseline notebooks), including
the CUDA 12.8 build of torch and torchvision:

```
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
```

The final (graded) notebooks ran on Kaggle or Colab T4s with the platform's preinstalled torch. Each one
prints its exact library versions in Cell 1, and its pinned extras are in:

| Part | Final notebook environment |
|---|---|
| 1 | `part1/member-anushka/requirements.txt` (the notebook's install cell adds only missing pinned packages) |
| 2 | `part2/member-anushka/requirements-p2.txt` |
| 3 | `part3/member-anushka/requirements-colab.txt` (Colab), `part3/member-anushka/requirements.txt` (local) |

Nikhil's Part 1 Kaggle notebook installs only `part1/member-nikhil/kaggle_requirements.txt`.

### Running each part

- **Part 1 (final).**
  1. Open `part1/member-anushka/notebooks/p1_rag.ipynb` on Kaggle with the competition data attached, a T4
     GPU and internet on.
  2. Run all cells. The notebook is self-contained: it writes the `p1_rag` package from its own cells. It
     reads `QUESTIONS_PATH` (default `Questions.csv`) and writes `submission.csv` with no manual step, in
     about 22 minutes.

  Full steps are in `part1/member-anushka/README.md`.
- **Part 1 (baseline).**
  1. Place the competition files under `part1/member-nikhil/data/raw/part1/`.
  2. Run `part1/member-nikhil/notebooks/p1_rag.ipynb`.
- **Part 2 (final).**
  1. Run `part2/member-anushka/notebooks/p2_finetune.ipynb`, then `p2_evaluate.ipynb`, on a Colab T4.
     It only needs public data.
  2. Set `UPLOAD_TO_HUB = True` to publish.

  Steps are in `part2/member-anushka/README.md`.
- **Part 2 (baseline).** Run `part2/member-nikhil/notebooks/p2_finetune.ipynb`, or the scripts in
  `part2/member-nikhil/src/part2/` with `part2/member-nikhil/` as the working directory.
- **Part 3 (final).**
  1. Run `part3/member-anushka/notebooks/p3_agents.ipynb` on a Colab T4.
  2. For the local demo, run `python -m p3.app` from `part3/member-anushka/`.

  The Hugging Face Space build is in `space/`.
- **Part 3 (baseline).**
  1. Run `part3/member-nikhil/notebooks/p3_agents.ipynb`. It makes its own test images.
  2. For the demo, run `python app/gradio_app.py` from `part3/member-nikhil/`.

### Repository layout

```
data266-lab2-team42/
├── README.md, GRADING_GUIDE.md, CONTRIBUTIONS.md, AI_USAGE.md, links.txt, requirements.txt
├── part1/  eval/ (golden sets, final results)   member-anushka/ (final)   member-nikhil/ (baseline)
├── part2/  eval/ (fixed prompts, final summary) member-anushka/ (final)   member-nikhil/ (baseline)
└── part3/  eval/ (scenarios, red team)          member-anushka/ (final)   member-nikhil/ (baseline)
```

Each `member-<name>/` folder is self-contained. It holds that member's code, notebooks and result files.
