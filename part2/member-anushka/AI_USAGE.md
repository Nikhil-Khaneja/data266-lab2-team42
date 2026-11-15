# AI usage - Part 2 (Anushka)

AI coding assistants were used as **pair-programming support**:

* explaining and fixing environment errors in Colab;
* reviewing the pipeline against the lab requirements;
* implementing parts of the revised evaluation notebook;
* drafting documentation.

The experiment design, data handling, training runs, prompt set, human evaluation, the final-run decision
and every reported result are my own work.

No AI service is called by the submitted code: every model the notebooks run is an open-weight model
(SD 1.5, its safety checker, OpenCLIP, LPIPS-AlexNet, Inception-v3 for KID).

## Contribution by stage

| stage | my work | where AI assistance was used | how I checked it |
|---|---|---|---|
| Environment (Colab T4) | Drive layout, pinned installs, Diffusers source install at a fixed commit, `bitsandbytes` for 8-bit Adam, Hugging Face login, global seeding (266) | diagnosing four errors (below) | each fix re-run in Colab until the training script started and the import checks passed |
| 2.1 Data preparation | loading `naruto-blip-captions`, decode check, exact-duplicate removal, seed-266 90/10 split, 512×512 resizing, dataset description and limitations | a later audit cell (strict re-decode, dHash near-duplicate search, caption statistics) | audit numbers checked against my original cleaning report (1,221 → 1,098 / 123, 0 removed) |
| 2.2 Fine-tuning | choosing SD 1.5 + LoRA, the rank-4 vs rank-16 design, hyper-parameters (LR 1e-4 constant, 1,500 steps, batch 1 × accumulation 4, fp16, gradient checkpointing, center crop + flip), running both trainings (0.81 GPU-h each), peak-memory measurement with an `nvidia-smi` poller, pushing both adapters | none for the design; error diagnosis only | training logs and adapter files on Drive; parameter counts 797,184 / 3,188,736 match LoRA rank × the attention dimensions |
| 2.3 Generation | writing the 20 prompts (10 held-out test captions, 5 out-of-domain, 5 compositional) and freezing them before training; generating 3 × 40 images with fixed sampler settings; generation metadata; first Hugging Face dataset upload | restructuring the executed notebook so the set-up cells come first; the comparison-grid cell | image counts checked in the notebook's counts table (40 per model; 20 in-domain / 10 out-of-domain / 10 compositional) |
| 2.4 Evaluation | first evaluation pass (CLIP, LPIPS, KID, NSFW check, human-evaluation sheet); organising the three raters; rating the pairs; collecting all 75 ratings; running the final evaluation on Colab; choosing and justifying the final run; the written analysis | revised evaluation cells (details below) after a requirements review found gaps in my first pass | below |
| Release | Hugging Face account, write token as a Colab secret, publishing adapters, model card and the filtered dataset, removing flagged images from the first upload | model-card and dataset-release cells | the notebook confirms the uploads and the two deletions; release counts 40 / 38 / 40 match the NSFW report |

## Debugging the training environment

| symptom | diagnosis | fix I applied |
|---|---|---|
| Diffusers import errors inside the LoRA script | the pip release did not match the example script, and the cloned source folder shadowed the installed package on `sys.path` | install Diffusers from source at commit `6dcb307d`; remove the shadowing paths from `sys.path` and insert the source path explicitly |
| PEFT failed to import | the preinstalled `torchao` was incompatible with the PEFT version | `pip uninstall -y torchao`, plus an import check that prints whether it is still importable |
| training script rejected `--report_to=none` | value not supported by that script version | flag removed |
| KID crashed in the first evaluation | metric state and image tensors on different devices | metric and inputs moved to the same device |

## Revised evaluation notebook (`p2_evaluate.ipynb`)

A review of my first evaluation pass against tasks 2.1–2.4 and C1–C4 found these problems:

* KID came from a single subset, so its std was 0;
* the memorization check compared *real test images* with the training set instead of *generations*;
* human-evaluation pairs 21–25 repeated pairs 1–5;
* the grids were not saved;
* there was no single results table;
* the notebook could not run top to bottom.

I chose which of these to fix. The revised cells were implemented with AI assistance, and I reviewed and
ran them. The cells (source in `scripts/p2_cells.py`) and their key technical choices:

**KID**

* Real Inception features of the 123 test images are extracted once (`reset_real_features=False`).
* Fake features are reset per model.
* 100 subsets of size 10, with seeded subset sampling, so the ± is a real spread.

**Memorization**

* OpenCLIP ViT-B/32 embeddings of all 1,098 training images.
* The top-1 neighbour of each of the 10 in-domain generations of the final run, with LPIPS per pair.
* Calibration: the same nearest-neighbour statistic for the 123 real held-out frames, with the median
  and 95th percentile as reference.
* The base model as a control.

**Human evaluation**

* Duplicates are detected by the key `(prompt_id, seed)`, and the 20 unique pairs with my team's existing
  ratings are kept.
* 5 new pairs are drawn from unused prompt/seed combinations with a seeded RNG, excluding NSFW-flagged
  images, with sides randomized.
* Blinded PDF rating sheets.
* Statistics:
  * win rate with ties excluded and with ties counted as ½, per rater, pooled and per category;
  * majority vote and pairwise agreement;
  * Fleiss' κ;
  * a two-sided binomial test of each rater's left/right choices.

**NSFW**

* The SD safety checker with the pipeline's own `CLIPImageProcessor`, run on all 120 saved images.
* Cross-checked against my first safety pass: 0 disagreements.

**Failure modes**

* Examples are chosen by fixed rules on the metrics, not by eye:
  * the largest out-of-domain CLIP drop vs base;
  * the lowest compositional CLIP score;
  * every NSFW flag;
  * the lowest LPIPS between seeds.
* A decision cell prints the facts the final-run choice rests on.

**Release**

* The model card is generated from computed values.
* The dataset is a Hugging Face `imagefolder` with a root `metadata.csv` keyed by `file_name`.
* Flagged files are deleted from the unfiltered repo, skipping files that are already gone.

## How AI-assisted code was verified

* **Regression against my own results.** A full CPU dry run of the revised notebook against a copy of
  my project folder reproduced the CLIP and LPIPS values of my original notebook exactly.
* **Bugs found and fixed before submission:**
  * an operator-precedence error in a set assertion;
  * a peak-memory call that could not see the training subprocess;
  * a KID cell that re-extracted the real features three times;
  * a crash in the NSFW cell caused by memory-mapped weights on a nearly full local disk (a local-only
    issue).
* **Final runs.** The final notebook ran on a Colab T4 with *Restart and run all*. It has no errors,
  execution counts in order, and all outputs saved. Its printed output confirms that all 75 ratings were
  loaded and the Hub uploads succeeded.
* **Figures inspected visually.** The grids, the three memorization pairs and the failure-mode figure. The
  image descriptions in `failure_analysis.md` come from that inspection.
* **Statistics can be recomputed.** Win rates, κ and the position-bias test can be recomputed by hand
  from `eval/part2_human_eval/blind_pairs.json` and `answer_key.json`.

## Not done with AI

* Choice of dataset, base model, method (LoRA) and the rank-4 vs rank-16 comparison.
* Training hyper-parameters and step budget, and running both training runs.
* The 20 evaluation prompts, frozen before training.
* All human judgements, by Anushka, Nikhil and Ayush (outside rater), on blinded sheets.
* The final-run decision (rank 16) and its justification.
* The analysis and wording of the IEEE report.

I can explain every cell of both notebooks in the demo (S.2).
