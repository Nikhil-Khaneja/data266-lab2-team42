# Part 2 - Failure analysis and final-run choice (Anushka)

Task 2.4.1: at least two failure modes with example images, and which run we would choose and why.

**Sources.** Every number below is printed by `notebooks/p2_evaluate.ipynb` (Colab T4 run, seed 266). The
cell that prints it is named in brackets. The example images are in the failure-mode figure, which the
failure-mode cell of that notebook draws and embeds (rows in the same order as the sections below).

**Set-up.**

* **Models.** SD 1.5 base, LoRA rank 4 (797,184 trainable parameters) and LoRA rank 16 (3,188,736).
  Both runs use the same 1,098 training images, 1,500 steps, effective batch 4 (1 × 4 accumulation),
  learning rate 1e-4 constant, fp16 and 8-bit AdamW.
* **Generation.** 20 frozen prompts (10 in-domain, 5 out-of-domain, 5 compositional) × seeds {0, 1}.
  PNDM scheduler, 40 steps, guidance 7.5.
* **Selection.** The failure-mode cell picks the examples by fixed rules on the metrics, not by eye:
  * the out-of-domain prompt with the most negative CLIP(final) − CLIP(base);
  * the compositional prompt with the lowest CLIP score for the final run;
  * every NSFW-flagged image;
  * the prompt with the smallest LPIPS distance between the final run's two seeds.

## Summary of the measurements behind the analysis

[results-table cell; `eval/part2_metrics.json`]

| model | KID in-domain ↓ (mean ± std, 100 subsets of 10) | CLIP, 20 prompts | CLIP, in-domain | LPIPS seed diversity | human win rate vs base (excl. ties) | NSFW flagged / 40 | GPU-hours | peak GiB |
|---|---|---|---|---|---|---|---|---|
| base | 0.1420 ± 0.0260 | 0.3126 | 0.2953 | 0.6902 | reference | 0 | - | - |
| LoRA r4 | 0.0336 ± 0.0155 | 0.3067 | 0.2885 | 0.6924 | not rated | 2 | 0.812 | 6.15 |
| LoRA r16 (final) | 0.0351 ± 0.0162 | 0.3037 | 0.2865 | 0.6761 | 42.6% | 0 | 0.808 | 6.17 |

**CLIP score by prompt category** [CLIP/LPIPS cell]

| model | in-domain | out-of-domain | compositional |
|---|---|---|---|
| base | 0.2953 | 0.3170 | 0.3427 |
| r4 | 0.2885 | 0.3219 | 0.3277 |
| r16 | 0.2865 | 0.3196 | 0.3224 |

**Human evaluation, final run vs base** [human-evaluation summary cell]

* 25 unique pairs × 3 raters, sides randomized, ties excluded from the win rate.
* Win rate of the final run by category:
  * in-domain: 28.0% (7 wins / 18 losses / 8 ties);
  * out-of-domain: 72.7% (8 / 3 / 10);
  * compositional: 44.4% (8 / 10 / 3).

## Failure mode 1: compositional prompts lose objects, relations and lighting

**Example.** P20, *"two characters exchanging a scroll in front of a red wooden gate at sunset"*
[failure-mode cell, row 2].

| CLIP | base | r4 | r16 |
|---|---|---|---|
| P20 | 0.3307 | 0.2880 | 0.2993 |

**What the images show.**

* **Base.** Photographic: two people at a red wooden fence with warm backlight.
* **Rank 4.** Two flat-coloured anime figures face each other in front of a red gate, but there is no
  sunset light. The scroll is a small object in one hand.
* **Rank 16.** One character fills the frame and holds up a long blade-like object. The second
  character is cut off at the right edge, so the *exchange* relation is lost. The lighting is flat
  daylight, so *at sunset* is dropped too. The red gate and the anime style are present.

**Pattern across the category.**

* Compositional prompts show the largest CLIP drop of any category: 0.3427 (base) → 0.3224 (r16), −0.020.
  For comparison, in-domain drops by −0.009 and out-of-domain rises by +0.003.
* Within the category, the final run is below base on 4 of 5 prompts [per-prompt CLIP table]:
  P16 −0.010, P17 −0.037, P19 −0.024, P20 −0.031, P18 +0.000.

**Likely cause.**

* The BLIP captions in the training set are short: 8.93 words on average [data-audit cell]. They mostly
  describe one or two people ("a man with a beard"), with almost no lighting, counting or interaction
  words.
* The adapter therefore learns a strong prior for flat-shaded, front-facing character shots and
  over-writes the attribute and relation tokens of longer prompts.
* This is a known weakness of SD 1.5's CLIP text encoder (attribute binding), and the domain prior makes
  it worse.
* Rank 4 vs rank 16 makes no consistent difference here (category means 0.3277 vs 0.3224).

## Failure mode 2: NSFW false positives on harmless in-domain prompts (rank 4)

**Examples.** P04 seed 0, *"a man with pink hair and a hat"*, and P06 seed 0, *"two people hugging each
other people in the background"* [NSFW cell; failure-mode cell, rows 3–4].

**What happened.**

* The Stable Diffusion safety checker flagged both rank-4 images, so they are shown as black tiles and
  are not displayed.
* The same prompts and seeds were **not** flagged for the base model or rank 16.
* The prompts are benign. The checker compares CLIP image embeddings against fixed concept embeddings
  with per-concept thresholds, so it is not a precise classifier.
* Close-up faces and large flat skin-tone regions, which are typical of the anime style, are a plausible
  trigger.
* This NSFW pass agrees with the first pass in the training notebook on all 120 images (0 disagreements).

**Consequences.**

* Rank 4 has 38 releasable images instead of 40 [release cell].
* The two files were deleted from the first public dataset upload [clean-up cell].
* For a deployed adapter, a 2/40 flag rate on harmless prompts means 5% of user requests come back as
  black images.

**Limitation.** With 40 images per model, 2 vs 0 flags is weak statistical evidence. The difference may
not hold on a larger prompt set.

## Failure mode 3: style transfer does not raise prompt alignment, and raters preferred base in-domain

**Observation.**

* Fine-tuning moves in-domain generations much closer to the held-out Naruto frames: KID drops by
  75–76% for both runs (`kid_reduction_vs_base_r4` = 0.764, `_r16` = 0.753) [decision cell].
* But in-domain CLIP falls slightly (0.2953 → 0.2865 for r16).
* Raters preferred base on in-domain pairs (final-run win rate 28.0%).

**Example.** P07, *"a man in a blue jacket and tie"*, has the largest in-domain CLIP drop:
0.3088 → 0.2677 (−0.041) [per-prompt CLIP table].

**Interpretation.**

* KID measures *distribution* match (Inception features) and rewards the anime look.
* CLIP ViT-B/32 measures *text–image* match. The generic captions ("a man in a blue jacket") are matched
  at least as well by the base model's photographic output.
* So the adapter succeeds at the style objective but adds nothing to literal prompt following.
* In the human evaluation, raters judged *domain style and prompt match* together. They preferred the
  final run where style transfer is wanted least (out-of-domain, 72.7%) and the base where the content is
  simplest (in-domain).

**Caveat on the human result.**

* Agreement is low: Fleiss' κ = 0.086, and pairwise agreement is 0.36–0.44 [human-evaluation cell].
* Two raters show a significant side preference [`position_bias`, two-sided binomial test]:
  * rater 2 chose *right* in 19 of 23 non-tie pairs (p = 0.0026);
  * rater 3 chose *left* in 14 of 18 (p = 0.031).
* Because the sides are randomized, this bias adds noise rather than favouring one model. The human win
  rates are therefore weak evidence, and KID/CLIP carry the comparison.

## Failure mode 4 (minor): reduced seed diversity on a photographic still life

**Example.** P14, *"a bowl of fresh tropical fruit on a wooden table"*: LPIPS(seed 0, seed 1) = 0.553
for r16, its lowest of the 20 prompts. The r16 mean is 0.676 [failure-mode cell, row 5].

**What the images show.**

* Both seeds produce the same composition: a top-down bowl of fruit on wooden planks. Only the fruit
  arrangement differs.
* The output stays photographic, so this is mild mode collapse rather than style leakage.

**Overall.** r16 is slightly less diverse than base (0.6761 vs 0.6902) and r4 (0.6924). This fits the
higher-capacity adapter fitting the training distribution more tightly, but the effect is small.

## What did *not* fail: memorization

[memorization cell]

* **Method.** The 10 in-domain generations of r16 were matched against all 1,098 training images by
  OpenCLIP ViT-B/32 cosine similarity.
* **Result.**
  * The highest similarity is 0.928 (P07 vs training image `00814.png`).
  * That is below the 95th percentile of *real held-out test frames* against the same training set
    (0.943): 0/10 generations exceed it.
  * The smallest LPIPS distance of any top-1 pair is 0.553, so these are not pixel-level copies.
* **Control.** Base-model generations average 0.610 top-1 similarity vs 0.810 for r16. The adapter
  moves outputs into the training *style*, not onto specific training frames.
* **Visual check.** In the three closest pairs, the generated characters differ in face, pose and
  clothing from their nearest training frame (memorization figure in `p2_evaluate.ipynb`).
* **Limit.** CLIP similarity is a screening signal, not proof. Carlini et al. (2023) extract
  memorized images with far more samples per prompt than the 2 used here.

## Which run we choose and why

**Final run: LoRA rank 16** [decision cell; `eval/part2_data_and_decision.json`].

| criterion | r4 | r16 | verdict |
|---|---|---|---|
| KID in-domain | 0.0336 ± 0.0155 | 0.0351 ± 0.0162 | gap 0.0015 ≪ subset std 0.016: no difference |
| CLIP, 20 prompts | 0.3067 | 0.3037 | −0.003: negligible |
| LPIPS diversity | 0.6924 | 0.6761 | r4 slightly more diverse |
| NSFW flags / 40 | 2 | 0 | **r16 better** |
| GPU-hours / peak memory | 0.812 / 6.15 GiB | 0.808 / 6.17 GiB | equal |
| adapter size | 797k params | 3.19M params | r4 4× smaller |

**Reasoning.**

* Quality, alignment and cost are statistically indistinguishable between the two ranks on this
  evaluation.
* The only measured difference that affects users is the safety filter: rank 16 produced no blocked
  outputs on the fixed prompts, and rank 4 blocked 2 harmless ones.
* We therefore release rank 16.
* If adapter size or diversity mattered more than this filter result (which rests on only 40 images),
  rank 4 would be an equally defensible choice. The data does not show that the larger rank improves
  image quality.

**What we would change next.**

* Longer, richer captions (re-caption with a stronger VLM) to fix compositional prompts.
* More seeds per prompt for KID and the NSFW comparison.
* A fixed position (or two presentations per pair with sides swapped) in the human evaluation, to cancel
  side bias.
