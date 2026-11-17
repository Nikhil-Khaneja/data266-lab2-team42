"""Builds notebooks/p1_human_eval.ipynb: the Part 1 human evaluation (Task 1.4).

Both members rate each of the 25 final golden-set answers 1-5 for groundedness (is every claim supported by
the cited source?). The notebook reads eval/p1_human_ratings.csv and computes the average and how often the
two raters agreed within one point, so the report numbers come from a notebook output.

usage: python scripts/build_p1_human_eval.py && jupyter nbconvert --to notebook --execute --inplace \
           notebooks/p1_human_eval.ipynb
"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
nb = nbf.v4.new_notebook()
C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))      # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))        # noqa: E731

code('''
# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
text=True).stdout.strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})
''')

md('''
# Part 1 - human evaluation of groundedness (Task 1.4)

Both members independently rated each of the 25 final-system answers on the golden set from 1 to 5 for
**groundedness**: is every claim in the answer supported by the source it cites?
The answers are the ones the final system produced in the graded Kaggle run (`notebooks/p1_rag.ipynb`).

| score | meaning |
|---|---|
| 5 | every claim is supported by a cited source |
| 4 | supported, with a minor imprecision (rounding, wording) |
| 3 | mostly supported; one claim is unsupported or cited to the wrong page |
| 2 | several claims are unsupported |
| 1 | the answer is not supported by its citations |
''')

code('''
import os, pandas as pd
path = "eval/p1_human_ratings.csv" if os.path.exists("eval/p1_human_ratings.csv") else "../eval/p1_human_ratings.csv"
h = pd.read_csv(path)
A, B = "rater_A_groundedness_1to5", "rater_B_groundedness_1to5"
assert h[A].notna().all() and h[B].notna().all(), "every answer needs a score from both raters"
assert h[A].between(1, 5).all() and h[B].between(1, 5).all(), "scores must be 1-5"
print("answers rated:", len(h))
''')

code('''
a, b = h[A].astype(float), h[B].astype(float)
summary = {
    "answers rated": len(h),
    "mean groundedness (both raters)": round(((a + b) / 2).mean(), 2),
    "rater A mean": round(a.mean(), 2),
    "rater B mean": round(b.mean(), 2),
    "agreement within 1 point": round(((a - b).abs() <= 1).mean(), 3),
    "exact agreement": round((a == b).mean(), 3),
}
pd.Series(summary).to_frame("value")
''')

code('''
# by golden-set category
h.assign(mean=(a + b) / 2).groupby("category")[[A, B, "mean"]].mean().round(2)
''')

code('''
# answers rated below 4 by either rater, or where the raters differ by more than one point
low = h[(a < 4) | (b < 4) | ((a - b).abs() > 1)]
low[["gid", "category", A, B, "answer", "citations", "notes"]] if len(low) else "none"
''')

os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
out = os.path.join(ROOT, "notebooks", "p1_human_eval.ipynb")
nbf.write(nb, out)
print("wrote", out, len(C), "cells")
