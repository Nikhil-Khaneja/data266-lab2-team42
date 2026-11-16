"""Builds notebooks/p1_rag.ipynb from the p1_rag package so the Kaggle notebook is self-contained
(the package source is written out with %%writefile; no private-repo clone is needed on Kaggle).

usage: python scripts/build_p1_notebook.py
"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULES = ["__init__.py", "vocab.py", "ingest.py", "llm.py", "figures.py", "index.py", "facts.py",
           "answer.py", "evaluate.py"]

nb = nbf.v4.new_notebook()
import argparse, subprocess as _sp
_ap = argparse.ArgumentParser()
_ap.add_argument("--commit", default="", help="git commit of the p1_rag code (printed by Cell 1 when the run has no .git)")
_args = _ap.parse_args()
BUILD_COMMIT = _args.commit or _sp.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                                       text=True).stdout.strip() or "unknown (built outside git)"

C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))      # noqa: E731
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))        # noqa: E731

code('''
# Cell 1 - run information (required)
import subprocess, sys, random, numpy as np, torch
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
text=True).stdout.strip()
commit = commit or "BUILD_COMMIT"   # no .git on Kaggle: commit of the p1_rag code this notebook was built from
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})
'''.replace("BUILD_COMMIT", "code@" + BUILD_COMMIT))

md('''
# Part 1 - Multimodal RAG over a read-only knowledge base (Team 42)

**Pipeline**

1. **Ingestion (1.1)** - PyMuPDF splits the three PDFs into ~140-word text chunks (doc_id + page). Every numbered
   Table and Figure is cropped as an image with its label and caption (captions are found by their bold font, so
   in-text mentions are not mistaken for captions; *Figure B-5* and *Figure B-5 (corrected)* are separate objects).
   Tables are also parsed into a grid and into structured facts (entity, variable, year, value, forecast?, trust).
2. **Figure search (1.1.3)** - Qwen2.5-VL-3B-Instruct writes a text description of every figure (axes, series,
   printed values, other text in the image). Text inside an image that looks like an instruction is fenced off.
3. **Vector DB (1.1.4)** - ChromaDB with `doc_id, page, figure_id, trust_level, kind` metadata, trust read from
   `kb_manifest.json` (Appendix C pages = low).
4. **Retrieval (1.2)** - BAAI/bge-small-en-v1.5 dense + BM25, reciprocal-rank fusion; a named Figure/Table is
   fetched by exact label (plus its corrected version) and other figures are kept out of the context; low-trust
   units are never returned.
5. **Answering (1.2.3)** - a structured precedence layer applies the KB rules (revision > outturn > forecast;
   disagreement without a revision note = conflict, prefer higher trust). Qwen2.5-VL-3B writes the JSON answer
   from the evidence (and the figure image when a figure is named). Citations come from the evidence it used.
6. **Conversations (1.3)** - one `Conversation` object per conversation_id: follow-ups are rewritten with that
   conversation's history only; user-supplied values live in that object (never in ChromaDB) and disappear with it;
   permanent-change requests are refused.

Model versions: `Qwen/Qwen2.5-VL-3B-Instruct` (Hugging Face, fp16 on one T4/P100) and `BAAI/bge-small-en-v1.5`.
No hosted API is used.
''')

code('''
# Kaggle / Colab: install the pinned extras (Kaggle images already ship torch)
import importlib, subprocess, sys
need = {"fitz": "pymupdf==2.1", "chromadb": "chromadb==1.5.9", "rank_bm25": "rank_bm25==0.2.2",
        "sentence_transformers": "sentence-transformers==6.3.0", "accelerate": "accelerate"}
missing = [pkg for mod, pkg in need.items() if importlib.util.find_spec(mod) is None]
import transformers
from packaging.version import Version
if Version(transformers.__version__) < Version("4.51.0"):      # Qwen2.5-VL support
    missing.append("transformers==5.22.2")
if missing:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=True)
print("installed:", missing or "nothing needed")
''')

for m in MODULES:
    with open(os.path.join(ROOT, "p1_rag", m)) as f:
        src = f.read()
    code(f"%%writefile p1_rag/{m}\n" + src.rstrip() if m != "__init__.py" else
         "import os; os.makedirs('p1_rag', exist_ok=True)\n" +
         f"open('p1_rag/__init__.py', 'w').write({src!r})")

with open(os.path.join(ROOT, "eval", "p1_golden_set.csv")) as f:
    golden = f.read()
code("import os; os.makedirs('eval', exist_ok=True)\n"
     "# golden set: eval/p1_golden_set.csv in the repo. Written by the team from the three PDFs on 2026-11-11, after\n"
     "# tuning on the public practice questions had started (disclosed in the report); three questions were reworded\n"
     "# later so none resembles a practice question. Answers and sources were not changed.\n"
     f"open('eval/p1_golden_set.csv', 'w').write({golden!r})\nprint('golden rows:', {golden.count(chr(10)) - 1})")

ratings = os.path.join(ROOT, "eval", "p1_human_ratings.csv")
if os.path.exists(ratings):
    with open(ratings) as f:
        code("open('eval/p1_human_ratings.csv', 'w').write(" + repr(f.read()) + ")")

code('''
import os, glob, json, time, importlib, pandas as pd
import transformers, chromadb, sentence_transformers, fitz
print({"transformers": transformers.__version__, "chromadb": chromadb.__version__,
       "sentence_transformers": sentence_transformers.__version__, "pymupdf": fitz.VersionBind})

# data location: Kaggle competition input, or ./data locally
cands = glob.glob("/kaggle/input/**/kb_manifest.json", recursive=True) + glob.glob("data/kb_manifest.json") \\
        + glob.glob("../data/kb_manifest.json")
DATA_DIR = os.path.dirname(cands[0])
QUESTIONS_PATH = os.environ.get("QUESTIONS_PATH", "Questions.csv")    # graders swap in the hidden set
if not os.path.exists(QUESTIONS_PATH):
    QUESTIONS_PATH = os.path.join(DATA_DIR, "Questions.csv")
OUT = "artifacts/p1"
print("DATA_DIR:", DATA_DIR, "| QUESTIONS_PATH:", QUESTIONS_PATH)
''')

md("## 1.1 Ingestion & indexing")
code('''
from p1_rag.ingest import ingest
t0 = time.time()
units, facts, docs = ingest(DATA_DIR, OUT)
print(f"ingested in {time.time()-t0:.1f}s")
print(pd.DataFrame([(u.doc_id, u.kind) for u in units], columns=["doc_id", "kind"]).value_counts().unstack(fill_value=0))
objs = pd.DataFrame([{"doc_id": u.doc_id, "page": u.page, "kind": u.kind, "label": u.label, "trust": u.trust,
                      "caption": u.caption[:70]} for u in units if u.kind != "text"])
objs
''')
code('''
fdf = pd.DataFrame(facts)
print(len(fdf), "structured facts; low-trust facts (never used):", int((fdf.trust == "low").sum()))
fdf[fdf.kind != "table"][["doc_id", "page", "kind", "trust", "entity", "variable", "year", "value", "revised_from", "raw"]]
''')
code('''
from p1_rag.llm import VLM
from p1_rag.figures import describe_figures
vlm = VLM()
print("VLM backend:", vlm.backend, "|", vlm.model_name)
t0 = time.time()
describe_figures(units, vlm, cache_path=f"{OUT}/figure_descriptions.json")
print(f"figure descriptions: {time.time()-t0:.0f}s")
for u in units:
    if u.label in ("Figure A-3", "Figure C-1"):
        print("=====", u.label, "| embedded instruction fenced off:", u.extra.get("embedded_instruction", "-"))
        print(u.extra["vlm_description"][:900])
''')
code('''
from p1_rag.index import KBIndex
t0 = time.time()
index = KBIndex(units, persist_dir=f"{OUT}/chroma")
print(f"indexed {index.col.count()} units in ChromaDB in {time.time()-t0:.1f}s")
print(index.col.get(ids=["DOC2|Figure A-3", "DOC3|Figure B-5 (corrected)", "DOC1|Table 1-2"], include=["metadatas"])["metadatas"])
''')
code('''
# Bonus: cross-encoder reranker over the top 20 hybrid candidates (named objects and fact tables stay pinned)
from p1_rag.index import RERANK_MODEL
USE_RERANKER = True
t0 = time.time()
if USE_RERANKER:
    index.enable_reranker(RERANK_MODEL)
print("reranker:", index.rerank_model, f"| loaded in {time.time()-t0:.0f}s")
''')

md("## 1.2 Retrieval & grounded answers")
code('''
for q in ["On the rebased chart, Figure A-5, what level had the Euro area series reached by December 2010?",   # golden G07
          "How does the Okun's-law slope in Document 1 compare with Briefing B's re-estimate?"]:      # golden G14
    print("Q:", q)
    print("  hybrid+rerank:", [u.uid for u in index.retrieve(q, k=5)])
    print("  hybrid :", [u.uid for u in index.retrieve(q, k=5, rerank=False)])
    print("  dense  :", [u.uid for u in index.retrieve_dense_only(q, k=5)])
''')
code('''
from p1_rag.answer import RAGSystem
from p1_rag import evaluate as ev
final = RAGSystem(index, facts, docs, vlm, mode="final")                       # hybrid + reranker
hybrid = RAGSystem(index, facts, docs, vlm, mode="final", rerank=False)        # same system, no reranker
baseline = RAGSystem(index, facts, docs, vlm, mode="baseline")                 # dense retrieval only
def new_conversation():          # empty memory for one conversation_id
    return final.new_conversation()
''')

md("## 1.5 Kaggle submission (Questions.csv or the hidden set via QUESTIONS_PATH)")
code('''
q = pd.read_csv(QUESTIONS_PATH, dtype=str, keep_default_na=False)
t0 = time.time()
rows, traces = ev.run_questions(final, q)
print(f"answered {len(rows)} questions in {time.time()-t0:.0f}s")
sub = ev.write_submission(rows, q, "submission.csv")
ev.save_traces(traces, f"{OUT}/questions_traces.jsonl")
subprocess.run([sys.executable, os.path.join(DATA_DIR, "validate_submission.py"), "submission.csv", QUESTIONS_PATH])
sub.head(10)
''')
code('''
# Same questions with the reranker switched off (comparison file; submission.csv above is the final system)
if index.reranker is not None:
    t0 = time.time()
    rows_h, traces_h = ev.run_questions(hybrid, q, log=lambda s: None)
    sub_h = ev.write_submission(rows_h, q, "submission_no_rerank.csv")
    print(f"answered {len(rows_h)} questions without the reranker in {time.time()-t0:.0f}s")
    subprocess.run([sys.executable, os.path.join(DATA_DIR, "validate_submission.py"), "submission_no_rerank.csv", QUESTIONS_PATH])
    diff = (sub.set_index("id")[["answer", "status", "citations"]] != sub_h.set_index("id")[["answer", "status", "citations"]]).any(axis=1)
    print("rows that differ with / without the reranker:", list(diff[diff].index))
''')

md("## 1.4 Evaluation on our golden set - baseline (dense only) vs final (hybrid + reranker)\n"
   "The middle column is the final system with the reranker switched off (bonus ablation).")
code('''
g = ev.load_golden("eval/p1_golden_set.csv")
results, scored = {}, {}
for name, system in [("baseline", baseline), ("hybrid_no_reranker", hybrid), ("final", final)]:
    rows_g, traces_g = ev.run_questions(system, g, log=lambda s: None)
    ev.save_traces(traces_g, f"{OUT}/golden_traces_{name}.jsonl")
    scored[name], results[name] = ev.score_golden(g, rows_g, traces_g, index)
    scored[name].to_csv(f"{OUT}/golden_scored_{name}.csv", index=False)
res = pd.DataFrame(results).loc[["answer_correctness", "status_accuracy", "recall@5", "citation_accuracy", "median_seconds"]]
res
''')
code('''
pd.set_option("display.max_colwidth", 120)
scored["final"][["gid", "category", "expected_status", "status", "correct", "missing_numbers", "recall@5", "answer"]]
''')
code('''
print(scored["final"].groupby("category")[["correct", "status_ok"]].mean())
ev.human_eval_template(scored["final"], f"{OUT}/p1_human_ratings_template.csv")
hpath = "eval/p1_human_ratings.csv"
print("human evaluation:", ev.human_eval_summary(hpath) if os.path.exists(hpath) else "ratings file not filled yet")
''')

md("## Bonus - reranker improvement on the golden set (retrieval only)\n"
   "Same queries and fact-table pins as the answering step; Recall@k = a gold page/figure/table in the top k.")
code('''
tr_final = [json.loads(l) for l in open(f"{OUT}/golden_traces_final.jsonl")]
rr_summary, rr_perq = ev.retrieval_ablation(g, tr_final, index)
rr_summary.to_csv(f"{OUT}/reranker_ablation.csv")
pv = rr_perq.pivot(index="gid", columns="mode", values="first_hit_rank")
print("questions where the first gold hit moved:")
print(pv[pv.nunique(axis=1, dropna=False) > 1])
rr_summary
''')

md("## C4 Safety - untrusted Appendix C (text and chart)")
code('''
for name in ["final"]:
    tr = [json.loads(l) for l in open(f"{OUT}/golden_traces_{name}.jsonl")]
low = [u.uid for u in units if u.trust == "low"]
print("low-trust units kept out of evidence:", low)
hits = [t["id"] for t in tr if any(r in low for r in t.get("retrieved", []))]
print("golden questions whose evidence contained a low-trust unit:", hits or "none")
print(scored["final"][scored["final"].category == "unanswerable_injection"][["gid", "status", "answer", "citations"]].to_string())
''')

os.makedirs(os.path.join(ROOT, "notebooks"), exist_ok=True)
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(ROOT, "notebooks", "p1_rag.ipynb")
nbf.write(nb, out)
print("wrote", out, len(nb.cells), "cells")
