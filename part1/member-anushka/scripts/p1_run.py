"""End-to-end Part 1 run (same steps as notebooks/p1_rag.ipynb).

usage: python scripts/p1_run.py [--data data] [--out artifacts/p1] [--skip-golden] [--skip-questions]
"""
import argparse
import json
import os
import subprocess
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from p1_rag.ingest import ingest                       # noqa: E402
from p1_rag.figures import describe_figures            # noqa: E402
from p1_rag.index import KBIndex                       # noqa: E402
from p1_rag.llm import VLM                             # noqa: E402
from p1_rag.answer import RAGSystem                    # noqa: E402
from p1_rag import evaluate as ev                      # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="data")
ap.add_argument("--out", default="artifacts/p1")
ap.add_argument("--golden", default="eval/p1_golden_set.csv")
ap.add_argument("--questions", default=os.environ.get("QUESTIONS_PATH", "data/Questions.csv"))
ap.add_argument("--skip-golden", action="store_true")
ap.add_argument("--skip-questions", action="store_true")
ap.add_argument("--only", default="", help="comma list of question ids to run (debug)")
ap.add_argument("--no-rerank", action="store_true", help="final system without the reranker")
args = ap.parse_args()

units, facts, docs = ingest(args.data, args.out)
vlm = VLM()
describe_figures(units, vlm, cache_path=os.path.join(args.out, "figure_descriptions.json"))
index = KBIndex(units, persist_dir=os.path.join(args.out, "chroma"))
if not args.no_rerank:
    index.enable_reranker()
final = RAGSystem(index, facts, docs, vlm, mode="final")
hybrid = RAGSystem(index, facts, docs, vlm, mode="final", rerank=False)
base = RAGSystem(index, facts, docs, vlm, mode="baseline")
print("backend:", vlm.backend, vlm.model_name, "| units:", len(units), "| facts:", len(facts))

if not args.skip_questions:
    q = pd.read_csv(args.questions, dtype=str, keep_default_na=False)
    if args.only:
        q = q[q["id"].isin(args.only.split(","))]
    rows, traces = ev.run_questions(final, q)
    ev.save_traces(traces, os.path.join(args.out, "questions_traces.jsonl"))
    if not args.only:
        ev.write_submission(rows, q, os.path.join(args.out, "submission.csv"))
        subprocess.run([sys.executable, os.path.join(args.data, "validate_submission.py"),
                        os.path.join(args.out, "submission.csv"), args.questions])

if not args.skip_golden:
    g = ev.load_golden(args.golden)
    results = {}
    systems = [("baseline", base)] + ([("hybrid_no_reranker", hybrid)] if index.reranker else []) + [("final", final)]
    for name, system in systems:
        rows, traces = ev.run_questions(system, g)
        ev.save_traces(traces, os.path.join(args.out, f"golden_traces_{name}.jsonl"))
        df, summary = ev.score_golden(g, rows, traces, index)
        df.to_csv(os.path.join(args.out, f"golden_scored_{name}.csv"), index=False)
        results[name] = summary
        print(name, summary)
    with open(os.path.join(args.out, "golden_summary.json"), "w") as f:
        json.dump(results, f, indent=1)
    print(pd.DataFrame(results))
    if index.reranker is not None:
        tr_final = [json.loads(l) for l in open(os.path.join(args.out, "golden_traces_final.jsonl"))]
        rr, perq = ev.retrieval_ablation(g, tr_final, index)
        rr.to_csv(os.path.join(args.out, "reranker_ablation.csv"))
        print(rr.to_string())
