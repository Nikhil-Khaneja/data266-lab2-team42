"""Task 1.4 / 1.5 - running question files, golden-set metrics and the submission file.

Metrics (computed, never typed by hand):
* Primary   - answer correctness on the golden set: correct when the status matches AND every
              key number of the reference answer appears in the answer within +/-1% (sign-insensitive,
              because "fell by 1.9 points" and "-1.9" state the same number).
* Secondary - Retrieval Recall@5 (a gold page/figure/table is among the first 5 retrieved units)
              and citation accuracy (share of cited sources that are gold sources for the question).
* Human     - both members rate groundedness 1-5; mean and within-one-point agreement.
* Efficiency- median seconds per question.
"""
import json
import statistics
import time

import pandas as pd

from .vocab import numbers_in


def run_questions(system, qdf, log=print):
    """qdf has id, conversation_id, turn, question_id, question (as strings). Conversations are answered
    separately, turn by turn, each starting with an empty memory."""
    rows, traces = [], []
    for conv_id, turns in qdf.groupby("conversation_id", sort=False):
        chat = system.new_conversation()
        for _, r in turns.sort_values("turn", key=lambda t: t.astype(int)).iterrows():
            t0 = time.time()
            a = chat.answer(r["question"])
            secs = time.time() - t0
            rows.append({"id": r["id"], "conversation_id": conv_id, "question_id": r.get("question_id", r["id"]),
                         "answer": " ".join(a["answer"].split()),
                         "figure_references": ",".join(a["figure_references"]) or "NONE",
                         "citations": ",".join(a["citations"]) or "NONE", "status": a["status"]})
            tr = dict(chat.last_trace or {})
            tr.update(id=r["id"], conversation_id=conv_id, seconds=secs, output=a)
            traces.append(tr)
            log(f"[{r['id']:>3}] {conv_id} t{r['turn']} {secs:5.1f}s {a['status']:16s} {a['answer'][:110]}")
        del chat                                  # conversation memory is discarded here
    return rows, traces


def write_submission(rows, qdf, path="submission.csv"):
    sub = pd.DataFrame(rows).set_index("id").loc[qdf["id"]].reset_index()
    sub = sub[["id", "conversation_id", "question_id", "answer", "figure_references", "citations", "status"]]
    sub.to_csv(path, index=False)
    return sub


# --------------------------------------------------------------------------- golden set
def load_golden(path):
    g = pd.read_csv(path, dtype=str, keep_default_na=False)
    g["id"] = g["gid"]
    g["question_id"] = g["gid"]
    return g


def _keys(s):
    return [float(x) for x in s.split(",") if x.strip()] if s else []


def numbers_match(keys, text, rel=0.01):
    found = numbers_in(text)
    miss = [k for k in keys if not any(abs(abs(x) - abs(k)) <= max(rel * abs(k), 1e-9) for x in found)]
    return not miss, miss


def _unit_cites(uid, index):
    u = index.by_uid[uid]
    c = {f"{u.doc_id}:p{u.page}"}
    if u.label:
        c.add(f"{u.doc_id}:{u.label}")
    return c


def score_golden(golden, rows, traces, index):
    gold_map = {r["gid"]: r for _, r in golden.iterrows()}
    tr_map = {t["id"]: t for t in traces}
    label_pages = {f"{u.doc_id}:{u.label}": f"{u.doc_id}:p{u.page}" for u in index.units if u.label}
    out = []
    for row in rows:
        g = gold_map[row["id"]]
        tr = tr_map[row["id"]]
        gold = set(x for x in g["gold_sources"].split(",") if x)
        gold_pages = gold | {label_pages[x] for x in gold if x in label_pages}
        status_ok = row["status"] == g["expected_status"]
        nums_ok, miss = numbers_match(_keys(g["key_numbers"]), row["answer"])
        correct = status_ok and nums_ok
        top5 = tr.get("retrieved", [])[:5]
        retrieved = set().union(*[_unit_cites(u, index) for u in top5]) if top5 else set()
        recall = (len(retrieved & gold_pages) > 0) if gold else None
        cites = [c for c in row["citations"].split(",") if c and c != "NONE"]
        support = [c in gold or c in gold_pages for c in cites]
        out.append({"gid": row["id"], "category": g["category"], "status": row["status"],
                    "expected_status": g["expected_status"], "status_ok": status_ok, "numbers_ok": nums_ok,
                    "missing_numbers": ",".join(str(m) for m in miss), "correct": correct,
                    "recall@5": recall, "n_citations": len(cites), "n_supported": sum(support),
                    "seconds": round(tr["seconds"], 2), "answer": row["answer"], "citations": row["citations"],
                    "figure_references": row["figure_references"]})
    df = pd.DataFrame(out)
    rec = df["recall@5"].dropna()
    summary = {
        "answer_correctness": f"{int(df['correct'].sum())}/{len(df)}",
        "answer_correct_n": int(df["correct"].sum()),
        "status_accuracy": round(df["status_ok"].mean(), 3),
        "recall@5": round(rec.astype(float).mean(), 3),
        "citation_accuracy": round(df["n_supported"].sum() / max(df["n_citations"].sum(), 1), 3),
        "median_seconds": round(statistics.median(df["seconds"]), 2),
    }
    return df, summary


# --------------------------------------------------------------------------- human evaluation
def human_eval_template(golden_scored, path):
    t = golden_scored[["gid", "category", "answer", "citations", "status"]].copy()
    t["rater_A_groundedness_1to5"] = ""
    t["rater_B_groundedness_1to5"] = ""
    t.to_csv(path, index=False)
    return path


def human_eval_summary(path):
    h = pd.read_csv(path)
    a, b = h["rater_A_groundedness_1to5"], h["rater_B_groundedness_1to5"]
    ok = a.notna() & b.notna()
    if not ok.any():
        return {"status": "not rated yet"}
    a, b = a[ok].astype(float), b[ok].astype(float)
    return {"n_rated": int(ok.sum()), "mean_groundedness": round(((a + b) / 2).mean(), 2),
            "rater_A_mean": round(a.mean(), 2), "rater_B_mean": round(b.mean(), 2),
            "agreement_within_1": round(((a - b).abs() <= 1).mean(), 3)}


def save_traces(traces, path):
    with open(path, "w") as f:
        for t in traces:
            f.write(json.dumps(t, default=str) + "\n")


# --------------------------------------------------------------------------- bonus: reranker ablation
def retrieval_ablation(golden, traces, index, depth=10, boost_fn=None):
    """Retrieval-only comparison on the golden questions (standalone query and fact-table pins taken from
    each trace, exactly as the answering step used them):
    dense only vs hybrid (dense+BM25, RRF) vs hybrid + cross-encoder reranker.
    Recall@1/3/5: a gold page/figure/table is among the first k units; MRR@10: 1/rank of the first hit."""
    gold_map = {r["gid"]: r for _, r in golden.iterrows()}
    label_pages = {f"{u.doc_id}:{u.label}": f"{u.doc_id}:p{u.page}" for u in index.units if u.label}
    modes = {"dense only": lambda q, b: index.retrieve_dense_only(q, k=depth),
             "hybrid": lambda q, b: index.retrieve(q, k=depth, boost_uids=b, rerank=False)}
    if index.reranker is not None:
        modes["hybrid + reranker"] = lambda q, b: index.retrieve(q, k=depth, boost_uids=b, rerank=True)
    per_q, summary = [], {}
    for name, fn in modes.items():
        hits_at, rr, ms = {1: [], 3: [], 5: []}, [], []
        for t in traces:
            g = gold_map.get(t["id"])
            q = t.get("retrieval_query") or t.get("standalone")
            if g is None or not q or not g["gold_sources"]:
                continue
            gold = set(x for x in g["gold_sources"].split(",") if x)
            gold = gold | {label_pages[x] for x in gold if x in label_pages}
            b = t.get("boost") if t.get("boost") is not None else (boost_fn(q) if boost_fn else [])
            t0 = time.time()
            units = fn(q, b)
            ms.append((time.time() - t0) * 1000)
            ranks = [i + 1 for i, u in enumerate(units) if _unit_cites(u.uid, index) & gold]
            first = ranks[0] if ranks else None
            for k in hits_at:
                hits_at[k].append(first is not None and first <= k)
            rr.append(1.0 / first if first else 0.0)
            per_q.append({"mode": name, "gid": t["id"], "first_hit_rank": first})
        n = len(rr)
        summary[name] = {"recall@1": round(sum(hits_at[1]) / n, 3), "recall@3": round(sum(hits_at[3]) / n, 3),
                         "recall@5": round(sum(hits_at[5]) / n, 3), "MRR@10": round(sum(rr) / n, 3),
                         "median_retrieval_ms": round(statistics.median(ms), 1), "questions": n}
    return pd.DataFrame(summary).T, pd.DataFrame(per_q)
