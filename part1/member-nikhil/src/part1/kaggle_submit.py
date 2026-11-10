import json
import os
import subprocess
import sys
import time

import pandas as pd

from src.part1.answer_engine import AnswerEngine
from src.part1.config import load_config
from src.part1.conversation import ConversationStore

QUESTIONS_PATH = os.environ.get("QUESTIONS_PATH", "Questions.csv")
INDEX_DIR = os.environ.get("INDEX_DIR", "chroma_index")


def build_submission(questions_path=None, index_dir=None, out_path="submission.csv", index=None, config=None):
    config = config or load_config()
    if index is None:
        from src.part1.index_store import HybridIndex
        index = HybridIndex(index_dir or INDEX_DIR, text_model_name=config.embedder,
                            reranker_name=config.reranker or None)
    engine = AnswerEngine(index, config=config)
    store = ConversationStore()

    q = pd.read_csv(questions_path or QUESTIONS_PATH, dtype=str, keep_default_na=False)
    rows = []
    started = time.time()
    for conv_id, turns in q.groupby("conversation_id", sort=False):
        conversation = store.new_conversation(conv_id)
        turns_sorted = turns.sort_values("turn", key=lambda t: t.astype(int))
        for _, r in turns_sorted.iterrows():
            a = engine.answer(conversation, r["question"])
            rows.append({
                "id": r["id"],
                "conversation_id": conv_id,
                "question_id": r["question_id"],
                "answer": " ".join(a["answer"].split()),
                "figure_references": ",".join(a["figure_references"]) or "NONE",
                "citations": ",".join(a["citations"]) or "NONE",
                "status": a["status"],
            })

    submission = pd.DataFrame(rows).set_index("id").loc[q["id"]].reset_index()
    submission.to_csv(out_path, index=False)
    write_run_manifest(out_path, config, len(q), time.time() - started, submission)
    return submission


def write_run_manifest(out_path, config, n_questions, seconds, submission):
    """Record exactly what produced this submission.csv, so a leaderboard score can be
    traced back to one configuration (and one git commit) instead of guessed at."""
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    manifest = {
        "commit": commit,
        "config": config.as_dict(),
        "n_questions": n_questions,
        "seconds": round(seconds, 1),
        "status_counts": submission["status"].value_counts().to_dict(),
        "rows_with_no_citations": int((submission["citations"] == "NONE").sum()),
        "rows_with_figure_refs": int((submission["figure_references"] != "NONE").sum()),
    }
    manifest_path = os.path.join(os.path.dirname(os.path.abspath(out_path)), "run_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def find_data_dir():
    """Locate the competition files (kb_manifest.json + PDFs) on Kaggle or locally."""
    candidates = [os.environ.get("DATA_DIR", ""), "data/raw/part1", "../data/raw/part1"]
    for path in candidates:
        if path and os.path.exists(os.path.join(path, "kb_manifest.json")):
            return path
    # Kaggle's mount layout varies (/kaggle/input/<slug> or /kaggle/input/competitions/<slug>).
    for root, _, files in os.walk("/kaggle/input"):
        if "kb_manifest.json" in files:
            return root
    raise FileNotFoundError("kb_manifest.json not found; set DATA_DIR to the folder holding the competition files")


def ensure_index(data_dir, work_dir, config=None):
    from src.part1.describe_figures import describe_figures
    from src.part1.index_store import HybridIndex
    from src.part1.ingest import ingest_documents

    config = config or load_config()
    index = HybridIndex(os.path.join(work_dir, "chroma_index"), text_model_name=config.embedder,
                        reranker_name=config.reranker or None)
    if index.collection.count() == 0:
        chunks, _ = ingest_documents(
            os.path.join(data_dir, "kb_manifest.json"), data_dir, os.path.join(work_dir, "figures"))
        describe_figures(chunks, cache_path=os.path.join(work_dir, "figure_descriptions.json"))
        index.build(chunks)
    return index


def main():
    """End to end, no manual step: ingest -> index -> answer every conversation -> validate."""
    data_dir = find_data_dir()
    work_dir = os.environ.get("WORK_DIR", "/kaggle/working" if os.path.isdir("/kaggle/working") else ".")
    questions_path = os.environ.get("QUESTIONS_PATH") or os.path.join(data_dir, "Questions.csv")
    out_path = os.path.join(work_dir, "submission.csv")

    index = ensure_index(data_dir, work_dir)
    submission = build_submission(questions_path, out_path=out_path, index=index)
    print(submission[["id", "status", "figure_references", "citations"]].to_string())

    validator = os.path.join(data_dir, "validate_submission.py")
    if os.path.exists(validator):
        result = subprocess.run([sys.executable, validator, out_path, questions_path], capture_output=True, text=True)
        print(result.stdout, result.stderr)
        if result.returncode != 0:
            raise SystemExit("validate_submission.py failed -- do not submit this file")
    return submission


if __name__ == "__main__":
    main()
