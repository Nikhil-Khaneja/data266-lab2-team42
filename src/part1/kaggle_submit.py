import os

import pandas as pd

from src.part1.answer_engine import AnswerEngine
from src.part1.conversation import ConversationStore
from src.part1.index_store import HybridIndex

QUESTIONS_PATH = os.environ.get("QUESTIONS_PATH", "Questions.csv")
INDEX_DIR = os.environ.get("INDEX_DIR", "chroma_index")


def build_submission():
    index = HybridIndex(persist_dir=INDEX_DIR)
    engine = AnswerEngine(index)
    store = ConversationStore()

    q = pd.read_csv(QUESTIONS_PATH, dtype=str, keep_default_na=False)
    rows = []
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
    submission.to_csv("submission.csv", index=False)
    return submission


if __name__ == "__main__":
    build_submission()
