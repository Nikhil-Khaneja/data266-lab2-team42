import csv
import json
import re
import statistics
import time

from src.part1.answer_engine import AnswerEngine, detect_figure_mentions
from src.part1.conversation import ConversationStore

GOLDEN_SET_PATH = "../eval/part1_golden_set.json"  # shared definition, one level up
NUMBER_PATTERN = re.compile(r"(?<![\w.-])-?\d+(?:\.\d+)?")
# "Figure A-2", "Table 1-2" etc. contain a hyphen+digit that NUMBER_PATTERN misreads as a
# negative number (e.g. "-2"). "Document 1"/"Document 2" are source attributions, not
# required facts, but their trailing digit is still a bare number NUMBER_PATTERN would
# pick up. Strip all three label kinds before extracting numbers for comparison.
LABEL_PATTERN = re.compile(r"(?:Figure|Table|Document)\s+[A-Z0-9]+-?[0-9]*", re.IGNORECASE)


def extract_numbers(text):
    text = LABEL_PATTERN.sub(" ", text)
    return [float(n) for n in NUMBER_PATTERN.findall(text)]


def number_present(expected, actual_numbers, tolerance):
    for actual in actual_numbers:
        if expected == 0:
            if abs(actual) <= tolerance:
                return True
        elif abs(actual - expected) / abs(expected) <= tolerance:
            return True
    return False


def numbers_match(expected_text, actual_text, tolerance=0.01):
    # Every number the golden answer requires must appear somewhere in the predicted
    # answer, within tolerance. We don't require the same count/order: a longer,
    # fully-worded answer that restates the right number is still correct.
    expected_numbers = extract_numbers(expected_text)
    actual_numbers = extract_numbers(actual_text)
    if not expected_numbers:
        return True
    return all(number_present(expected, actual_numbers, tolerance) for expected in expected_numbers)


def citation_is_supported(citation, shown):
    # A citation is supported when it names a page or labelled object the model was shown.
    for r in shown:
        if citation == "{}:p{}".format(r["doc_id"], r["page"]):
            return True
        if r.get("figure_id") and citation == "{}:{}".format(r["doc_id"], r["figure_id"]):
            return True
    return False


def score_item(item, predicted, recall_sources, shown):
    correct = numbers_match(item["expected_answer"], predicted["answer"]) and predicted["status"] == item["expected_status"]
    expected = item.get("expected_sources", [])
    recall_hit = (not expected) or any(src in recall_sources for src in expected)
    hits = [citation_is_supported(c, shown) for c in predicted["citations"]]
    return correct, recall_hit, (sum(hits) / len(hits) if hits else 1.0)


def cited_text(citations, shown, max_chars=1200):
    """The text of each cited source the model was shown, for the human groundedness rating."""
    parts = []
    for c in citations:
        for r in shown:
            ids = {"{}:p{}".format(r["doc_id"], r["page"])}
            if r.get("figure_id"):
                ids.add("{}:{}".format(r["doc_id"], r["figure_id"]))
            if c in ids:
                parts.append("[{}] {}".format(c, " ".join(r.get("text", "").split())[:max_chars]))
                break
        else:
            parts.append("[{}] (not among the sources shown to the model)".format(c))
    return "\n\n".join(parts)


def run_golden_set(index, baseline=False, rerank=True):
    """baseline: dense retrieval only. rerank=False: the final pipeline with the cross-encoder
    switched off, so the reranker's own effect can be measured (everything else unchanged)."""
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
    saved_reranker = index.reranker
    if not rerank:
        index.reranker = None
    try:
        return _run(index, golden_set, baseline)
    finally:
        index.reranker = saved_reranker


def _run(index, golden_set, baseline):

    engine = AnswerEngine(index, retrieval_mode=("dense" if baseline else "hybrid"))
    store = ConversationStore()

    correct_count = 0
    recall_hits = 0
    citation_accuracies = []
    durations = []
    details = []

    for item in golden_set:
        conv = store.new_conversation(item.get("conversation_id", item["id"]))
        start = time.time()
        predicted = engine.answer(conv, item["question"])
        durations.append(time.time() - start)
        shown = engine.last_shown

        # Recall@5 is measured at k=5 as the lab specifies, plus exact-label lookups.
        if baseline:
            top5 = index.dense_search_text(item["question"], top_k=5)
        else:
            top5 = index.hybrid_search_text(item["question"], top_k=5)
        named = [c for label in detect_figure_mentions(item["question"]) for c in index.find_by_label(label)]
        recall_sources = ["{}:p{}".format(r["doc_id"], r["page"]) for r in top5 + named]

        correct, recall_hit, citation_accuracy = score_item(item, predicted, recall_sources, shown)
        correct_count += int(correct)
        recall_hits += int(recall_hit)
        citation_accuracies.append(citation_accuracy)
        details.append({"id": item["id"], "question": item["question"], "correct": correct,
                        "recall_hit": recall_hit, "expected_status": item["expected_status"],
                        "predicted": predicted, "cited_text": cited_text(predicted["citations"], shown)})

    return {
        "primary_correct_out_of": "{}/{}".format(correct_count, len(golden_set)),
        "secondary_recall_at_5": recall_hits / len(golden_set),
        "secondary_citation_accuracy": sum(citation_accuracies) / len(citation_accuracies),
        "efficiency_median_seconds": statistics.median(durations),
        "details": details,
    }


RATING_COLUMNS = ["rater_nikhil", "rater_anushka", "note"]


def write_rating_sheet(report, path):
    """One row per golden-set answer with its cited source text and empty 1-5 rating columns."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "status", "answer", "citations", "figure_references", "cited_text"] + RATING_COLUMNS)
        for d in report["details"]:
            p = d["predicted"]
            w.writerow([d["id"], d["question"], p["status"], p["answer"], ",".join(p["citations"]),
                        ",".join(p.get("figure_references", [])), d["cited_text"]] + [""] * len(RATING_COLUMNS))


def summarize_ratings(path, raters=("rater_nikhil", "rater_anushka")):
    """Mean groundedness per rater and overall, and the share of items rated within one point."""
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if all(r[x].strip() for x in raters)]
    scores = {x: [int(r[x]) for r in rows] for x in raters}
    pairs = list(zip(*scores.values()))
    if not pairs:
        return {"items_rated": 0}
    return {"items_rated": len(rows),
            "mean_by_rater": {x: statistics.mean(v) for x, v in scores.items()},
            "mean_overall": statistics.mean([s for v in scores.values() for s in v]),
            "within_one_point": sum(max(p) - min(p) <= 1 for p in pairs) / len(pairs),
            "exact_agreement": sum(len(set(p)) == 1 for p in pairs) / len(pairs)}


if __name__ == "__main__":
    import sys
    from src.part1.kaggle_submit import ensure_index, find_data_dir

    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    index = ensure_index(find_data_dir(), "data/processed/part1")
    report = run_golden_set(index, baseline=(mode == "baseline"), rerank=(mode != "no_rerank"))
    print(json.dumps(report, indent=2))
