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


def run_golden_set(index, baseline=False):
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

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
        details.append({"id": item["id"], "correct": correct, "recall_hit": recall_hit,
                        "expected_status": item["expected_status"], "predicted": predicted})

    return {
        "primary_correct_out_of": "{}/{}".format(correct_count, len(golden_set)),
        "secondary_recall_at_5": recall_hits / len(golden_set),
        "secondary_citation_accuracy": sum(citation_accuracies) / len(citation_accuracies),
        "efficiency_median_seconds": statistics.median(durations),
        "details": details,
    }


if __name__ == "__main__":
    import sys
    from src.part1.kaggle_submit import ensure_index, find_data_dir

    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    index = ensure_index(find_data_dir(), "data/processed/part1")
    report = run_golden_set(index, baseline=(mode == "baseline"))
    print(json.dumps(report, indent=2))
