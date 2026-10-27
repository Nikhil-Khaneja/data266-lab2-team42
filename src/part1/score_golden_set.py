import json
import re
import statistics
import time

from src.part1.answer_engine import AnswerEngine, detect_figure_mentions
from src.part1.conversation import ConversationStore

GOLDEN_SET_PATH = "eval/part1_golden_set.json"
NUMBER_PATTERN = re.compile(r"-?\d+\.?\d*")
# "Figure A-2", "Table 1-2" etc. contain a hyphen+digit that NUMBER_PATTERN misreads as a
# negative number (e.g. "-2"). Strip these labels before extracting numbers for comparison.
LABEL_PATTERN = re.compile(r"(?:Figure|Table)\s+[A-Z0-9]+-?[0-9]*", re.IGNORECASE)


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


def citation_is_supported(citation, retrieved_sources):
    return citation in retrieved_sources


def score_item(item, predicted, retrieved_sources):
    correct = numbers_match(item["expected_answer"], predicted["answer"]) and predicted["status"] == item["expected_status"]
    recall_hit = any(src in retrieved_sources for src in item.get("expected_sources", []))
    citation_hits = [citation_is_supported(c, retrieved_sources) for c in predicted["citations"]]
    citation_accuracy = sum(citation_hits) / len(citation_hits) if citation_hits else 1.0
    return correct, recall_hit, citation_accuracy


def run_golden_set(index, baseline=False):
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    engine = AnswerEngine(index)
    store = ConversationStore()

    correct_count = 0
    recall_hits = 0
    citation_accuracies = []
    durations = []

    for item in golden_set:
        conv = store.new_conversation(item.get("conversation_id", item["id"]))
        start = time.time()
        if baseline:
            retrieved = index.dense_search_text(item["question"], top_k=5)
        else:
            retrieved = index.hybrid_search_text(item["question"], top_k=5)
        for label in detect_figure_mentions(item["question"]):
            retrieved.extend(index.find_figure_by_label(label))
        predicted = engine.answer(conv, item["question"])
        durations.append(time.time() - start)

        retrieved_sources = ["{}:p{}".format(r["doc_id"], r["page"]) for r in retrieved]
        correct, recall_hit, citation_accuracy = score_item(item, predicted, retrieved_sources)
        correct_count += int(correct)
        recall_hits += int(recall_hit)
        citation_accuracies.append(citation_accuracy)

    return {
        "primary_correct_out_of": "{}/{}".format(correct_count, len(golden_set)),
        "secondary_recall_at_5": recall_hits / len(golden_set),
        "secondary_citation_accuracy": sum(citation_accuracies) / len(citation_accuracies),
        "efficiency_median_seconds": statistics.median(durations),
    }


if __name__ == "__main__":
    import sys
    from src.part1.index_store import HybridIndex

    index = HybridIndex(persist_dir="data/processed/part1/chroma_index")
    mode = sys.argv[1] if len(sys.argv) > 1 else "final"
    report = run_golden_set(index, baseline=(mode == "baseline"))
    print(json.dumps(report, indent=2))
