import csv
import json
import os
import statistics
import time

from PIL import Image

from src.part3.orchestrator import run_request, save_trace

SCENARIOS_PATH = "../eval/part3_scenarios.json"   # shared definitions, written before any run
REDTEAM_PATH = "../eval/part3_redteam.json"
IMAGES_DIR = "data/raw/part3"
TRACE_DIR = "eval/part3_traces"
OUTPUT_IMAGE_DIR = "data/processed/part3/images"  # generated images stay out of git
HUMAN_DIR = "eval/part3_human_eval"
RATERS = ("nikhil", "anushka")


def load_image_if_available(file_name):
    if not file_name:
        return None
    path = os.path.join(IMAGES_DIR, file_name)
    if not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        # Broken/empty/truncated file: handled like a missing image, so the orchestrator answers
        # with a clear message instead of the batch crashing (this is what those cases test).
        return None


def run_case(case, condition, use_critique):
    image = load_image_if_available(case["input_image"])
    start = time.time()
    state = run_request(case["request"], image, use_critique=use_critique)
    seconds = time.time() - start
    trace_dir = os.path.join(TRACE_DIR, condition)
    os.makedirs(trace_dir, exist_ok=True)
    trace_path = os.path.join(trace_dir, "{}.json".format(case["id"]))
    save_trace(state, trace_path, os.path.join(OUTPUT_IMAGE_DIR, condition))
    return {
        "id": case["id"], "category": case["category"], "condition": condition,
        "route": state.route, "refused": state.refused, "refusal_reason": state.refusal_reason,
        "final_status": state.final_status, "critique_history": state.critique_history,
        "revision_count": state.revision_count, "generation_model": state.generation_model,
        "seconds": round(seconds, 2), "trace": trace_path,
    }


def run_all():
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)
    with open(REDTEAM_PATH, "r", encoding="utf-8") as f:
        redteam = json.load(f)
    results = {
        "with_critique": [run_case(s, "with_critique", True) for s in scenarios],
        "no_critique": [run_case(s, "no_critique", False) for s in scenarios],
        "redteam": [run_case(c, "redteam", True) for c in redteam],
    }
    with open("eval/part3_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    write_human_templates(scenarios, redteam, results)
    return results


def _write_if_absent(path, header, rows):
    # Never overwrite a sheet someone has started filling in.
    if os.path.exists(path):
        return
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_human_templates(scenarios, redteam, results):
    os.makedirs(HUMAN_DIR, exist_ok=True)
    rubric = {s["id"]: s["rubric"] for s in scenarios}
    rows = [[r["id"], r["condition"], r["final_status"], rubric[r["id"]]] + [""] * (len(RATERS) + 2)
            for cond in ("with_critique", "no_critique") for r in results[cond]]
    _write_if_absent(os.path.join(HUMAN_DIR, "scenario_judgements.csv"),
                     ["id", "condition", "final_status", "rubric"] + ["pass_" + r for r in RATERS]
                     + ["agreed_pass", "resolution_note"], rows)
    # Critic agreement: one row per critique decision, judged on the image the critic saw.
    rows = [[r["id"], i, d] + [""] * len(RATERS)
            for r in results["with_critique"] for i, d in enumerate(r["critique_history"])]
    _write_if_absent(os.path.join(HUMAN_DIR, "critic_agreement.csv"),
                     ["id", "critique_step", "critic_decision"] + ["ours_" + r for r in RATERS], rows)
    expected = {c["id"]: c.get("expected", "") for c in redteam}
    rows = [[r["id"], r["category"], expected[r["id"]], r["refused"], r["refusal_reason"], ""]
            for r in results["redteam"]]
    _write_if_absent(os.path.join(HUMAN_DIR, "redteam_judgements.csv"),
                     ["id", "category", "expected", "refused", "refusal_reason", "attack_succeeded"], rows)


def _yes(value):
    return str(value).strip().lower() in ("1", "y", "yes", "true", "pass", "p")


def summarize(results):
    """The four Part 3 measurements. Human columns count only once every row is filled."""
    summary = {}
    for cond in ("with_critique", "no_critique"):
        secs = [r["seconds"] for r in results[cond]]
        gen = [r["seconds"] for r in results[cond] if r["generation_model"]]
        summary[cond] = {"median_seconds_per_request": statistics.median(secs),
                         "median_seconds_per_generation_request": statistics.median(gen) if gen else None}

    path = os.path.join(HUMAN_DIR, "scenario_judgements.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []
    for cond in ("with_critique", "no_critique"):
        mine = [r for r in rows if r["condition"] == cond]
        if mine and all(r["agreed_pass"].strip() for r in mine):
            summary[cond]["task_success"] = "{}/{}".format(sum(_yes(r["agreed_pass"]) for r in mine), len(mine))
        if mine and all(r["pass_" + x].strip() for r in mine for x in RATERS):
            agree = sum(len({_yes(r["pass_" + x]) for x in RATERS}) == 1 for r in mine)
            summary[cond]["rater_agreement"] = "{}/{}".format(agree, len(mine))

    path = os.path.join(HUMAN_DIR, "critic_agreement.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []
    if rows and all(r["ours_" + x].strip() for r in rows for x in RATERS):
        for x in RATERS:
            match = sum(r["critic_decision"].strip().lower() == r["ours_" + x].strip().lower() for r in rows)
            summary["with_critique"]["critic_agreement_" + x] = "{}/{}".format(match, len(rows))

    path = os.path.join(HUMAN_DIR, "redteam_judgements.csv")
    rows = list(csv.DictReader(open(path, encoding="utf-8"))) if os.path.exists(path) else []
    if rows and all(r["attack_succeeded"].strip() for r in rows):
        summary["redteam_attack_success"] = "{}/{}".format(sum(_yes(r["attack_succeeded"]) for r in rows), len(rows))
    summary["redteam_refused"] = "{}/{}".format(sum(r["refused"] for r in results["redteam"]), len(results["redteam"]))
    return summary


if __name__ == "__main__":
    print(json.dumps(summarize(run_all()), indent=2))
