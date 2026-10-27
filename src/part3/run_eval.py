import json
import os
import statistics
import time

from PIL import Image

from src.part3.orchestrator import run_request, save_trace

SCENARIOS_PATH = "eval/part3_scenarios.json"
REDTEAM_PATH = "eval/part3_redteam.json"
IMAGES_DIR = "data/raw/part3"
TRACE_DIR = "eval/part3_traces"


def load_image_if_available(file_name):
    if not file_name:
        return None
    path = os.path.join(IMAGES_DIR, file_name)
    if not os.path.exists(path):
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        # Broken/empty/truncated file: treat like a missing image so the orchestrator's
        # existing "missing_or_broken_input_image" refusal path handles it, rather than
        # crashing the whole evaluation batch (this is exactly what these test cases check).
        return None


def run_scenarios():
    with open(SCENARIOS_PATH, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    os.makedirs(TRACE_DIR, exist_ok=True)
    results = []
    durations = []
    for scenario in scenarios:
        image = load_image_if_available(scenario["input_image"])
        start = time.time()
        state = run_request(scenario["request"], image)
        duration = time.time() - start
        durations.append(duration)

        trace_path = os.path.join(TRACE_DIR, "{}.json".format(scenario["id"]))
        save_trace(state, trace_path)

        results.append({
            "id": scenario["id"],
            "category": scenario["category"],
            "refused": state.refused,
            "route": state.route,
            "critique_decision": state.critique_decision,
            "duration_seconds": duration,
            # pass/fail against the rubric is a human judgement call, recorded separately
            "auto_pass": None,
        })

    median_seconds = statistics.median(durations) if durations else None
    return results, median_seconds


def run_redteam():
    with open(REDTEAM_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)

    os.makedirs(TRACE_DIR, exist_ok=True)
    results = []
    for case in cases:
        image = load_image_if_available(case["input_image"])
        state = run_request(case["request"], image)
        trace_path = os.path.join(TRACE_DIR, "redteam_{}.json".format(case["id"]))
        save_trace(state, trace_path)
        results.append({
            "id": case["id"],
            "category": case["category"],
            "refused": state.refused,
            "refusal_reason": state.refusal_reason,
            # an attack "succeeded" if the system was NOT refused and NOT safely handled;
            # judged against the "expected" field by a human reviewer
            "attack_succeeded": None,
        })
    return results


if __name__ == "__main__":
    scenario_results, median_seconds = run_scenarios()
    redteam_results = run_redteam()

    report = {
        "scenario_results": scenario_results,
        "median_seconds_per_request": median_seconds,
        "redteam_results": redteam_results,
    }
    with open("eval/part3_eval_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
