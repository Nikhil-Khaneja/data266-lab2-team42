"""Evaluation for Part 3: 15 scenarios (with / without critique), 10 red-team cases,
critic agreement and human-evaluation agreement (from CSV files filled in by the team)."""
from __future__ import annotations

import csv
import json
import os
import statistics
from pathlib import Path
from typing import Optional

from PIL import Image

from .graph import Orchestrator
from .models import fit_resolution
from .state import AgentState

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = ROOT / "data" / "images"


def load_json(name: str) -> dict:
    return json.loads((ROOT / "eval" / name).read_text(encoding="utf-8"))


def _img_path(name: Optional[str]) -> Optional[str]:
    return None if name is None else str(IMG_DIR / name)


def check_rubric(s: AgentState, sc: dict, orch: Orchestrator) -> dict:
    ck = sc["checks"]
    res: dict = {}
    res["status"] = s.status == ck["status"]
    if "intent" in ck:
        res["intent"] = bool(s.planner) and s.planner.intent == ck["intent"]
    if ck.get("no_image"):
        res["no_image"] = s.final_image_path is None and not s.generations
    if ck.get("has_image"):
        res["has_image"] = s.final_image_path is not None
    if "mode" in ck:
        res["mode"] = bool(s.generations) and s.generations[-1].mode == ck["mode"]
    if ck.get("uses_lora"):
        res["uses_lora"] = bool(s.generations) and all(g.lora for g in s.generations)
    if "min_qa" in ck:
        res["min_qa"] = bool(s.vision) and sum(1 for q in s.vision.qa if q["answer"].strip()) >= ck["min_qa"]
    for field, words in ck.get("keywords_any", {}).items():
        text = ""
        for part in field.split("+"):
            if part == "message":
                text += " " + s.message
            elif s.vision is not None:
                text += " " + str(getattr(s.vision, part, ""))
        res[f"keywords:{field}"] = any(w.lower() in text.lower() for w in words)
    if "message_any" in ck:
        res["message"] = any(w.lower() in s.message.lower() for w in ck["message_any"])
    if ck.get("no_internal_error"):
        res["no_internal_error"] = not any("error" in t.output for t in s.trace)
    metrics = {}
    if s.final_image_path and "target_text" in ck:
        out = Image.open(s.final_image_path).convert("RGB")
        metrics["target_clip"] = round(orch.hub.clip.text_image(out, ck["target_text"]), 4)
        res["target_clip"] = metrics["target_clip"] >= ck["min_target_clip"]
        if "min_preservation" in ck and s.image is not None:
            ref = fit_resolution(s.image, orch.cfg.resolution)
            metrics["preservation"] = round(orch.hub.clip.image_image(out, ref), 4)
            res["preservation"] = metrics["preservation"] >= ck["min_preservation"]
    elif "target_text" in ck:
        res["target_clip"] = False
    return {"passed": all(res.values()), "checks": res, "metrics": metrics}


def run_scenarios(orch: Orchestrator, use_critique: bool, ids: Optional[list[str]] = None) -> list[dict]:
    rows = []
    for sc in load_json("scenarios.json")["scenarios"]:
        if ids and sc["id"] not in ids:
            continue
        tag = f"{sc['id']}_{'crit' if use_critique else 'nocrit'}"
        s = orch.run(sc["request"], _img_path(sc["image"]), use_critique=use_critique, tag=tag)
        r = check_rubric(s, sc, orch)
        rows.append({"id": sc["id"], "category": sc["category"], "condition": "with_critique" if use_critique else "no_critique",
                     "request": sc["request"], "image": sc["image"], "status": s.status,
                     "intent": s.planner.intent if s.planner else None, "revisions": s.revisions,
                     "attempts": len(s.generations), "passed": r["passed"], "checks": r["checks"], **r["metrics"],
                     "seconds": s.total_seconds, "message": s.message, "final_image": s.final_image_path,
                     "request_id": s.request_id,
                     "critic_decisions": [c.decision for c in s.critiques]})
        print(f"{sc['id']} [{rows[-1]['condition']}] {s.status:<20} pass={r['passed']}  {s.total_seconds:6.1f}s  "
              f"{ {k: v for k, v in r['checks'].items() if not v} or ''}")
    return rows


def run_redteam(orch: Orchestrator, use_critique: bool = True) -> list[dict]:
    rows = []
    for rc in load_json("redteam.json")["cases"]:
        s = orch.run(rc["request"], _img_path(rc["image"]), use_critique=use_critique, tag=rc["id"])
        failures = []
        if s.status not in rc["expect_status"]:
            failures.append(f"status={s.status}")
        if rc.get("expect_no_image") and (s.generations or s.final_image_path):
            failures.append("image generated")
        if rc.get("no_internal_error") and any("error" in t.output for t in s.trace):
            failures.append("internal error")
        rows.append({"id": rc["id"], "category": rc["category"], "request": rc["request"], "image": rc["image"],
                     "status": s.status, "attack_success": bool(failures), "why_failed": "; ".join(failures),
                     "safety_flags": s.planner.safety_flags if s.planner else [], "message": s.message,
                     "seconds": s.total_seconds, "request_id": s.request_id})
        print(f"{rc['id']} {s.status:<12} attack_success={bool(failures)}  {s.message[:110]}")
    return rows


# ---------------------------------------------------------------------------
# human evaluation (filled in by both team members; never generated by code)
# ---------------------------------------------------------------------------
HUMAN_DIR = Path(os.environ.get("P3_HUMAN_DIR", ROOT / "human_eval"))


def write_templates(scen_rows: list[dict], out_dir: str) -> None:
    HUMAN_DIR.mkdir(exist_ok=True)

    def unfilled(path: Path) -> bool:
        """True if the file is missing or nobody has entered a judgement yet (safe to regenerate)."""
        if not path.exists():
            return True
        rows = list(csv.DictReader(open(path, encoding="utf-8")))
        return not any((v or "").strip() for r in rows for k, v in r.items() if k and k.startswith(("member", "resolved", "agreed")))

    p = HUMAN_DIR / "scenario_judgements.csv"
    if unfilled(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scenario_id", "condition", "request_id", "final_image", "member1_pass", "member2_pass",
                        "resolved_pass", "resolution_note"])
            for r in scen_rows:
                w.writerow([r["id"], r["condition"], r["request_id"], r["final_image"] or "", "", "", "", ""])
    p = HUMAN_DIR / "critic_labels.csv"
    if unfilled(p):
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["scenario_id", "request_id", "attempt", "image_path", "critic_decision", "member1_decision",
                        "member2_decision", "agreed_decision"])
            for r in scen_rows:
                if r["condition"] != "with_critique":
                    continue
                tp = Path(out_dir) / "traces" / f"{r['request_id']}.json"
                if not tp.exists():
                    continue
                tr = json.loads(tp.read_text(encoding="utf-8"))
                for c, g in zip(tr["critiques"], tr["generations"]):
                    w.writerow([r["id"], r["request_id"], c["attempt"], g["image_path"] or "",
                                "accept" if c["decision"] == "accept" else "revise", "", "", ""])


def _yes(v: str) -> Optional[bool]:
    v = (v or "").strip().lower()
    if v in {"1", "pass", "p", "yes", "y", "true"}:
        return True
    if v in {"0", "fail", "f", "no", "n", "false"}:
        return False
    return None


def human_agreement() -> Optional[dict]:
    p = HUMAN_DIR / "scenario_judgements.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    pairs = [(_yes(r["member1_pass"]), _yes(r["member2_pass"]), r) for r in rows]
    done = [(a, b, r) for a, b, r in pairs if a is not None and b is not None]
    if not done:
        return None
    out = {"n_judged": len(done), "agreement": sum(a == b for a, b, _ in done) / len(done),
           "disagreements": [r["scenario_id"] + "/" + r["condition"] for a, b, r in done if a != b]}
    for cond in ("with_critique", "no_critique"):
        res = [_yes(r["resolved_pass"]) if _yes(r["resolved_pass"]) is not None else (a if a == b else None)
               for a, b, r in done if r["condition"] == cond]
        res = [x for x in res if x is not None]
        out[f"human_pass_{cond}"] = f"{sum(res)}/{len(res)}" if res else None
    return out


def critic_agreement() -> Optional[dict]:
    p = HUMAN_DIR / "critic_labels.csv"
    if not p.exists():
        return None
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    labels = []
    for r in rows:
        ref = (r["agreed_decision"] or "").strip().lower() or \
              ((r["member1_decision"] or "").strip().lower() if (r["member1_decision"] or "").strip().lower() ==
               (r["member2_decision"] or "").strip().lower() else "")
        if ref in {"accept", "revise"}:
            labels.append(ref == r["critic_decision"].strip().lower())
    if not labels:
        return None
    return {"n_labelled": len(labels), "critic_agreement": sum(labels) / len(labels)}


def results_table(no_crit: list[dict], crit: list[dict], red_base: list[dict], red_final: list[dict]):
    import pandas as pd

    def med(rows):
        return round(statistics.median(r["seconds"] for r in rows), 1) if rows else None

    gen = lambda rows: [r for r in rows if r["category"] not in ("caption_question", "ambiguous_broken")]
    ha = human_agreement()
    ca = critic_agreement()
    table = pd.DataFrame([
        {"system": "baseline (no critique loop)",
         "task success (/15)": f"{sum(r['passed'] for r in no_crit)}/{len(no_crit)}",
         "critic agreement": "n/a (no critic)",
         "red-team attack success (/10)": f"{sum(r['attack_success'] for r in red_base)}/{len(red_base)}",
         "human pass (resolved)": (ha or {}).get("human_pass_no_critique") or "pending (fill human_eval/*.csv)",
         "median s / request": med(no_crit),
         "median s / generation request": med(gen(no_crit))},
        {"system": "final (critique loop, max 2 revisions)",
         "task success (/15)": f"{sum(r['passed'] for r in crit)}/{len(crit)}",
         "critic agreement": f"{ca['critic_agreement']:.2f} (n={ca['n_labelled']})" if ca else "pending (fill human_eval/critic_labels.csv)",
         "red-team attack success (/10)": f"{sum(r['attack_success'] for r in red_final)}/{len(red_final)}",
         "human pass (resolved)": (ha or {}).get("human_pass_with_critique") or "pending (fill human_eval/*.csv)",
         "median s / request": med(crit),
         "median s / generation request": med(gen(crit))},
    ]).set_index("system")
    return table, ha, ca
