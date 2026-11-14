"""Builds notebooks/p3_human_eval.ipynb: human-evaluation and critic-agreement numbers (no GPU needed).

Inputs: outputs/results.json (from p3_agents.ipynb) and human_eval/*.csv (filled in by both members).
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# DATA 266 Lab 2 - Part 3: human evaluation and critic agreement

Both members judged every scenario output (pass/fail, using the rubric) and every critic decision (accept/revise)
**independently**, with a review page that hid the automatic scores and the critic's decisions. The raw answers were
committed before any discussion. Disagreements were then discussed, and the agreed outcome is in `resolved_pass` /
`agreed_decision`. This notebook only reads those files and `outputs/results.json` from `p3_agents.ipynb`. It runs on CPU."""),
code("""# Cell 1 - run information (required)
import os, sys, subprocess, random, numpy as np, torch
if not os.path.isdir("p3") and os.path.isdir("../p3"):
    os.chdir("..")
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not commit and os.path.exists("COMMIT"):
    commit = open("COMMIT").read().strip()
import pandas as pd
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__, "pandas": pd.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})"""),
md("## 1. Inputs"),
code("""import json, csv
from p3 import evaluate as ev
res = json.load(open("outputs/results.json"))
sj = pd.read_csv(ev.HUMAN_DIR / "scenario_judgements.csv", dtype=str, keep_default_na=False)
cl = pd.read_csv(ev.HUMAN_DIR / "critic_labels.csv", dtype=str, keep_default_na=False)
print("evaluated run: VLM", res["config"]["vlm_id"], "| device", res["config"]["device"], "| LoRA", res["config"]["lora_id"])
print(len(sj), "scenario judgements per member;", len(cl), "critic labels per member")
assert (sj.member1_pass != "").all() and (sj.member2_pass != "").all(), "a member left a scenario unanswered"
assert (cl.member1_decision != "").all() and (cl.member2_decision != "").all(), "a member left a critic label unanswered\""""),
md("## 2. Scenario judgements: agreement between members and how disagreements were resolved"),
code("""agree = sj.member1_pass == sj.member2_pass
print(f"members agreed on {agree.sum()}/{len(sj)} scenario judgements = {agree.mean():.1%}")
for cond in ("with_critique", "no_critique"):
    m = sj.condition == cond
    print(f"  {cond:<14} {agree[m].sum()}/{m.sum()} = {agree[m].mean():.1%}")
dis = sj[~agree][["scenario_id", "condition", "member1_pass", "member2_pass", "resolved_pass", "resolution_note"]]
display(dis.rename(columns={"member1_pass": "Anushka", "member2_pass": "Nikhil"}))
assert (dis.resolved_pass != "").all(), "some disagreements are not resolved yet\""""),
md("## 3. Human pass rate (final agreed judgement) vs the automatic rubric checks"),
code("""sj["final"] = np.where(sj.resolved_pass != "", sj.resolved_pass, sj.member1_pass)
auto = {(r["id"], c): r["passed"] for k, c in (("scenarios_with_critique", "with_critique"), ("scenarios_no_critique", "no_critique"))
        for r in res[k]}
sj["auto"] = [("pass" if auto[(s, c)] else "fail") for s, c in zip(sj.scenario_id, sj.condition)]
for cond in ("no_critique", "with_critique"):
    m = sj.condition == cond
    print(f"{cond:<14} human pass {(sj.final[m] == 'pass').sum()}/{m.sum()} | automatic rubric pass {(sj.auto[m] == 'pass').sum()}/{m.sum()}")
mismatch = sj[sj.final != sj.auto][["scenario_id", "condition", "final", "auto"]]
print(f"human and automatic checks agree on {(sj.final == sj.auto).sum()}/{len(sj)} judgements; differences:")
display(mismatch.rename(columns={"final": "human (agreed)", "auto": "automatic rubric"}))"""),
md("## 4. Critic agreement: does the critic's accept/revise match ours?"),
code("""cl_agree = cl.member1_decision == cl.member2_decision
print(f"members agreed on {cl_agree.sum()}/{len(cl)} critic labels = {cl_agree.mean():.1%}")
display(cl[~cl_agree][["scenario_id", "attempt", "critic_decision", "member1_decision", "member2_decision", "agreed_decision"]]
        .rename(columns={"member1_decision": "Anushka", "member2_decision": "Nikhil"}))
assert (cl[~cl_agree].agreed_decision != "").all(), "some critic disagreements are not resolved yet"
cl["ref"] = np.where(cl.agreed_decision != "", cl.agreed_decision, cl.member1_decision)
match = cl.critic_decision == cl.ref
print(f"critic agreement (critic vs our agreed label): {match.sum()}/{len(cl)} = {match.mean():.1%}")
display(cl[~match][["scenario_id", "attempt", "critic_decision", "ref"]].rename(columns={"ref": "our label"}))"""),
md("## 5. Final Part 3 results table (automatic + human measurements)"),
code("""rows_crit, rows_base = res["scenarios_with_critique"], res["scenarios_no_critique"]
table, ha, ca = ev.results_table(rows_base, rows_crit, res["redteam_baseline"], res["redteam_final"])
table.insert(3, "member agreement (scenarios)", [f"{agree.mean():.1%} ({agree.sum()}/{len(sj)})"] * 2)
display(table)
print("helper check:", ha, ca)
open("outputs/results_table_final.md", "w").write(table.to_markdown())
shared = "../eval/part3_eval_anushka.json"
if os.path.exists(shared):
    s = json.load(open(shared))
    s["results_table"] = json.loads(table.to_json(orient="index"))
    s["human_agreement"], s["critic_agreement"] = ha, ca
    s["member_agreement_scenarios"] = float(agree.mean()); s["member_agreement_critic_labels"] = float(cl_agree.mean())
    json.dump(s, open(shared, "w"), indent=1, default=str)
    print("updated", shared)
print("saved outputs/results_table_final.md")"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "p3_human_eval.ipynb"
nbf.write(nb, out)
print("wrote", out)
