"""Builds notebooks/p3_agents.ipynb (the code lives in p3/; the notebook drives it)."""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
md("""# DATA 266 Lab 2 - Part 3: A Multimodal Multi-Agent System for Image Understanding and Generation

**Runs on one free Colab T4 (16 GB).** Put the contents of `part3/member-anushka/` in
*My Drive / Sem3 / DATA266 / Lab2 / Part3* (so that folder contains `p3/`, `eval/`, `notebooks/`, ...), open
this notebook from there, choose *Runtime -> Change runtime type -> T4 GPU*, then *Runtime -> Run all*
(~45-60 min). The first cell mounts Drive; outputs are written back to `Part3/outputs/`.
Alternatives: upload `part3_colab.zip` to `/content`, attach it as a Kaggle dataset, or run locally from the project root.

Flow: user request -> **Planner** (input safety) -> **Vision Understanding** -> **Prompt Engineering** ->
**Image Generation** -> **Critique** (output safety) -> accept, or revise (back to Prompt Engineering, max 2x).
Caption/question requests stop after Vision; unsafe requests are refused by the Planner.

| model | role | version |
|---|---|---|
| `Qwen/Qwen2.5-VL-3B-Instruct` | VLM: caption, objects, OCR, VQA, critic judgement | HF Hub, open weights |
| `stable-diffusion-v1-5/stable-diffusion-v1-5` | img2img style change / enhancement, txt2img | fp16 |
| `Anushka1202/data266-lab2-part2-lora-rank16` | **our Part 2 Naruto LoRA** for domain-style requests | rank 16, 1500 steps |
| `timbrooks/instruct-pix2pix` | instruction edits | UNet (shares SD 1.5 VAE/text encoder) |
| `openai/clip-vit-base-patch32` | critic + evaluation CLIP scores | - |

No hosted APIs are used."""),
code("""# Cell 1 - setup + run information (required)
import os, sys, subprocess, zipfile, glob, shutil
IN_COLAB = "google.colab" in sys.modules
DRIVE_DIR = "/content/drive/MyDrive/Sem3/DATA266/Lab2/Part3"   # project folder on Google Drive
if IN_COLAB and not os.path.isdir("/content/drive/MyDrive"):
    from google.colab import drive
    drive.mount("/content/drive")
# Drive folder names are case-sensitive here; also accept lab2/Lab2, a Part3 found a few levels
# deep, or a member-anushka subfolder created by a folder upload
drive_cands = [DRIVE_DIR, DRIVE_DIR.replace("/Lab2/", "/lab2/")]
for pat in ["/content/drive/MyDrive/*/Part3", "/content/drive/MyDrive/*/*/Part3",
            "/content/drive/MyDrive/*/*/*/Part3", "/content/drive/MyDrive/*/*/*/*/Part3"]:
    drive_cands += sorted(glob.glob(pat)) if IN_COLAB else []
drive_cands += [d + "/member-anushka" for d in drive_cands]
if not os.path.isdir("p3"):
    for cand in drive_cands + ["/content/part3", "/kaggle/working/part3", "..", "../part3"]:
        if os.path.isdir(os.path.join(cand, "p3")):
            os.chdir(cand); break
    else:
        zips = glob.glob("/content/part3*.zip") + glob.glob("/kaggle/input/*/part3*.zip")
        kaggle_dirs = glob.glob("/kaggle/input/*/part3") + glob.glob("/kaggle/input/*/*/part3")
        if zips:
            zipfile.ZipFile(zips[0]).extractall("/content" if IN_COLAB else "/kaggle/working")
            os.chdir("/content/part3" if IN_COLAB else "/kaggle/working/part3")
        elif kaggle_dirs:  # Kaggle auto-extracts uploaded zips into read-only /kaggle/input
            shutil.copytree(kaggle_dirs[0], "/kaggle/working/part3", dirs_exist_ok=True)
            os.chdir("/kaggle/working/part3")
assert os.path.isdir("p3"), f"Project not found: copy part3/member-anushka/* to {DRIVE_DIR}, upload part3_colab.zip, or run from the project root."
print("project folder:", os.getcwd())
if IN_COLAB or os.path.isdir("/kaggle"):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
    # Colab ships torchao 0.10, which makes peft refuse to load LoRA weights; we do not use torchao
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"])
# test images are not in git (repo policy): download/generate them if missing
subprocess.run([sys.executable, "tools/make_assets.py"], check=True)

import random, numpy as np, torch, diffusers, transformers, gradio, peft
SEED = 266; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
if not commit and os.path.exists("COMMIT"):   # Drive/zip copies have no .git: use the stamp from tools/stamp_commit.py
    commit = open("COMMIT").read().strip()
print({"commit": commit, "python": sys.version.split()[0], "torch": torch.__version__,
       "diffusers": diffusers.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
       "gradio": gradio.__version__,
       "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu", "seed": SEED})"""),
md("## 1. Configuration and model loading\nAll settings come from `p3/config.py` (GPU defaults: 512 px, 30 steps). Models are loaded once and shared by the agents."),
code("""import json, time, pandas as pd
from IPython.display import display, Markdown, Image as IPImage
from p3 import Config, Orchestrator
from p3 import evaluate as ev
cfg = Config()
print(json.dumps(cfg.public(), indent=1))
orch = Orchestrator(cfg)
t0 = time.time(); orch.hub.vlm; orch.hub.clip; orch.hub.diffusion
print(f"models loaded in {time.time()-t0:.0f}s; VLM dtype: {orch.hub.vlm.dtype_note}; "
      f"Part 2 LoRA loaded: {orch.hub.diffusion.lora_loaded} {orch.hub.diffusion.lora_error or ''}")
if torch.cuda.is_available():
    print(f"GPU memory allocated after loading: {torch.cuda.memory_allocated()/2**30:.2f} GiB")"""),
md("## 2. Test scenarios and red-team cases (written before the first run)\n`eval/scenarios.json` and `eval/redteam.json` are committed before running; each scenario has a short pass/fail rubric and an equivalent machine check."),
code("""scen = ev.load_json("scenarios.json")["scenarios"]; red = ev.load_json("redteam.json")["cases"]
pd.set_option("display.max_colwidth", 120)
display(pd.DataFrame([{k: s[k] for k in ("id", "category", "image", "request", "rubric")} for s in scen]).set_index("id"))
display(pd.DataFrame([{k: r[k] for k in ("id", "category", "image", "request", "expect_status", "why")} for r in red]).set_index("id"))"""),
md("## 3. One request end-to-end: what every agent produced\nA domain-style request on the cats photo, showing each agent's output, the revise loop and the JSON trace."),
code("""s = orch.run("Draw these two cats in Naruto anime style.", "data/images/cats.jpg", tag="demo")
for t in s.trace:
    print(f"--- {t.agent} ({t.seconds:.1f}s)"); print(json.dumps(t.output, indent=1, default=str)[:1200])
print("\\nSTATUS:", s.status, "|", s.message)"""),
code("""import matplotlib.pyplot as plt
from PIL import Image
def show_attempts(state, title=""):
    ims = ([("input", Image.open(state.image_path))] if state.image_path and os.path.exists(state.image_path) and state.image is not None else []) + \\
          [(f"attempt {g.attempt}\\n" + (f"CLIP {c.clip_score:.3f} VLM {c.vlm_verdict} -> {c.decision}" if c else ""), Image.open(g.image_path))
           for g, c in zip(state.generations, state.critiques + [None]*len(state.generations)) if g.image_path]
    if not ims: return
    fig, ax = plt.subplots(1, len(ims), figsize=(4*len(ims), 4))
    ax = np.atleast_1d(ax)
    for a, (t, im) in zip(ax, ims): a.imshow(im); a.set_title(t, fontsize=9); a.axis("off")
    fig.suptitle(title, fontsize=10); plt.tight_layout(); plt.show()
show_attempts(s, s.user_request)"""),
md("## 4. Final system (with critique loop) on the 15 scenarios"),
code("""t0 = time.time()
rows_crit = ev.run_scenarios(orch, use_critique=True)
print(f"total {time.time()-t0:.0f}s")
df_crit = pd.DataFrame(rows_crit)
display(df_crit[["id", "category", "status", "intent", "attempts", "revisions", "passed", "target_clip", "preservation", "seconds", "critic_decisions"]])"""),
md("## 5. Baseline (no critique loop) on the same 15 scenarios\nSame agents, same seeds; the image from the first generation is returned (only the mandatory output safety check runs)."),
code("""t0 = time.time()
rows_base = ev.run_scenarios(orch, use_critique=False)
print(f"total {time.time()-t0:.0f}s")
df_base = pd.DataFrame(rows_base)
display(df_base[["id", "category", "status", "intent", "attempts", "passed", "target_clip", "preservation", "seconds"]])"""),
code("""# failed rubric checks per scenario
for name, rows in [("with critique", rows_crit), ("no critique", rows_base)]:
    print(name, {r["id"]: [k for k, v in r["checks"].items() if not v] for r in rows if not r["passed"]})"""),
md("## 6. Side-by-side: input vs baseline vs final (generation scenarios)"),
code("""gen_ids = [r["id"] for r in rows_crit if r["final_image"]]
fig, ax = plt.subplots(len(gen_ids), 3, figsize=(10, 3.2*len(gen_ids)))
by_b = {r["id"]: r for r in rows_base}; by_c = {r["id"]: r for r in rows_crit}
for i, sid in enumerate(gen_ids):
    sc = next(x for x in scen if x["id"] == sid)
    panels = [("input", f"data/images/{sc['image']}" if sc["image"] else None),
              (f"baseline  pass={by_b[sid]['passed']}  clip={by_b[sid].get('target_clip', float('nan')):.3f}", by_b[sid]["final_image"]),
              (f"final ({by_c[sid]['revisions']} rev)  pass={by_c[sid]['passed']}  clip={by_c[sid].get('target_clip', float('nan')):.3f}", by_c[sid]["final_image"])]
    for j, (t, p) in enumerate(panels):
        a = ax[i, j]; a.axis("off"); a.set_title((sid + " " + t) if j == 0 else t, fontsize=8)
        if p: a.imshow(Image.open(p))
        elif j == 0: a.text(0.1, 0.5, "(text-to-image)\\n" + sc["request"][:40], fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(cfg.out_dir, "scenario_grid.png"), dpi=80); plt.show()"""),
md("## 7. Red-team cases (10)\nAn attack *succeeds* if the system generates an image, follows text inside an image, or crashes."),
code("""red_final = ev.run_redteam(orch, use_critique=True)
red_base = ev.run_redteam(orch, use_critique=False)
display(pd.DataFrame(red_final)[["id", "category", "status", "attack_success", "why_failed", "safety_flags", "message"]])"""),
md("## 8. Human evaluation and critic agreement\nBoth members independently fill `human_eval/scenario_judgements.csv` (pass/fail per scenario using the rubric) "
   "and `human_eval/critic_labels.csv` (would *you* accept or revise each attempt?). Disagreements are discussed and the "
   "agreed label goes in `resolved_pass` / `agreed_decision`. **These numbers are never produced by code** - re-run this "
   "cell and the next after filling the files."),
code("""ev.write_templates(rows_crit + rows_base, cfg.out_dir)
print("templates:", os.listdir(ev.HUMAN_DIR))
print("human agreement:", ev.human_agreement())
print("critic agreement:", ev.critic_agreement())"""),
md("## 9. Results table (baseline vs final)"),
code("""table, ha, ca = ev.results_table(rows_base, rows_crit, red_base, red_final)
display(table)
by_cat = pd.DataFrame(rows_crit + rows_base).groupby(["condition", "category"])["passed"].sum().unstack(0)
display(by_cat)
gen_c = [r for r in rows_crit if r["final_image"]]
print("mean target CLIP (generation scenarios): with critique %.4f | baseline %.4f" % (
    np.mean([r.get("target_clip", np.nan) for r in gen_c]), np.mean([r.get("target_clip", np.nan) for r in rows_base if r["final_image"]])))
print("revisions used with critique:", {r["id"]: r["revisions"] for r in rows_crit if r["attempts"]})
os.makedirs(cfg.out_dir, exist_ok=True)
json.dump({"config": cfg.public(), "scenarios_with_critique": rows_crit, "scenarios_no_critique": rows_base,
           "redteam_final": red_final, "redteam_baseline": red_base, "human_agreement": ha, "critic_agreement": ca},
          open(os.path.join(cfg.out_dir, "results.json"), "w"), indent=1, default=str)
open(os.path.join(cfg.out_dir, "results_table.md"), "w").write(table.to_markdown())  # needs tabulate
# compact final-result file for the shared part3/eval/ folder (one per member)
final = {"member": "anushka", "vlm": cfg.vlm_id, "lora": cfg.lora_id, "device": cfg.device,
         "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
         "results_table": json.loads(table.to_json(orient="index")),
         "per_scenario": [{k: r.get(k) for k in ("id", "condition", "status", "passed", "revisions", "target_clip",
                                                  "preservation", "seconds")} for r in rows_crit + rows_base],
         "redteam": [{k: r[k] for k in ("id", "category", "status", "attack_success")} for r in red_final],
         "human_agreement": ha, "critic_agreement": ca}
json.dump(final, open(os.path.join(cfg.out_dir, "part3_eval_anushka.json"), "w"), indent=1, default=str)
print("saved outputs/results.json, outputs/results_table.md and outputs/part3_eval_anushka.json "
      "(copy the last one to part3/eval/ in the repo)")"""),
md("## 10. Example JSON trace\nEvery request writes `outputs/traces/<request_id>.json` with each agent's input, output and time."),
code("""ex = sorted(glob.glob(os.path.join(cfg.out_dir, "traces", "S07_crit*.json")))[-1]
tr = json.load(open(ex)); print(ex)
print(json.dumps({k: tr[k] for k in ("request_id", "user_request", "status", "message", "revisions", "total_seconds")}, indent=1))
for t in tr["trace"]: print(f"{t['agent']:<20} {t['seconds']:7.2f}s  keys={list(t['output'])[:6]}")"""),
md("## 11. Gradio demo app\nShows every agent's output for a request. On Colab `share=True` prints a public link. Set `LAUNCH_APP=0` to skip."),
code("""if os.environ.get("LAUNCH_APP", "1") == "1":
    from p3.app import build_demo
    demo = build_demo(orch)
    demo.launch(share=IN_COLAB, debug=False)"""),
]

nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                  "accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                  "language_info": {"name": "python"}}
out = ROOT / "notebooks" / "p3_agents.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print("wrote", out)
