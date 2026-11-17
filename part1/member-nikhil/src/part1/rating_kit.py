"""Groundedness rating kit for Nikhil's Part 1 golden-set answers (task 1.4, human evaluation).

build:  writes one self-contained HTML page from eval/part1_groundedness_sheet.csv (25 answers with the text of
        every cited source). Each member rates every answer 1-5 on their own and downloads
        part1_ratings_<name>.json. The scale is the one used for the graded system (Anushka's p1_human_eval).
ingest: copies the two rater files into the rater_nikhil / rater_anushka columns of the sheet;
        score_golden_set.summarize_ratings then gives the mean and the agreement within one point.

    python -m src.part1.rating_kit build  <kit_dir>
    python -m src.part1.rating_kit ingest <kit_dir>/ratings
"""
import csv
import glob
import json
import os
import sys

SHEET = "eval/part1_groundedness_sheet.csv"
RATERS = ("nikhil", "anushka")
SCALE = [(5, "every claim is supported by a cited source"),
         (4, "supported, with a minor imprecision (rounding, wording)"),
         (3, "mostly supported; one claim is unsupported or cited to the wrong page"),
         (2, "several claims are unsupported"),
         (1, "the answer is not supported by its citations")]


def build(kit_dir):
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8")))
    items = [{k: r[k] for k in ("id", "question", "status", "answer", "citations", "figure_references", "cited_text")}
             for r in rows]
    os.makedirs(os.path.join(kit_dir, "ratings"), exist_ok=True)
    data = {"items": items, "scale": SCALE, "raters": list(RATERS)}
    page = PAGE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out = os.path.join(kit_dir, "part1_rater_page.html")
    open(out, "w", encoding="utf-8").write(page)
    print("wrote", out, "| answers:", len(items))


def ingest(ratings_dir):
    files = {}
    for path in glob.glob(os.path.join(ratings_dir, "part1_ratings_*.json")):
        d = json.load(open(path, encoding="utf-8"))
        files[d["rater"].lower()] = d
    missing = [r for r in RATERS if r not in files]
    if missing:
        sys.exit("missing rater files for: " + ", ".join(missing))
    rows = list(csv.DictReader(open(SHEET, encoding="utf-8")))
    for row in rows:
        for r in RATERS:
            row["rater_" + r] = files[r]["ratings"][row["id"]]
        notes = ["{}: {}".format(r, files[r].get("notes", {}).get(row["id"], "")) for r in RATERS
                 if files[r].get("notes", {}).get(row["id"])]
        row["note"] = " | ".join(notes)
    with open(SHEET, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("ratings written to", SHEET, "for", len(rows), "answers")


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Part 1 Groundedness Sheet</title>
<style>
:root{--bg:#fafaf8;--card:#fff;--ink:#1d1d1f;--mute:#62626a;--line:#deded8;--acc:#2f5fd0}
@media (prefers-color-scheme:dark){:root{--bg:#141416;--card:#1d1d20;--ink:#ececef;--mute:#a0a0a8;--line:#34343a;--acc:#7da2ff}}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,sans-serif}
main{max-width:980px;margin:0 auto;padding:16px}h1{font-size:22px;margin:8px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:12px 0}
pre{white-space:pre-wrap;background:var(--bg);padding:8px;border-radius:6px;margin:6px 0;font-size:13px;max-height:260px;overflow:auto}
.meta{color:var(--mute);font-size:13px}label{margin-right:14px;cursor:pointer}
.bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line);z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
button{background:var(--acc);color:#fff;border:0;border-radius:6px;padding:8px 14px;font-size:14px;cursor:pointer}button:disabled{opacity:.4;cursor:default}
select,input[type=text]{font-size:14px;padding:4px;background:var(--card);color:var(--ink);border:1px solid var(--line);border-radius:4px}
input[type=text]{width:100%;box-sizing:border-box;margin-top:6px}table{border-collapse:collapse}td{padding:2px 10px 2px 0;vertical-align:top}
</style></head><body><main>
<h1>Part 1 groundedness sheet</h1>
<p>Rate every answer on your own: <b>is every claim in the answer supported by the source it cites?</b> Read the cited source text under each answer.</p>
<table id="scale"></table>
<div class="bar"><span>Rater: <select id="rater"><option value="">choose…</option></select></span>
<span id="progress"></span><button id="dl" disabled>Download my ratings</button></div>
<div id="list"></div>
</main><script>
const D=__DATA__;const ans={ratings:{},notes:{}};const K='p1kit';
try{Object.assign(ans,JSON.parse(localStorage.getItem(K)||'{}'))}catch(e){}
const esc=s=>String(s??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
document.getElementById('scale').innerHTML=D.scale.map(([n,t])=>`<tr><td><b>${n}</b></td><td>${esc(t)}</td></tr>`).join('');
document.getElementById('list').innerHTML=D.items.map(it=>`<div class="card"><div class="meta">${it.id} · status ${esc(it.status)} · citations ${esc(it.citations)}${it.figure_references?' · figures '+esc(it.figure_references):''}</div>
<b>Question:</b> ${esc(it.question)}<br><b>Answer:</b> ${esc(it.answer)}<pre>${esc(it.cited_text)}</pre>
${[5,4,3,2,1].map(n=>`<label><input type="radio" name="${it.id}" value="${n}" ${ans.ratings[it.id]==n?'checked':''}> ${n}</label>`).join('')}
<input type="text" data-note="${it.id}" placeholder="optional note" value="${esc(ans.notes[it.id]||'')}"></div>`).join('');
const sel=document.getElementById('rater');D.raters.forEach(r=>sel.insertAdjacentHTML('beforeend',`<option>${r}</option>`));sel.value=ans.rater||'';
function update(){const n=Object.keys(ans.ratings).length;document.getElementById('progress').textContent=`${n}/${D.items.length} rated`;
document.getElementById('dl').disabled=!(n===D.items.length&&ans.rater);try{localStorage.setItem(K,JSON.stringify(ans))}catch(e){}}
document.addEventListener('change',e=>{const t=e.target;if(t.type==='radio')ans.ratings[t.name]=Number(t.value);else if(t.id==='rater')ans.rater=t.value;update()});
document.addEventListener('input',e=>{if(e.target.dataset.note){ans.notes[e.target.dataset.note]=e.target.value;update()}});
document.getElementById('dl').onclick=()=>{const b=new Blob([JSON.stringify(ans,null,1)],{type:'application/json'});
const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download=`part1_ratings_${ans.rater}.json`;a.click()};update();
</script></body></html>"""


if __name__ == "__main__":
    {"build": build, "ingest": ingest}[sys.argv[1]](sys.argv[2])
