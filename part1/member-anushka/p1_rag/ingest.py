"""Task 1.1 - ingestion.

Reads kb_manifest.json and the three PDFs and produces "units" (the things we
index) plus a structured fact table parsed from the printed tables:

* text units    - ~140-word chunks of page text, with doc_id + page
* table units   - every numbered Table as an image crop + a clean grid
* figure units  - every numbered Figure as an image crop + label + caption
                  ("Figure B-5" and "Figure B-5 (corrected)" are separate)
* facts         - one row per numeric table cell / stated revision
                  (entity, variable, year, value, forecast?, trust)
"""
import json
import os
import re
from dataclasses import dataclass, field, asdict

import pymupdf

from .vocab import (LABEL_RE, norm_label, find_labels, find_entities, find_variables,
                    find_years, parse_number, RANGE_RE)

BOLD_HINTS = ("bold", "-hv", "heavy", "black", "-bd")


@dataclass
class Unit:
    uid: str
    kind: str                 # "text" | "table" | "figure"
    doc_id: str
    page: int
    trust: str
    text: str                 # what gets embedded / BM25-indexed
    label: str = ""           # "Figure A-3", "Table 1-2", "" for text
    caption: str = ""
    image_path: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def cite(self):
        return f"{self.doc_id}:{self.label}" if self.label else f"{self.doc_id}:p{self.page}"


# --------------------------------------------------------------------------- manifest
def load_manifest(data_dir):
    with open(os.path.join(data_dir, "kb_manifest.json")) as f:
        man = json.load(f)
    docs = []
    for d in man["documents"]:
        low = {}
        for o in d.get("section_trust_overrides", []):
            for p in o.get("pages", []):
                low[int(p)] = (o["trust_level"], o.get("section", ""))
        vint = re.search(r"(19|20)\d\d", d.get("data_vintage", "") or "")
        docs.append({
            "doc_id": d["doc_id"], "file": d["file"], "title": d.get("title", ""),
            "trust": d["trust_level"], "page_overrides": low,
            "vintage_year": int(vint.group(0)) if vint else None,
            "vintage": d.get("data_vintage", ""), "published": d.get("published", ""),
        })
    return man, docs


def page_trust(doc, page):
    if page in doc["page_overrides"]:
        return doc["page_overrides"][page]
    return doc["trust"], ""


# --------------------------------------------------------------------------- helpers
def clean_text(s):
    s = s.replace("­", "")
    s = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", s)        # de-hyphenate line breaks
    s = re.sub(r"[ \t]*\n[ \t]*", " ", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def block_text(b):
    return " ".join(s["text"] for l in b["lines"] for s in l["spans"]).strip()


def is_bold(b):
    s = b["lines"][0]["spans"][0]
    f = s["font"].lower()
    return any(h in f for h in BOLD_HINTS) or bool(s["flags"] & 16)


def find_captions(page):
    """Caption blocks: start with a Figure/Table label and are set in a bold font
    (in-text references such as 'Figure 2-1 plots ...' are in the body font)."""
    caps = []
    blocks = [b for b in page.get_text("dict")["blocks"] if b["type"] == 0]
    for b in blocks:
        txt = block_text(b)
        m = LABEL_RE.match(txt)
        if m and m.start() == 0 and is_bold(b):
            lab = norm_label(m.group(1), m.group(2), m.group(3), m.group(4))
            caps.append({"label": lab, "kind": m.group(1).lower(), "bbox": pymupdf.Rect(b["bbox"]),
                         "text": clean_text(txt)})
    return caps, blocks


def caption_column_text(cap, blocks):
    """For margin captions (label alone in a block) collect the title/source lines below it."""
    out, last = [cap["text"]], cap["bbox"]
    if len(cap["text"].split()) > 3:          # caption already carries its title
        return cap["text"], cap["bbox"]
    rect = pymupdf.Rect(cap["bbox"])
    for b in sorted(blocks, key=lambda b: b["bbox"][1]):
        r = pymupdf.Rect(b["bbox"])
        if r.y0 < last.y1 - 1 or abs(r.x0 - cap["bbox"].x0) > 12:
            continue
        if r.y0 - last.y1 > 25 or LABEL_RE.match(block_text(b)):
            break
        out.append(clean_text(block_text(b)))
        rect |= r
        last = r
        if len(out) >= 6:
            break
    return " ".join(out), rect


def cluster_rects(rects, gap=14):
    clusters = []
    for r in rects:
        r = pymupdf.Rect(r)
        merged = True
        while merged:
            merged = False
            for c in clusters:
                if (pymupdf.Rect(c) + (-gap, -gap, gap, gap)).intersects(r):
                    r |= c
                    clusters.remove(c)
                    merged = True
                    break
        clusters.append(r)
    return clusters


def rect_distance(a, b):
    dx = max(0, max(a.x0, b.x0) - min(a.x1, b.x1))
    dy = max(0, max(a.y0, b.y0) - min(a.y1, b.y1))
    return (dx ** 2 + dy ** 2) ** 0.5


def figure_region(page, cap, cap_rect, used):
    """Locate the graphic belonging to a figure caption: the nearest raster image
    (DOC2/DOC3 put the caption above the image) or the nearest cluster of vector drawings."""
    pr = page.rect
    imgs = []
    for im in page.get_images(full=True):
        try:
            r = page.get_image_bbox(im)
        except Exception:
            continue
        if r.is_empty or r.width < 50:
            continue
        imgs.append(pymupdf.Rect(r))
    imgs = [r for r in imgs if not any(r == u for u in used)]
    if imgs:
        below = [r for r in imgs if r.y0 >= cap["bbox"].y0 - 5]
        cand = below or imgs
        return min(cand, key=lambda r: (r.y0 - cap["bbox"].y1) if r in below else 1e4 + rect_distance(r, cap_rect)), True
    rects = []
    for d in page.get_drawings():
        r = pymupdf.Rect(d["rect"]) & pr
        if r.is_empty or r.width > 0.9 * pr.width or r.height > 0.9 * pr.height:
            continue
        if r.width < 0.5 and r.height < 0.5:
            continue
        rects.append(r)
    clusters = [c for c in cluster_rects(rects) if c.width * c.height > 4000]
    clusters = [c for c in clusters if not any(c.intersects(u) and (c & u).get_area() > 0.5 * c.get_area() for u in used)]
    if not clusters:
        return None, False
    reg = min(clusters, key=lambda c: rect_distance(c, cap_rect))
    # pull in axis ticks / legend words (text, not drawings) sitting next to the graphic
    for _ in range(2):
        for b in page.get_text("dict")["blocks"]:
            if b["type"] != 0:
                continue
            r = pymupdf.Rect(b["bbox"])
            t = block_text(b)
            if len(t.split()) <= 8 and rect_distance(r, reg) < 22 and not LABEL_RE.match(t):
                reg |= r
    return reg, False


def render_clip(page, rect, path, dpi=200):
    rect = (rect + (-6, -6, 6, 6)) & page.rect
    page.get_pixmap(dpi=dpi, clip=rect).save(path)
    return path


# --------------------------------------------------------------------------- tables
def cell_text(page, rect):
    if rect is None:
        return ""
    words = page.get_text("words", clip=pymupdf.Rect(rect))
    words.sort(key=lambda w: (round(w[3]), w[0]))
    return clean_text(" ".join(w[4] for w in words))


def extract_table_grid(page, tab):
    grid = []
    for row in tab.rows:
        grid.append([cell_text(page, c) for c in row.cells])
    # drop empty columns
    keep = [j for j in range(len(grid[0])) if any(r[j] for r in grid)]
    return [[r[j] for j in keep] for r in grid]


DEC_RE = re.compile(r"^[-−–]?\s?\d+\.\d+")


def is_data_cell(s):
    return bool(DEC_RE.match((s or "").strip()))


def is_yearlike(s):
    return bool(re.fullmatch(r"(19|20)\d\d\*?", (s or "").strip()))


def grid_to_markdown(grid):
    lines = ["| " + " | ".join(c.replace("|", "/") for c in r) + " |" for r in grid]
    if len(lines) > 1:
        lines.insert(1, "|" + "---|" * len(grid[0]))
    return "\n".join(lines)


def table_facts(grid, caption, notes, doc, page, label, trust):
    """Generic grid -> facts parser (years as columns or as rows, one or two label columns)."""
    data_rows = [i for i, r in enumerate(grid) if any(is_data_cell(c) for c in r)]
    if not data_rows:
        return []
    first = data_rows[0]
    header = grid[:first]
    ncol = len(grid[0])
    label_cols = [j for j in range(ncol) if not any(is_data_cell(grid[i][j]) for i in data_rows)]
    col_head = [" ".join(h[j] for h in header if h[j]).strip() for j in range(ncol)]
    facts, carry = [], {}
    ctx = caption + " " + notes
    for i in data_rows:
        row = grid[i]
        for j in label_cols:                      # carry forward merged label cells
            if row[j]:
                carry[j] = row[j]
        row_lab = " ".join(carry.get(j, "") for j in label_cols).strip()
        for j in range(ncol):
            if j in label_cols or not is_data_cell(row[j]):
                continue
            raw = row[j]
            m = re.match(r"^([-−–]?\s?\d+(?:\.\d+)?)\s*(?:\(([-−–]?\s?\d+(?:\.\d+)?)\))?", raw)
            val = parse_number(m.group(1))
            bracket = parse_number(m.group(2)) if m.group(2) else None
            ch = col_head[j]
            yrs, rngs = find_years(ch)
            if not yrs and not rngs:
                yrs, rngs = find_years(row_lab)
            ents = find_entities(row_lab) or find_entities(ch) or find_entities(caption)
            vars_ = find_variables(row_lab) or find_variables(ch) or find_variables(caption)
            forecast = ("*" in ch) or ("forecast" in ch.lower())
            year = yrs[0] if len(yrs) == 1 else None
            if year and doc.get("vintage_year") and year >= doc["vintage_year"]:
                forecast = True                     # beyond the data vintage = forecast
            base = {"doc_id": doc["doc_id"], "page": page, "label": label, "trust": trust,
                    "entity": ents[0] if ents else "", "variable": vars_[0] if vars_ else "",
                    "year": year, "period": f"{rngs[0][0]}-{rngs[0][1]}" if (rngs and not year) else "",
                    "row": row_lab, "col": ch, "raw": raw, "kind": "table"}
            facts.append({**base, "value": val, "forecast": forecast,
                          "outturn": "outturn" in ch.lower() or (not forecast and doc["doc_id"] != "DOC1"),
                          "revised_from": None})
            if bracket is not None and "forecast" in ctx.lower():
                facts.append({**base, "value": bracket, "forecast": True, "outturn": False,
                              "revised_from": None, "raw": raw + " [bracketed forecast]"})
    return facts


# --------------------------------------------------------------------------- text facts
PAIR_RE = re.compile(r"([-−]?\d+(?:\.\d+)?)%\s+in\s+((?:19|20)\d\d)")
REVISION_RE = re.compile(
    r"(?P<subj>.{0,80}?)\s+for\s+(?P<year>(?:19|20)\d\d)\s+(?:has\s+been|was|is)\s+revised\s+from\s+"
    r"(?P<old>[-−]?\d+(?:\.\d+)?)%(?P<mid>[^%]{0,80}?)\bto\s+(?P<new>[-−]?\d+(?:\.\d+)?)%", re.S)


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“(■])", text) if s.strip()]


def text_facts(text, doc, page, trust):
    facts = []
    for m in REVISION_RE.finditer(text):
        subj = re.split(r"(?<=[a-z]{2})\.\s", m.group("subj"))[-1]     # last sentence only
        ents, vars_ = find_entities(subj), find_variables(subj)
        if ents and vars_:
            facts.append({"doc_id": doc["doc_id"], "page": page, "label": "", "trust": trust,
                          "entity": ents[0], "variable": vars_[0], "year": int(m.group("year")),
                          "period": "", "value": parse_number(m.group("new")), "forecast": False,
                          "outturn": True, "revised_from": parse_number(m.group("old")),
                          "row": m.group(0)[:200], "col": "", "raw": m.group(0)[:200], "kind": "revision"})
    for s in sentences(text):
        if "revised" in s:
            continue
        pairs = PAIR_RE.findall(s)
        if not pairs:
            continue
        ents, vars_ = find_entities(s), find_variables(s)
        if len(ents) != 1 or len(vars_) != 1:
            continue
        if re.search(r"forecast|projected|expected", s, re.I):
            continue
        for v, y in pairs:
            y = int(y)
            fc = bool(doc.get("vintage_year") and y >= doc["vintage_year"])
            facts.append({"doc_id": doc["doc_id"], "page": page, "label": "", "trust": trust,
                          "entity": ents[0], "variable": vars_[0], "year": y, "period": "",
                          "value": parse_number(v), "forecast": fc, "outturn": not fc,
                          "revised_from": None, "row": s[:200], "col": "", "raw": s[:200], "kind": "text"})
    return facts


# --------------------------------------------------------------------------- main
def chunk_page_text(blocks_text, max_words=140, overlap_sents=1):
    sents = sentences(blocks_text)
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(x.split()) for x in cur) >= max_words:
            chunks.append(" ".join(cur))
            cur = cur[-overlap_sents:] if overlap_sents else []
    if cur and (not chunks or len(" ".join(cur).split()) > 15 or cur != chunks[-1:]):
        chunks.append(" ".join(cur))
    return chunks


def is_running_header(b, page):
    r = pymupdf.Rect(b["bbox"])
    t = block_text(b)
    if re.match(r"^(DOC\d\s*\|)|^Page \d+$", t):
        return True
    return (r.y1 < 48 or r.y0 > page.rect.height - 40) and len(t.split()) <= 8 and not LABEL_RE.match(t)


def ingest(data_dir, out_dir):
    os.makedirs(os.path.join(out_dir, "objects"), exist_ok=True)
    man, docs = load_manifest(data_dir)
    units, facts = [], []
    for doc in docs:
        pdf = pymupdf.open(os.path.join(data_dir, doc["file"]))
        for pno, page in enumerate(pdf, start=1):
            trust, section = page_trust(doc, pno)
            caps, blocks = find_captions(page)
            used, regions = [], []
            # ---- tables
            tabs = page.find_tables().tables
            for t in tabs:
                tb = pymupdf.Rect(t.bbox)
                tcaps = [c for c in caps if c["kind"] == "table"]
                if not tcaps:
                    continue
                cap = min(tcaps, key=lambda c: (0 if c["bbox"].y1 <= tb.y0 + 3 else 1, rect_distance(c["bbox"], tb)))
                grid = extract_table_grid(page, t)
                notes = []
                for b in sorted(blocks, key=lambda b: b["bbox"][1]):
                    r = pymupdf.Rect(b["bbox"])
                    if r.y0 >= tb.y1 - 2 and r.y0 - tb.y1 < 40 and not LABEL_RE.match(block_text(b)):
                        notes.append(clean_text(block_text(b)))
                        if len(notes) >= 2:
                            break
                notes_txt = " ".join(notes)
                region = tb | cap["bbox"]
                img = render_clip(page, region, os.path.join(out_dir, "objects", f"{doc['doc_id']}_{cap['label'].replace(' ', '_')}.png"))
                md = grid_to_markdown(grid)
                text = f"{cap['label']} ({doc['doc_id']}, page {pno}). {cap['text']}\n{md}\n{notes_txt}"
                units.append(Unit(uid=f"{doc['doc_id']}|{cap['label']}", kind="table", doc_id=doc["doc_id"], page=pno,
                                  trust=trust, text=text, label=cap["label"], caption=cap["text"], image_path=img,
                                  extra={"grid": grid, "notes": notes_txt}))
                facts += table_facts(grid, cap["text"], notes_txt, doc, pno, cap["label"], trust)
                regions.append(region)
                caps = [c for c in caps if c is not cap]
            # ---- figures
            for cap in [c for c in caps if c["kind"] == "figure"]:
                cap_txt, cap_rect = caption_column_text(cap, blocks)
                reg, raster = figure_region(page, cap, cap_rect, used)
                if reg is None:
                    continue
                used.append(reg)
                src = ""
                for b in sorted(blocks, key=lambda b: b["bbox"][1]):
                    r = pymupdf.Rect(b["bbox"])
                    if raster and r.y0 >= reg.y1 - 2 and r.y0 - reg.y1 < 25 and block_text(b).startswith("Source"):
                        src = clean_text(block_text(b))
                        cap_rect |= r
                        break
                region = reg | cap_rect
                layer_txt = "" if raster else clean_text(page.get_text("text", clip=reg))
                fname = f"{doc['doc_id']}_{cap['label'].replace(' ', '_').replace('(', '').replace(')', '')}.png"
                img = render_clip(page, region, os.path.join(out_dir, "objects", fname))
                text = f"{cap['label']} ({doc['doc_id']}, page {pno}). {cap_txt} {src}".strip()
                units.append(Unit(uid=f"{doc['doc_id']}|{cap['label']}", kind="figure", doc_id=doc["doc_id"], page=pno,
                                  trust=trust, text=text, label=cap["label"], caption=(cap_txt + " " + src).strip(),
                                  image_path=img, extra={"text_layer": layer_txt[:1500], "raster": raster,
                                                         "section": section}))
                regions.append(region)
            # ---- text
            body = []
            for b in blocks:
                r = pymupdf.Rect(b["bbox"])
                if is_running_header(b, page):
                    continue
                if any((r & g).get_area() > 0.6 * max(r.get_area(), 1) for g in regions):
                    continue
                para = clean_text("\n".join(" ".join(s["text"] for s in l["spans"]) for l in b["lines"]))
                body.append(para)
                facts += text_facts(para, doc, pno, trust)       # per paragraph, so headings stay apart
            page_txt = " ".join(body)
            for k, ch in enumerate(chunk_page_text(page_txt)):
                units.append(Unit(uid=f"{doc['doc_id']}|p{pno}|c{k}", kind="text", doc_id=doc["doc_id"], page=pno,
                                  trust=trust, text=ch, extra={"section": section,
                                                               "labels_mentioned": find_labels(ch)}))
    with open(os.path.join(out_dir, "units.json"), "w") as f:
        json.dump([asdict(u) for u in units], f, indent=1)
    with open(os.path.join(out_dir, "facts.json"), "w") as f:
        json.dump(facts, f, indent=1)
    return units, facts, docs


def load_units(out_dir):
    with open(os.path.join(out_dir, "units.json")) as f:
        units = [Unit(**u) for u in json.load(f)]
    with open(os.path.join(out_dir, "facts.json")) as f:
        facts = json.load(f)
    return units, facts
