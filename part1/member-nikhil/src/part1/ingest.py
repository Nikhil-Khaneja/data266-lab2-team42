"""PDF ingestion: text chunks, structured table chunks, and one cropped image per numbered
Figure/Table, each carrying doc_id, page, figure_id and trust_level from kb_manifest.json.

Captions are matched to images by page geometry, never by list position: on a page holding
"Table A-1" and then "Figure A-1", pairing by position labels the chart as the table and
loses the figure entirely.
"""
import json
import os
import re

import pymupdf as fitz

LABEL_CORE = r"(?:[12]|[A-C])-\d+"
CAPTION_RE = re.compile(r"^(Figure|Table)\s+(" + LABEL_CORE + r")(\s*\(corrected\))?(?=$|\s|\.)")
HEAVY_FONT_RE = re.compile(r"bold|black|heavy|-hv|-bd", re.IGNORECASE)
RENDER_DPI = 300
TEXT_CHUNK_CHARS = 1100

REPLACEMENTS = {"−": "-", "ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi",
                "‘": "'", "’": "'", "“": '"', "”": '"', "­": ""}


def normalize_text(text):
    for bad, good in REPLACEMENTS.items():
        text = text.replace(bad, good)
    # The textbook sets year ranges as "2000 2007<br>-"; put them back together.
    text = re.sub(r"(\d{4})\s+(\d{4})\s*(?:<br>)?\s*[-–]", r"\1-\2", text)
    return text


def load_manifest(manifest_path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_page_trust_lookup(doc_entry):
    base_trust = doc_entry["trust_level"]
    overrides = {}
    for override in doc_entry.get("section_trust_overrides", []):
        for page in override["pages"]:
            overrides[page] = override["trust_level"]

    def trust_for_page(page_number):
        return overrides.get(page_number, base_trust)

    return trust_for_page


def find_captions(page):
    """Numbered captions on a page, identified by bold type at the start of a line, so a
    sentence that merely begins with "Figure 1-4 shows..." is not taken for a caption."""
    captions = []
    for block in page.get_text("dict")["blocks"]:
        lines = block.get("lines", [])
        for idx, line in enumerate(lines):
            spans = line["spans"]
            if not spans:
                continue
            text = normalize_text("".join(s["text"] for s in spans)).strip()
            match = CAPTION_RE.match(text)
            if not match:
                continue
            first = spans[0]
            is_bold = bool(HEAVY_FONT_RE.search(first["font"])) or bool(first["flags"] & 16)
            if not is_bold:
                continue
            label = "{} {}".format(match.group(1), match.group(2))
            if match.group(3):
                label += " (corrected)"
            title = [text]
            for following in lines[idx + 1:idx + 3]:
                title.append(normalize_text("".join(s["text"] for s in following["spans"])).strip())
            captions.append({"label": label, "kind": match.group(1).lower(), "bbox": fitz.Rect(line["bbox"]),
                             "title": " ".join(t for t in title if t)})
    return captions


def nearby_title_text(page, caption):
    """The textbook prints a short caption then an italic title in the next block."""
    if len(caption["title"]) > len(caption["label"]) + 2:
        return caption["title"]
    best = caption["label"]
    for block in page.get_text("blocks"):
        rect = fitz.Rect(block[:4])
        if rect.y0 >= caption["bbox"].y1 - 1 and rect.y0 - caption["bbox"].y1 < 25 and abs(rect.x0 - caption["bbox"].x0) < 30:
            return "{} {}".format(caption["label"], normalize_text(block[4]).replace("\n", " ").strip())
    return best


def table_to_text(page, table):
    # Read each cell's own rectangle: table.extract() splits decimals apart ("3.2" -> "32 .").
    rows = []
    row_header = ""
    for row in table.rows:
        cells = []
        for cell in row.cells:
            text = page.get_text("text", clip=fitz.Rect(cell)) if cell else ""
            cells.append(normalize_text(re.sub(r"\s+", " ", text)).strip())
        if not any(cells):
            continue
        # Merged row headers: "Unemployment rate" under "Euro area" has an empty first cell,
        # and without the country the row is ambiguous once read as flat text.
        if cells[0]:
            row_header = cells[0]
        elif len(cells) > 2 and row_header and any(cells[1:2]):
            cells[0] = row_header
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def table_footnotes(page, bbox):
    notes = []
    for block in sorted(page.get_text("blocks"), key=lambda b: b[1]):
        rect = fitz.Rect(block[:4])
        if bbox.y1 - 2 <= rect.y0 <= bbox.y1 + 45 and rect.x0 < bbox.x1 and rect.x1 > bbox.x0:
            text = normalize_text(block[4]).replace("\n", " ").strip()
            if text.startswith(("Source", "Output growth", "*", "Percent", "A negative", "Note")):
                notes.append(text)
    return " ".join(notes)


def extract_tables(page, captions):
    """Numbered tables only (matched to the caption directly above them). Returns the table
    objects' chunks plus every table bbox, so their cells are not repeated as loose prose."""
    found, bboxes = [], []
    table_captions = [c for c in captions if c["kind"] == "table"]
    for table in page.find_tables().tables:
        bbox = fitz.Rect(table.bbox)
        above = [c for c in table_captions if c["bbox"].y1 <= bbox.y0 + 5 and bbox.y0 - c["bbox"].y1 < 60]
        if not above:
            continue  # unnumbered layout table: leave it in the page text
        caption = max(above, key=lambda c: c["bbox"].y1)
        bboxes.append(bbox)
        found.append({"caption": caption, "bbox": bbox, "body": table_to_text(page, table),
                      "notes": table_footnotes(page, bbox)})
    return found, bboxes


def vector_figure_region(page, exclude):
    """Figures drawn with PDF vector commands have no raster image. Cluster the page's
    drawings and take the largest cluster outside any table."""
    rects = []
    for d in page.get_drawings():
        r = fitz.Rect(d["rect"])
        # Printer's crop marks span past the page edge and would swallow the whole page.
        if r.is_empty or r.x0 < 0 or r.y0 < 0 or r.width > page.rect.width * 0.95:
            continue
        if any(r.intersects(e) for e in exclude):
            continue
        rects.append(r)
    clusters = []
    for r in rects:
        grown = r + (-18, -18, 18, 18)
        merged = fitz.Rect(r)
        rest = []
        for c in clusters:
            if grown.intersects(c):
                merged |= c
            else:
                rest.append(c)
        rest.append(merged)
        clusters = rest
    if not clusters:
        return None
    best = max(clusters, key=lambda c: c.width * c.height)
    if best.width * best.height < 8000:
        return None
    # Axis labels and rotated tick text are text objects just outside the drawn lines.
    return (best + (-45, -14, 20, 48)) & page.rect


def raster_regions(page):
    regions = []
    for img in page.get_images(full=True):
        for rect in page.get_image_rects(img[0]):
            if rect.width > 150 and rect.height > 100:
                regions.append(fitz.Rect(rect))
    return sorted(regions, key=lambda r: r.y0)


def match_figures_to_regions(page, figure_captions, table_bboxes):
    """One region per figure caption: the nearest raster image BELOW its caption (the
    supplements caption above the chart), else the page's vector drawing, else the page."""
    regions = raster_regions(page)
    matched, used = [], set()
    for caption in sorted(figure_captions, key=lambda c: c["bbox"].y0):
        candidates = [(r.y0 - caption["bbox"].y1, i) for i, r in enumerate(regions)
                      if i not in used and r.y0 >= caption["bbox"].y1 - 5]
        if candidates:
            _, i = min(candidates)
            used.add(i)
            matched.append((caption, regions[i]))
            continue
        region = vector_figure_region(page, table_bboxes)
        matched.append((caption, region if region is not None else page.rect))
    return matched


def safe_name(label):
    return re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")


def save_crop(page, rect, path):
    page.get_pixmap(dpi=RENDER_DPI, clip=rect).save(path)


def text_chunks_for_page(page, skip_rects):
    """Paragraph-aware chunks, so one passage is retrievable without the rest of the page."""
    blocks = []
    for block in sorted(page.get_text("blocks"), key=lambda b: (b[1], b[0])):
        if block[6] != 0:
            continue
        rect = fitz.Rect(block[:4])
        if any(rect.intersects(s) and (rect & s).get_area() > 0.5 * max(rect.get_area(), 1) for s in skip_rects):
            continue
        text = re.sub(r"\s*\n\s*", " ", normalize_text(block[4])).strip()
        if len(text) >= 3:
            blocks.append(text)
    chunks, current = [], ""
    for text in blocks:
        if current and len(current) + len(text) > TEXT_CHUNK_CHARS:
            chunks.append(current)
            current = ""
        current = (current + "\n" + text).strip()
    if current:
        chunks.append(current)
    return chunks


def ingest_documents(manifest_path, pdf_dir, figures_output_dir):
    manifest = load_manifest(manifest_path)
    os.makedirs(figures_output_dir, exist_ok=True)
    all_chunks, all_objects = [], []
    for doc_entry in manifest["documents"]:
        doc_id = doc_entry["doc_id"]
        trust_for_page = build_page_trust_lookup(doc_entry)
        doc = fitz.open(os.path.join(pdf_dir, doc_entry["file"]))
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_number = page_index + 1
            trust = trust_for_page(page_number)
            base = {"doc_id": doc_id, "page": page_number, "trust_level": trust}
            captions = find_captions(page)
            tables, table_bboxes = extract_tables(page, captions)

            for t in tables:
                label = t["caption"]["label"]
                path = os.path.join(figures_output_dir, "{}_p{}_{}.png".format(doc_id, page_number, safe_name(label)))
                save_crop(page, (t["bbox"] | t["caption"]["bbox"]) & page.rect, path)
                chunk = dict(base, kind="table", figure_id=label, image_path=path,
                             text="{}\n{}\n{}".format(nearby_title_text(page, t["caption"]), t["body"], t["notes"]).strip())
                all_chunks.append(chunk)
                all_objects.append(chunk)

            for caption, region in match_figures_to_regions(
                    page, [c for c in captions if c["kind"] == "figure"], table_bboxes):
                label = caption["label"]
                path = os.path.join(figures_output_dir, "{}_p{}_{}.png".format(doc_id, page_number, safe_name(label)))
                save_crop(page, region, path)
                chunk = dict(base, kind="figure", figure_id=label, image_path=path,
                             caption=nearby_title_text(page, caption), text=nearby_title_text(page, caption))
                all_chunks.append(chunk)
                all_objects.append(chunk)

            for text in text_chunks_for_page(page, table_bboxes):
                all_chunks.append(dict(base, kind="text", figure_id="", image_path="", text=text))
        doc.close()
    return all_chunks, all_objects
