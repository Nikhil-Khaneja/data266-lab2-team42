import os
import re
import json
import pymupdf as fitz


FIGURE_LABEL_PATTERN = re.compile(r"((?:Figure|Table)\s+[A-Z0-9]+-?[0-9]*)(\s*\(corrected\))?", re.IGNORECASE)


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


def extract_text_chunks(pdf_path, doc_id, trust_for_page, chunk_chars=3000, overlap_chars=150):
    # chunk_chars is a safety cap for unusually long pages; a whole page normally
    # becomes one chunk so a table's rows are never split across chunks (splitting
    # tables mid-row was causing the model to read numbers from the wrong row/column).
    doc = fitz.open(pdf_path)
    chunks = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        text = page.get_text("text")
        text = text.strip()
        if not text:
            continue
        trust_level = trust_for_page(page_number)
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunk_text = text[start:end]
            chunks.append({
                "doc_id": doc_id,
                "page": page_number,
                "trust_level": trust_level,
                "text": chunk_text,
                "kind": "text",
            })
            if end == len(text):
                break
            start = end - overlap_chars
    doc.close()
    return chunks


def find_figure_labels(caption_text):
    labels = []
    for match in FIGURE_LABEL_PATTERN.finditer(caption_text):
        label = match.group(1).strip()
        if match.group(2):
            label = label + " (corrected)"
        labels.append(label)
    return labels


def find_caption_labels(page_text):
    # A real figure/table caption starts its own line, e.g. "Figure A-4 Stock prices...".
    # A cross-reference like "as in Figure 1-1 of Document 1" appears mid-sentence and
    # must not be mistaken for a caption, or image-to-label alignment breaks.
    labels = []
    for line in page_text.splitlines():
        match = FIGURE_LABEL_PATTERN.match(line.strip())
        if match:
            label = match.group(1).strip()
            if match.group(2):
                label = label + " (corrected)"
            labels.append(label)
    return labels


def extract_figures(pdf_path, doc_id, trust_for_page, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    figures = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        page_number = page_index + 1
        page_text = page.get_text("text")
        image_list = page.get_images(full=True)
        trust_level = trust_for_page(page_number)
        captions_on_page = find_caption_labels(page_text)

        if not image_list:
            # Charts drawn with PDF vector commands (common in this textbook) have no
            # embedded raster image for get_images() to find. Render the whole page as
            # a fallback so the chart is still indexed and citable, one entry per
            # "Figure ..." caption found on the page (captions for "Table ..." are
            # skipped here since table data is already captured in the text chunks).
            figure_captions = [c for c in captions_on_page if c.lower().startswith("figure")]
            if figure_captions:
                pixmap = page.get_pixmap(dpi=150)
                for label in figure_captions:
                    safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
                    file_name = "{}_p{}_{}_page.png".format(doc_id, page_number, safe_label)
                    file_path = os.path.join(output_dir, file_name)
                    pixmap.save(file_path)
                    figures.append({
                        "doc_id": doc_id,
                        "page": page_number,
                        "trust_level": trust_level,
                        "figure_id": label,
                        "caption": label,
                        "image_path": file_path,
                        "kind": "figure",
                    })
            continue

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            label = captions_on_page[img_index] if img_index < len(captions_on_page) else "Figure_unlabeled_p{}_{}".format(page_number, img_index)
            safe_label = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_")
            file_name = "{}_p{}_{}.{}".format(doc_id, page_number, safe_label, ext)
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(image_bytes)
            figures.append({
                "doc_id": doc_id,
                "page": page_number,
                "trust_level": trust_level,
                "figure_id": label,
                "caption": label,
                "image_path": file_path,
                "kind": "figure",
            })
    doc.close()
    return figures


def ingest_documents(manifest_path, pdf_dir, figures_output_dir):
    manifest = load_manifest(manifest_path)
    all_text_chunks = []
    all_figures = []
    for doc_entry in manifest["documents"]:
        doc_id = doc_entry["doc_id"]
        file_name = doc_entry["file"]
        trust_for_page = build_page_trust_lookup(doc_entry)
        pdf_path = os.path.join(pdf_dir, file_name)
        all_text_chunks.extend(extract_text_chunks(pdf_path, doc_id, trust_for_page))
        all_figures.extend(extract_figures(pdf_path, doc_id, trust_for_page, figures_output_dir))
    return all_text_chunks, all_figures
