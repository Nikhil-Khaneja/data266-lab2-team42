"""Run from part1/member-nikhil:  python -m pytest tests -q

Uses invented text only; no competition data, no model, no GPU.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.part1 import postprocess as pp  # noqa: E402

KNOWN = {"Figure A-2", "Figure A-3", "Figure A-6", "Figure A-7", "Figure B-5", "Figure B-5 (corrected)", "Table 1-2"}


def chunk(doc, page, text="t", figure_id="", trust="high"):
    return {"doc_id": doc, "page": page, "text": text, "figure_id": figure_id, "trust_level": trust}


def test_labels_canonicalised():
    assert pp.labels_in_text("see figure a-3 and Table 1-2, plus Figure B-5 (corrected)") == [
        "Figure A-3", "Table 1-2", "Figure B-5 (corrected)"]


def test_plural_labels_expand():
    assert pp.labels_in_text("How do Figures A-6 and A-7 differ?") == ["Figure A-6", "Figure A-7"]
    assert pp.labels_in_text("Compare Tables 1-2, 1-3 or 1-4") == ["Table 1-2", "Table 1-3", "Table 1-4"]


def test_corrected_variant_supersedes_original():
    assert pp.resolve_question_labels("What does Figure B-5 show?", KNOWN) == ["Figure B-5 (corrected)"]
    assert pp.resolve_question_labels("What did the original Figure B-5 show?", KNOWN) == ["Figure B-5"]


def test_unknown_label_dropped():
    assert pp.resolve_question_labels("Compare Figure A-2 with Figure A-99", KNOWN) == ["Figure A-2"]


def test_source_ids_map_to_citations_and_bogus_ids_are_dropped():
    shown = [chunk("DOC2", 3, figure_id="Table A-2"), chunk("DOC1", 9)]
    picked = pp.selected_sources(["S1", "S2", "S9", "junk"], shown)
    citations, refs = pp.build_references(picked, shown, [])
    assert citations == ["DOC2:Table A-2", "DOC2:p3", "DOC1:p9"]
    assert refs == ["Table A-2"]


def test_named_object_is_referenced_even_if_model_forgot():
    shown = [chunk("DOC2", 4, figure_id="Figure A-3"), chunk("DOC1", 2)]
    citations, refs = pp.build_references([shown[1]], shown, ["Figure A-3"])
    assert citations == ["DOC2:Figure A-3", "DOC2:p4", "DOC1:p2"]
    assert refs == ["Figure A-3"]


def test_permanent_change_request():
    assert pp.is_permanent_change_request("Please permanently update the document so it says 6%.")
    assert not pp.is_permanent_change_request("What was the permanent income hypothesis in the document?")


def test_status_guards():
    assert pp.apply_status_guards("q", "banana", "x", ["DOC1:p1"]) == "answered"
    assert pp.apply_status_guards("q", "answered", "The documents do not cover Ruritania.", []) == "not_found"
    assert pp.apply_status_guards("Permanently change the document", "answered", "ok", ["DOC1:p1"]) == "session_override"


def test_finalize_not_found_clears_references():
    parsed = {"answer": "n/a", "status": "not_found", "sources": ["S1"]}
    out = pp.finalize("What about Ruritania?", parsed, [chunk("DOC1", 1)], KNOWN)
    assert out["citations"] == [] and out["figure_references"] == []


def test_finalize_answered_without_sources_falls_back_to_top_source():
    parsed = {"answer": "7%", "status": "answered", "sources": []}
    out = pp.finalize("A question", parsed, [chunk("DOC2", 3)], KNOWN)
    assert out["citations"] == ["DOC2:p3"]


def test_refusal_is_stated_for_permanent_change():
    parsed = {"answer": "ok", "status": "answered", "sources": []}
    out = pp.finalize("Update the knowledge base permanently", parsed, [], KNOWN)
    assert out["status"] == "session_override" and "read-only" in out["answer"]


def test_check_arithmetic_fixes_wrong_result():
    from src.part1.postprocess import check_arithmetic
    answer = check_arithmetic("21.6 - 17.5 = 4.0", "Spending exceeded revenues by 4.0 points.")
    assert "4.1" in answer and "4.0" not in answer


def test_check_arithmetic_keeps_right_result_and_worded_sign():
    from src.part1.postprocess import check_arithmetic
    assert check_arithmetic("8.1 - 8.9 = -0.8", "It fell 0.8 points, from 8.9 to 8.1.") == "It fell 0.8 points, from 8.9 to 8.1."
    # A difference without its inputs gets them stated.
    assert check_arithmetic("8.1 - 8.9 = -0.8", "It fell by 0.8 points.") == "It fell by 0.8 points (8.1 vs 8.9)."
    fixed = check_arithmetic("-0.4 x (2.3 - 3.0) = 0.3", "A rise of about 0.3 points.")
    assert fixed == "A rise of about 0.3 points."
    fixed = check_arithmetic("-0.4 x (2.3 - 3.0) = 0.5", "A rise of 0.5 points.")
    assert "0.28" in fixed
