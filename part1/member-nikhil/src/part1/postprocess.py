"""Deterministic post-processing of the model's JSON answer.

Status, figure_references and citations are 60% of the Kaggle score. The model only names
which numbered sources it used ("S2"); this module turns those into correctly formatted
citations and labels, so a citation can be neither malformed nor invented. Everything here
is a general rule from the lab's answering rules and output grammar -- nothing is keyed to
a particular question.

Pure Python (no torch/chroma imports) so it can be unit-tested anywhere.
"""
import re

_ONE = r"(?:[12]|[A-C])\s*-\s*\d+(?:\s*\(corrected\))?"
# Handles singular and plural forms: "Figure A-6", "Figures A-6 and A-7", "Tables 1-2, 1-3".
LABEL_GROUP_RE = re.compile(
    r"\b(Figure|Table)s?\s+(" + _ONE + r"(?:\s*(?:,|/|&|\band\b|\bor\b|\bvs\.?|\bversus\b)\s*" + _ONE + r")*)",
    re.IGNORECASE)
LABEL_ONE_RE = re.compile(r"([12]|[A-C])\s*-\s*(\d+)(\s*\(corrected\))?", re.IGNORECASE)
VALID_STATUSES = {"answered", "not_found", "conflict", "false_premise", "session_override"}

# A request to change the documents themselves: needs a change verb, a target noun and a
# statement of permanence, so an ordinary question that says "permanent" is not caught.
CHANGE_VERB = re.compile(
    r"\b(change|update|edit|modify|overwrite|replace|correct|fix|rewrite|alter|save|store|remember)\b", re.IGNORECASE)
CHANGE_TARGET = re.compile(
    r"\b(documents?|pdfs?|knowledge base|kb|database|source|record|index|it)\b", re.IGNORECASE)
CHANGE_PERMANENCE = re.compile(
    r"\b(permanent(ly)?|forever|for good|from now on|for everyone|for all (users|conversations)|all future)\b",
    re.IGNORECASE)

NOT_FOUND_PHRASES = re.compile(
    r"\b(does not (support|contain|mention|cover|discuss)|do not (support|contain|mention|cover|discuss)|"
    r"not available|no information|not covered|not found|cannot be answered)\b", re.IGNORECASE)

REFUSAL_PREFIX = "I can't change the documents; they are read-only, and any value you supply applies to this conversation only. "


def canonical_label(kind, namespace, number, corrected=False):
    label = "{} {}-{}".format(kind.capitalize(), namespace.upper(), number)
    return label + (" (corrected)" if corrected else "")


def labels_in_text(text):
    """All Figure/Table labels in text, canonicalised, in order, de-duplicated."""
    out = []
    for group in LABEL_GROUP_RE.finditer(text):
        for m in LABEL_ONE_RE.finditer(group.group(2)):
            label = canonical_label(group.group(1), m.group(1), m.group(2), bool(m.group(3)))
            if label not in out:
                out.append(label)
    return out


def is_permanent_change_request(question):
    return bool(
        CHANGE_VERB.search(question) and CHANGE_TARGET.search(question) and CHANGE_PERMANENCE.search(question)
    )


def resolve_question_labels(question, known_labels):
    """Labels the question names, resolved against what the index contains. A correction
    replaces the earlier object, so "Figure B-5" means the corrected figure if one exists,
    unless the question explicitly asks about the original. Unknown labels are dropped."""
    wants_original = bool(re.search(r"\b(original|uncorrected|before the (correction|erratum))\b", question, re.I))
    resolved = []
    for label in labels_in_text(question):
        corrected = label.replace(" (corrected)", "") + " (corrected)"
        candidate = corrected if corrected in known_labels and not wants_original else label
        if candidate in known_labels and candidate not in resolved:
            resolved.append(candidate)
    return resolved


def citation_for(chunk):
    """The citations one source contributes: its label (if a table/figure), then its page."""
    out = []
    if chunk.get("figure_id"):
        out.append("{}:{}".format(chunk["doc_id"], chunk["figure_id"]))
    out.append("{}:p{}".format(chunk["doc_id"], chunk["page"]))
    return out


def selected_sources(model_sources, shown_chunks):
    """Map the model's "S2"-style ids onto the chunks shown to it. Unknown ids are dropped."""
    picked = []
    for raw in model_sources or []:
        match = re.search(r"\d+", str(raw))
        if match:
            i = int(match.group()) - 1
            if 0 <= i < len(shown_chunks) and shown_chunks[i] not in picked:
                picked.append(shown_chunks[i])
    return picked


def build_references(picked, shown_chunks, question_labels, max_citations=6):
    """Return (citations, figure_references). A figure or table the question names is always
    included, because the question is answered from it, even if the model did not list it."""
    ordered = [c for label in question_labels for c in shown_chunks if c.get("figure_id") == label]
    # A page chunk adds nothing when the same page's table/figure was selected; keep order.
    ordered += [c for c in picked if c not in ordered]
    citations, figure_refs = [], []
    for chunk in ordered:
        for citation in citation_for(chunk):
            if citation not in citations:
                citations.append(citation)
        if chunk.get("figure_id") and chunk["figure_id"] not in figure_refs:
            figure_refs.append(chunk["figure_id"])
    return citations[:max_citations], figure_refs


def apply_status_guards(question, status, answer, citations):
    status = (status or "").strip().lower()
    if status not in VALID_STATUSES:
        status = "answered" if citations else "not_found"
    if is_permanent_change_request(question):
        return "session_override"
    # Said "the documents don't cover this" in prose but labelled it answered with nothing to cite.
    if status == "answered" and not citations and NOT_FOUND_PHRASES.search(answer or ""):
        return "not_found"
    return status


_NUM = r"-?\d+(?:\.\d+)?"
ARITHMETIC = re.compile(
    r"((?:[(\-]\s*)*" + _NUM + r"\)*(?:\s*[-+*/x×]\s*(?:[(\-]\s*)*" + _NUM + r"\)*)+)\s*=\s*(" + _NUM + r")")


def _evaluate(expression):
    import ast
    import operator
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.USub: operator.neg, ast.UAdd: operator.pos}

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.left), walk(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in ops:
            return ops[type(node.op)](walk(node.operand))
        raise ValueError("unsupported")
    return walk(ast.parse(expression, mode="eval"))


def check_arithmetic(evidence, answer):
    """The model writes its arithmetic in "evidence" ("8.1 - 8.9 = -0.8"). A small model often
    gets the result wrong, so each expression is recomputed and a wrong result in the answer
    is replaced by the correct one."""
    text = (evidence or "").replace("−", "-").replace("–", "-")
    for match in ARITHMETIC.finditer(text):
        expression = match.group(1).replace("x", "*").replace("×", "*")
        stated = match.group(2)
        try:
            value = _evaluate(expression)
        except (ValueError, SyntaxError, ZeroDivisionError):
            continue
        decimals = max(len(stated.split(".")[1]) if "." in stated else 0, 1)
        if abs(abs(value) - abs(float(stated))) < 0.5 * 10 ** -decimals + 1e-9:
            continue  # right, or only the sign is expressed in words ("fell by 0.8")
        correct = "{:.{}f}".format(abs(value) if not stated.startswith("-") else value, max(decimals, 2))
        correct = correct.rstrip("0").rstrip(".") if "." in correct else correct
        if "." not in correct and "." in stated:
            correct += ".0"
        answer = re.sub(r"(?<![\w.])" + re.escape(stated.lstrip("-")) + r"(?![\w]|\.\d)",
                        correct.lstrip("-"), answer)
    # A difference is only checkable with its two inputs, so they are stated when missing.
    for match in ARITHMETIC.finditer(text):
        parts = re.fullmatch(r"\s*(" + _NUM + r")\s*-\s*(" + _NUM + r")\s*", match.group(1))
        if parts and not any(_mentions(answer, n) for n in parts.groups()):
            answer = answer.rstrip() + " ({} vs {}).".format(*parts.groups())
            answer = answer.replace(". (", " (").replace("..", ".")
    return answer


def correct_arithmetic(text):
    """Rewrite each "expression = result" in free-text working with the recomputed result."""
    text = (text or "").replace("−", "-").replace("–", "-")

    def fix(match):
        try:
            value = _evaluate(match.group(1).replace("x", "*").replace("×", "*"))
        except (ValueError, SyntaxError, ZeroDivisionError):
            return match.group(0)
        stated = match.group(2)
        decimals = max(len(stated.split(".")[1]) if "." in stated else 0, 1)
        if abs(value - float(stated)) < 0.5 * 10 ** -decimals + 1e-9:
            return match.group(0)
        shown = "{:.2f}".format(value).rstrip("0").rstrip(".")
        return "{} = {}".format(match.group(1).rstrip(), shown)
    return ARITHMETIC.sub(fix, text)


def _mentions(text, number):
    return bool(re.search(r"(?<![\w.])" + re.escape(number.lstrip("-")) + r"(?![\w]|\.\d)", text))


def truncate_words(text, limit=120):
    return " ".join((text or "").split()[:limit])


def finalize(question, parsed, shown_chunks, known_labels, max_citations=6):
    """Turn the model's parsed JSON into the final submission fields."""
    question_labels = resolve_question_labels(question, known_labels)
    picked = selected_sources(parsed.get("sources", []), shown_chunks)
    citations, figure_refs = build_references(picked, shown_chunks, question_labels, max_citations)
    answer = check_arithmetic(parsed.get("evidence", ""), parsed.get("answer", ""))
    status = apply_status_guards(question, parsed.get("status"), answer, citations)
    if status == "not_found":
        citations, figure_refs = [], []  # nothing supports an answer, so nothing can be cited
    elif not citations and status in ("answered", "conflict", "false_premise") and shown_chunks:
        citations, figure_refs = build_references([shown_chunks[0]], shown_chunks, [], max_citations)
    if is_permanent_change_request(question) and not answer.lower().startswith("i can't"):
        answer = REFUSAL_PREFIX + answer
    return {
        "answer": truncate_words(answer, 120) or "The documents do not support an answer to this question.",
        "status": status,
        "citations": citations,
        "figure_references": figure_refs,
    }
