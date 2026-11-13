"""Turn each numbered table into atomic facts, so the answer step gets the few cells relevant
to a question instead of having to find the right row and column itself.

"Euro area | Unemployment rate | 11.4 (9.9)" under column "2012" becomes
"Euro area, Unemployment rate, 2012: 11.4 outturn (fall-2011 forecast 9.9)".
Everything is derived from the table's own headers, so no table or question is special-cased.
"""
import re

NUMERIC_CELL = re.compile(r"^-?\$?\d[\d,]*(\.\d+)?%?(\s*\(-?\d+(\.\d+)?\))?$")
BRACKETED_CELL = re.compile(r"^(-?\d+(?:\.\d+)?)\s*\((-?\d+(?:\.\d+)?)\)$")
YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
RANGE = re.compile(r"\b((?:19|20)\d{2})\s*[-–]\s*((?:19|20)\d{2})\b")
# The lab states that DOC1 was prepared in the fall of 2011: its 2011-2012 values are forecasts.
FORECAST_YEARS = {"DOC1": {"2011", "2012"}}


def is_value(cell):
    # A bare year is a heading, never a value.
    return bool(NUMERIC_CELL.match(cell)) and not re.fullmatch(r"(?:19|20)\d{2}", cell)


def is_data_row(cells):
    return any(is_value(c) for c in cells[1:])


def parse_table(text):
    """(caption, header rows, data rows) from a table chunk's text: a caption line, rows of
    " | "-separated cells, and optionally a final footnote line."""
    lines = text.split("\n")
    caption = lines[0]
    rows = [[c.strip() for c in line.split(" | ")] for line in lines[1:] if " | " in line]
    headers = []
    for row in rows:
        if is_data_row(row):
            break
        headers.append(row)
    return caption, headers, rows[len(headers):]


def column_names(headers, width):
    names = []
    for col in range(width):
        parts = [h[col] for h in headers if col < len(h) and h[col]]
        names.append(" ".join(parts))
    return names


def table_facts(chunk):
    """Facts for one table chunk, each carrying the chunk's doc, page and label."""
    caption, headers, rows = parse_table(chunk["text"])
    width = max((len(r) for r in rows), default=0)
    columns = column_names(headers, width)
    title = re.sub(r"^Table\s+\S+\s*", "", caption).strip()
    facts = []
    for row in rows:
        leading = []
        for col, cell in enumerate(row):
            if not cell:
                continue
            # The first column always labels the row, even when it is a year or number.
            if col == 0 or not is_value(cell):
                leading.append(cell)
                continue
            name = columns[col] if col < len(columns) else ""
            label_text = " ".join([name] + leading)
            ranges = RANGE.findall(label_text)
            years = set(YEAR.findall(RANGE.sub(" ", label_text)))
            value = cell
            bracketed = BRACKETED_CELL.match(cell)
            if bracketed:
                value = "{} outturn (fall-2011 forecast {})".format(bracketed.group(1), bracketed.group(2))
            elif "*" in name or "forecast" in name.lower() or years & FORECAST_YEARS.get(chunk["doc_id"], set()):
                value = "{} (forecast)".format(cell)
            subject = ", ".join(leading)
            described = "{}, {}".format(subject, name) if name else subject
            facts.append({"text": "{}: {}".format(described, value), "context": "{}. {}".format(title, described),
                          "years": years, "ranges": ranges, "doc_id": chunk["doc_id"], "page": chunk["page"],
                          "figure_id": chunk["figure_id"], "trust_level": chunk["trust_level"],
                          "image_path": chunk.get("image_path", "")})
    return facts


def fact_matches_years(fact, years):
    """A fact qualifies if it is for a named year, or for a range whose two ends are both named."""
    return bool(years & fact["years"]) or any(a in years and b in years for a, b in fact["ranges"])


def group_facts(facts, tables_by_label):
    """Group selected facts under their table, as one source per table, for citation mapping."""
    groups, order = {}, []
    for fact in facts:
        key = (fact["doc_id"], fact["figure_id"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(fact["text"])
    chunks = []
    for key in order:
        table = tables_by_label[key]
        caption = table["text"].split("\n")[0]
        chunks.append(dict(table, kind="table_facts", text="{}\n{}".format(caption, "\n".join(groups[key]))))
    return chunks


# --- Deterministic lookups -------------------------------------------------------------------
# Every fact gets a key (entity, variable, year). Comparing the values under one key decides
# revision, forecast-versus-outturn and conflict by the manifest's precedence rules, instead of
# leaving those decisions to the model.
ENTITIES = [
    ("Advanced economies", r"advanced[- ]?(econom(y|ies)|countries)|rich(est)? countries"),
    ("Emerging economies", r"emerging|developing"),
    ("Euro area", r"euro[- ]?(area|zone)|\beurope(an)?\b"),
    ("China", r"\bchina\b|\bchinese\b"),
    ("United States", r"\bu\.?s\.?(a\.?)?\b|united states|\bamerica"),
    ("World", r"\bworld\b|\bglobal\b"),
]
VARIABLES = [
    ("participation", r"participation"),
    ("unemployment", r"unemploy|jobless|\bjobs\b|out of work"),
    ("inflation", r"inflation"),
    ("budget balance", r"budget|deficit|surplus"),
    ("revenues", r"revenues?"),
    ("spending", r"spending|expenditure"),
    ("output growth", r"output|growth|\bgdp\b|\bgrew\b|\bgrow\b"),
]
ECONOMY = re.compile(r"\beconom(y|ies)\b", re.IGNORECASE)
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
PERCENT = re.compile(r"-?\d+(?:\.\d+)?%")
REVISED = re.compile(r"revised from\s+(-?\d+(?:\.\d+)?)%.*?\bto\s+(-?\d+(?:\.\d+)?)%", re.IGNORECASE | re.DOTALL)
SINCE_YEAR = re.compile(r"\b(?:its )?(?:\w+ )*?since (?:19|20)\d{2}\b", re.IGNORECASE)
RANGE_WORDS = re.compile(r"\b(since|between|from|until|by the|average)\b", re.IGNORECASE)


def find_first(table, text):
    for name, pattern in table:
        if re.search(pattern, text, re.IGNORECASE):
            return name
    return None


def keyed_table_facts(chunk):
    title = re.sub(r"^Table\s+\S+\s*", "", chunk["text"].split("\n")[0]).strip()
    facts = table_facts(chunk)
    for fact in facts:
        described = fact["context"].rsplit(". ", 1)[-1]
        fact["kind"] = "table"
        fact["entity"] = find_first(ENTITIES, described) or find_first(ENTITIES, title)
        fact["variable"] = find_first(VARIABLES, described) or find_first(VARIABLES, title)
        fact["year"] = next(iter(fact["years"])) if len(fact["years"]) == 1 else None
        fact["value"] = re.match(r"-?\d+(?:\.\d+)?", fact["text"].rsplit(": ", 1)[1]).group()
        fact["is_forecast"] = "(forecast)" in fact["text"]
        fact["revised_from"] = None
        # A bracketed forecast in the same cell is recorded alongside the outturn.
        match = re.search(r"fall-2011 forecast (-?\d+(?:\.\d+)?)", fact["text"])
        fact["forecast_value"] = match.group(1) if match else None
    return facts


def prose_facts(chunk):
    """Single-value sentences from the supplement and the briefing, such as "X's output growth
    slowed to 7.8% in 2012", and stated revisions. DOC1 is excluded: its narrative quotes the
    same numbers as its own tables, often as period averages."""
    if chunk["doc_id"] == "DOC1" or chunk["trust_level"] == "low":
        return []
    facts = []
    # "U.S." must not end a sentence; a "since 1999" clause is context, not a second data point.
    text = chunk["text"].replace("\n", " ").replace("U.S.", "US")
    for sentence in SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        core = SINCE_YEAR.sub("", sentence)
        entity, variable = find_first(ENTITIES, sentence), find_first(VARIABLES, sentence)
        years = set(YEAR.findall(core))
        if not (entity and variable and years):
            continue
        fact = {"kind": "prose", "text": sentence, "doc_id": chunk["doc_id"], "page": chunk["page"], "figure_id": "",
                "trust_level": chunk["trust_level"], "entity": entity, "variable": variable, "is_forecast": False,
                "forecast_value": None, "revised_from": None, "years": years, "ranges": []}
        revised = REVISED.search(sentence)
        numbers = PERCENT.findall(sentence)
        if revised:
            # The year being revised comes first; a later date is when the revision happened.
            fact.update(year=sorted(years)[0], value=revised.group(2), revised_from=revised.group(1))
        elif len(years) == 1 and len(numbers) == 1 and not RANGE_WORDS.search(core):
            fact.update(year=next(iter(years)), value=numbers[0].rstrip("%"))
        else:
            continue
        facts.append(fact)
    return facts


def question_key(question, history=()):
    """(entity, variable, year) a question asks about; a follow-up inherits what it omits."""
    entity = find_first(ENTITIES, question)
    # "Why did X's economy shrink?" is a claim about its output.
    variable = find_first(VARIABLES, question) or ("output growth" if ECONOMY.search(question) else None)
    years = YEAR.findall(question)
    for earlier in reversed(list(history)):
        entity = entity or find_first(ENTITIES, earlier)
        variable = variable or find_first(VARIABLES, earlier)
        years = years or YEAR.findall(earlier)
    if entity and variable and len(set(years)) == 1:
        return entity, variable, years[0]
    return None


def resolve_lookup(key, all_facts):
    """Apply the manifest's precedence to every value recorded under one key."""
    members = [f for f in all_facts if (f.get("entity"), f.get("variable"), f.get("year")) == key]
    if not members:
        return None
    revision = next((f for f in members if f["revised_from"]), None)
    actual = [f for f in members if not f["is_forecast"] and not f["revised_from"]]
    if revision:
        actual = [f for f in actual if float(f["value"]) == float(revision["value"])]
    forecast = next((f["forecast_value"] for f in members if f["forecast_value"]), None)
    forecasts = [f for f in members if f["is_forecast"]]
    if forecast is None and forecasts and actual:
        forecast = forecasts[0]["value"]
    values = {}
    for fact in actual:
        values.setdefault(float(fact["value"]), fact)
    conflict = revision is None and len(values) > 1 and len({f["doc_id"] for f in values.values()}) > 1
    ranked = sorted(values.values(), key=lambda f: {"high": 0, "medium": 1}.get(f["trust_level"], 2))
    return {"members": members, "conflict": conflict, "revision": revision, "values": ranked,
            "forecast": forecast, "forecast_facts": forecasts}


def describe_source(fact):
    return "{} {}".format(fact["doc_id"], fact["figure_id"] or "p{}".format(fact["page"]))


def lookup_note(key, lookup):
    """One authoritative line for the prompt, and the answer used if the model drops the value."""
    entity, variable, year = key
    subject = "{} {} in {}".format(entity, variable, year)
    if lookup["conflict"]:
        parts = ", ".join("{} ({} trust) reports {}".format(describe_source(f), f["trust_level"], f["value"])
                          for f in lookup["values"])
        return "Sources disagree on {}: {}. Prefer {}.".format(subject, parts, lookup["values"][0]["value"])
    if lookup["revision"]:
        r = lookup["revision"]
        return "{}: {}, revised from {} ({}). The revision supersedes the earlier value.".format(
            subject, r["value"], r["revised_from"], describe_source(r))
    if lookup["values"] and lookup["forecast"]:
        f = lookup["values"][0]
        return "{}: {} outturn ({}), replacing the {} forecast.".format(
            subject, f["value"], describe_source(f), lookup["forecast"])
    if lookup["values"]:
        f = lookup["values"][0]
        return "{}: {} ({}).".format(subject, f["value"], describe_source(f))
    f = lookup["forecast_facts"][0]
    return "{}: {} forecast ({}); no outturn is available.".format(subject, f["value"], describe_source(f))


FALLING = re.compile(r"\b(contract\w*|shr[aiu]nk\w*|declin\w*|fell|fall\w*|drop\w*|recession)\b", re.IGNORECASE)
RISING = re.compile(r"\b(grew|grow\w*|expand\w*|rose|rise\w*|increas\w*)\b", re.IGNORECASE)
PRESUPPOSES = re.compile(r"\b(why|how did|since|given that|as|because|what (made|caused|led|drove))\b", re.IGNORECASE)


def direction_conflict(question, key, lookup):
    """A correction when the question presupposes output moved one way and the data shows the
    other. Only growth is checked: its sign is what such a claim is about."""
    entity, variable, year = key
    if variable != "output growth" or not PRESUPPOSES.search(question):
        return None
    value = float(lookup["values"][0]["value"])
    wrong = (FALLING.search(question) and value > 0) or (RISING.search(question) and not FALLING.search(question) and value < 0)
    if not wrong:
        return None
    direction = "grew" if value > 0 else "fell"
    return "False premise: the documents show {} output {} {}% in {} ({}).".format(
        entity, direction, abs(value), year, describe_source(lookup["values"][0]))
