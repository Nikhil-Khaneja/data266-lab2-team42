"""Generic vocabulary for economies, variables and years.

Used to read entity / variable / year mentions out of table labels, sentences
and questions. Nothing here refers to a specific question.
"""
import re

# canonical entity -> surface forms (lower case, matched on word boundaries)
ENTITIES = {
    "United States": ["united states", "u.s.", "u.s", "america", "american"],
    "Euro area": ["euro area", "euro zone", "eurozone", "euro-area", "euroland", "europe", "european economy"],
    "China": ["china", "chinese"],
    "World": ["world", "global economy", "world economy"],
    "Advanced economies": ["advanced economies", "advanced economy", "advanced-economy",
                           "advanced countries", "advanced"],
    "Emerging and developing economies": ["emerging and developing economies",
                                          "emerging and developing", "emerging economies",
                                          "developing economies", "emerging"],
}

# canonical variable -> surface forms
VARIABLES = {
    "output growth": ["output growth", "gdp growth", "growth rate of output", "output growth rate",
                      "growth", "grew", "grow", "output", "gdp", "economy", "expanded", "contracted",
                      "contract", "shrank", "shrink", "recession"],
    "unemployment rate": ["unemployment", "jobless", "unemployed", "jobs", "out of work", "out of a job", "joblessness"],
    "inflation rate": ["inflation", "gdp deflator", "price level"],
    "budget balance": ["budget balance", "budget deficit", "deficit", "budget surplus", "surplus"],
    "revenues": ["revenues", "revenue", "receipts"],
    "spending": ["spending", "outlays", "expenditure", "expenditures"],
    "labour force participation": ["participation"],
}

# words that mark a decline of output (for false-premise checks)
DECLINE_WORDS = ["contract", "contracted", "contraction", "shrink", "shrank", "shrunk",
                 "fell", "fall", "decline", "declined", "negative growth", "recession"]
RISE_WORDS = ["keep rising", "kept rising", "rose", "rise", "rising", "increase", "increased",
              "went up", "go up", "climb", "climbed"]
FALL_WORDS = ["fell", "fall", "falling", "drop", "dropped", "decline", "declined", "decrease",
              "decreased", "went down", "go down"]

LABEL_RE = re.compile(r"\b(Figure|Table)\s+([12]|[A-C])\s*-\s*(\d+)(\s*\(corrected\))?", re.I)
_LID = r"(?:[12]|[A-C])\s*-\s*\d+(?:\s*\(corrected\))?"
PLURAL_LABEL_RE = re.compile(rf"\b(Figures|Tables)\s+({_LID}(?:\s*(?:,|and|&|or|vs\.?|versus)\s*{_LID})+)", re.I)
LABEL_ID_RE = re.compile(r"([12]|[A-C])\s*-\s*(\d+)(\s*\(corrected\))?", re.I)
YEAR_RE = re.compile(r"\b(19[5-9]\d|20[0-4]\d)\b")
RANGE_RE = re.compile(r"\b((?:19|20)\d\d)\s*[-–—]\s*((?:19|20)\d\d)\b")
NUM_RE = re.compile(r"(?<![\w.])[-−]?\d+(?:,\d{3})*(?:\.\d+)?")


def norm_label(kind, part, num, corrected=None):
    lab = f"{kind.capitalize()} {part.upper()}-{int(num)}"
    return lab + (" (corrected)" if corrected else "")


def find_labels(text):
    """Exact Figure/Table labels mentioned in text, in order, deduplicated."""
    found = []
    for m in LABEL_RE.finditer(text or ""):
        found.append((m.start(), norm_label(m.group(1), m.group(2), m.group(3), m.group(4))))
    # plural forms: "Figures 1-1 and 1-2", "Tables 1-2, 1-3 and A-2"
    for m in PLURAL_LABEL_RE.finditer(text or ""):
        kind = m.group(1)[:-1]
        for p in LABEL_ID_RE.finditer(m.group(2)):
            found.append((m.start() + p.start(), norm_label(kind, p.group(1), p.group(2), p.group(3))))
    out = []
    for _, lab in sorted(found):
        if lab not in out:
            out.append(lab)
    return out


def _has(text_l, form):
    return re.search(r"(?<![a-z])" + re.escape(form) + r"(?![a-z])", text_l) is not None


def find_entities(text):
    t = " " + (text or "").lower() + " "
    found = []
    if re.search(r"\bUS\b|\bUSA\b", text or ""):      # upper-case "US" only, not the pronoun
        found.append("United States")
    # longest forms first so "emerging and developing economies" beats "emerging"
    pairs = sorted(((f, e) for e, fs in ENTITIES.items() for f in fs), key=lambda p: -len(p[0]))
    used = t
    for form, ent in pairs:
        if _has(used, form):
            if ent not in found:
                found.append(ent)
            used = re.sub(r"(?<![a-z])" + re.escape(form) + r"(?![a-z])", " ", used)
    return _by_position(found, ENTITIES, t)


WEAK_FORMS = {"output", "gdp", "economy"}   # e.g. "as a share of GDP" is not about growth


def find_variables(text):
    t = (text or "").lower()
    found, strong = [], set()
    pairs = sorted(((f, v) for v, fs in VARIABLES.items() for f in fs), key=lambda p: -len(p[0]))
    used = t
    for form, var in pairs:
        if _has(used, form):
            if var not in found:
                found.append(var)
            if form not in WEAK_FORMS:
                strong.add(var)
            used = re.sub(r"(?<![a-z])" + re.escape(form) + r"(?![a-z])", " ", used)
    # a variable matched only through a weak form is dropped when another variable is present
    if len(found) > 1:
        found = [v for v in found if v in strong] or found
    return _by_position(found, VARIABLES, t)


def _by_position(found, table, t):
    def first(name):
        pos = [m.start() for f in table[name] for m in
               re.finditer(r"(?<![a-z])" + re.escape(f) + r"(?![a-z])", t)]
        return min(pos) if pos else 10 ** 6
    return sorted(found, key=first)


COUNTRIES = ["India", "Japan", "Brazil", "Russia", "Mexico", "Canada", "Australia", "Korea", "South Korea",
              "Indonesia", "Turkey", "Saudi Arabia", "Argentina", "South Africa", "Nigeria", "Egypt", "Iran",
              "Pakistan", "Bangladesh", "Vietnam", "Thailand", "Malaysia", "Philippines", "Singapore",
              "United Kingdom", "UK", "Britain", "Switzerland", "Sweden", "Norway", "Denmark", "Poland",
              "Germany", "France", "Italy", "Spain", "Greece", "Portugal", "Ireland", "Netherlands", "Austria",
              "Belgium", "Finland", "Luxembourg", "Estonia", "Slovakia", "Slovenia", "Malta", "Cyprus",
              "Chile", "Colombia", "Peru", "Venezuela", "Israel", "New Zealand", "Taiwan", "Hong Kong", "Ukraine"]


def find_countries(text):
    return [c for c in COUNTRIES if re.search(r"\b" + re.escape(c) + r"(?:'s|’s)?\b", text or "")]


SEASON_YEAR_RE = re.compile(r"\b(?:fall|autumn|spring|summer|winter|mid)[-\s]((?:19|20)\d\d)\b", re.I)


def find_years(text):
    t = SEASON_YEAR_RE.sub(" ", text or "")      # "the fall-2011 forecast" is a vintage, not the year asked about
    ranges = [(int(a), int(b)) for a, b in RANGE_RE.findall(t)]
    in_range = set()
    for a, b in ranges:
        in_range.update({a, b})
    years = []
    for y in YEAR_RE.findall(t):
        y = int(y)
        if y not in years and y not in in_range:
            years.append(y)
    return years, ranges


def parse_number(s):
    s = (s or "").strip().replace("−", "-").replace("–", "-").replace(",", "")
    s = re.sub(r"^-\s+", "-", s)
    try:
        return float(s)
    except ValueError:
        return None


def numbers_in(text):
    vals = []
    for m in NUM_RE.finditer((text or "").replace("−", "-")):
        v = parse_number(m.group(0))
        if v is not None:
            vals.append(v)
    return vals
