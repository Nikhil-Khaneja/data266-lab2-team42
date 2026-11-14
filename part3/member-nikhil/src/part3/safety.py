import re

UNSAFE_PATTERNS = [
    r"\bnude\b", r"\bnaked\b", r"\bnsfw\b", r"\bsex(ual)?\b", r"\bporn", r"\bexplicit\b",
    r"\bgore\b", r"\bviolen(t|ce)\b", r"\bgraphic\b.*\b(violen|gore|blood)", r"\bkill\b",
    r"\bweapon\b", r"\bself[- ]harm\b",
]
UNSAFE_REGEX = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)

DECEPTIVE_EDIT_PATTERNS = [
    r"\b(real|this) (person|politician|celebrity|public figure)\b",
    r"\b(politician|celebrity|public figure)\b", r"\bmake (him|her|them) look like\b",
    r"\bdeepfake\b", r"\bfake (photo|image) of\b", r"\bcompromising\b", r"\bembarrassing\b",
    r"\bdidn'?t do\b",
]
DECEPTIVE_EDIT_REGEX = re.compile("|".join(DECEPTIVE_EDIT_PATTERNS), re.IGNORECASE)

# Common prompt-injection phrasings: an instruction aimed at the AI, a claimed higher authority,
# or an attempt to switch off rules. Matched against text read from an image or a document.
INJECTION_PATTERNS = [
    r"\bignore\b.{0,40}\b(instructions?|rules|guidelines|safety|restrictions)\b",
    r"\bdisregard\b.{0,40}\b(instructions?|rules|above|previous)\b",
    r"\bsystem prompt\b", r"^\s*system\s*:", r"\byou are now\b", r"\bnew instructions?\b",
    r"\binstructions? (to|for) (ai|the assistant|assistants|the model|language models?)\b",
    r"\b(ai|assistants?|chatbots?|models?) reading this\b",
    r"\bfollow (any|all|these|the following|my) instructions?\b",
    r"\bunrestricted (mode|access)\b", r"\b(developer|admin|jailbreak|god) mode\b",
    r"\bmessage from the (developer|admin|system)\b", r"\boverride\b.{0,30}\b(rules|safety|instructions?)\b",
]
INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE | re.MULTILINE)


def check_text_safety(text):
    if not text:
        return True, None
    if UNSAFE_REGEX.search(text):
        return False, "unsafe_content"
    if DECEPTIVE_EDIT_REGEX.search(text):
        return False, "deceptive_edit_of_real_person"
    return True, None


def check_for_injection(text):
    if not text:
        return False
    return bool(INJECTION_REGEX.search(text))
