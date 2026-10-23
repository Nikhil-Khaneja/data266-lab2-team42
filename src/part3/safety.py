import re

UNSAFE_PATTERNS = [
    r"\bnude\b", r"\bnaked\b", r"\bnsfw\b", r"\bsex(ual)?\b", r"\bporn", r"\bexplicit\b",
    r"\bgore\b", r"\bviolent\b", r"\bkill\b", r"\bweapon\b", r"\bself[- ]harm\b",
]
UNSAFE_REGEX = re.compile("|".join(UNSAFE_PATTERNS), re.IGNORECASE)

DECEPTIVE_EDIT_PATTERNS = [
    r"\breal (person|politician|celebrity)\b", r"\bmake (him|her|them) look like\b",
    r"\bdeepfake\b", r"\bfake (photo|image) of\b",
]
DECEPTIVE_EDIT_REGEX = re.compile("|".join(DECEPTIVE_EDIT_PATTERNS), re.IGNORECASE)

INJECTION_PATTERNS = [
    r"ignore (the )?(previous|above) instructions", r"disregard (the )?(previous|above)",
    r"system prompt", r"you are now", r"new instructions:",
]
INJECTION_REGEX = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


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
