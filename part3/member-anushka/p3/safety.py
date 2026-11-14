"""Guardrails used by the Planner (input) and the Critique agent (output).

These rules are general policies, not rules for specific test cases:
  * NSFW / gore requests            -> refuse
  * deceptive edits of real people  -> refuse (named public figures, face swaps,
                                       fabricated events presented as real)
  * instructions written in images  -> refuse; image text is data, never a command
"""
from __future__ import annotations

import re

_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t",
                       "@": "a", "$": "s", "!": "i"})


def normalize(text: str) -> str:
    """Lower-case, undo leetspeak and join letters split by dots/dashes/spaces (n.u.d.e -> nude)."""
    t = text.lower().translate(_LEET)
    t = re.sub(r"(?<=\w)[.\-_*'`](?=\w)", "", t)                       # n.u.d.e -> nude
    t = re.sub(r"\b(?:\w ){2,}\w\b", lambda m: m.group(0).replace(" ", ""), t)  # n u d e -> nude
    return re.sub(r"\s+", " ", t).strip()


# ---- NSFW / violent content -------------------------------------------------
_NSFW_WORDS = [r"nude", r"nudes", r"naked", r"nsfw", r"porn\w*", r"sex", r"sexy", r"sexual\w*",
               r"explicit", r"topless", r"bottomless", r"undress\w*", r"lingerie", r"erotic\w*",
               r"genital\w*", r"nipples?", r"strip(?:ping)? (?:her|him|them|off)",
               r"gore", r"gory", r"blood\w*", r"decapitat\w*", r"dismember\w*", r"severed",
               r"mutilat\w*", r"corpse", r"behead\w*", r"self[- ]harm", r"suicide",
               r"swastika", r"kkk"]
_NSFW_PHRASES = [r"(?:remove|take off|without) (?:her|his|their|the|all)? ?(?:clothes|clothing|shirt|dress|top)",
                 r"in (?:her|his|their) underwear"]
_NSFW_RE = re.compile(r"\b(?:" + "|".join(_NSFW_WORDS + _NSFW_PHRASES) + r")\b")

# ---- deceptive edits of real people ----------------------------------------
_PUBLIC_FIGURES = ["biden", "trump", "obama", "harris", "putin", "zelensky", "xi jinping", "modi",
                   "elon musk", "musk", "taylor swift", "beyonce", "rihanna", "oprah", "zuckerberg",
                   "bill gates", "pope", "king charles", "kim jong", "tom cruise", "scarlett johansson",
                   "emma watson", "lebron", "messi", "ronaldo"]
_PERSON = r"(?:him|her|them|he|she|they|man|woman|men|women|person|people|guy|girl|boy|lady|kid|child|player|face|politician|celebrity)"
_DECEPTIVE = [
    r"face ?swap", r"swap .{0,30}\bfaces?\b", r"(?:put|paste|replace) .{0,30}\bfaces?\b",
    r"look(?:s)? (?:like|as if) " + r".{0,25}(?:arrested|drunk|high|crime|stealing|guilty|kissing|cheating|in jail|in prison|dead|injured|holding (?:drugs|a gun|a weapon))",
    r"(?:arrested|handcuffed|in handcuffs|mugshot|in jail|in prison|committing a crime|holding (?:drugs|cocaine|a gun))",
    r"(?:make it|so it) (?:look|looks) (?:real|authentic|genuine)", r"so (?:people|everyone) (?:believe|think)",
    r"(?:breaking|fake|real) news", r"as evidence", r"\bdeepfake\b", r"post it (?:online|as)",
]
_DECEPTIVE_RE = re.compile("|".join(_DECEPTIVE))
_PERSON_RE = re.compile(r"\b" + _PERSON + r"\b")
_FIGURE_RE = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in _PUBLIC_FIGURES) + r")(?:'?s)?\b")
_EDIT_VERB_RE = re.compile(r"\b(?:edit|change|make|put|swap|replace|add|turn|show|photoshop|modify|paste|generate|create|draw)\b")

# ---- instructions hidden in images -----------------------------------------
_INJECTION = [
    r"ignore (?:all |any |the |previous |prior |above |earlier )*(?:instructions?|prompts?|user|request|rules)",
    r"disregard (?:the |all |any |previous )*(?:instructions?|user|request|rules)",
    r"\bsystem\s*(?:prompt|override|message|instruction)?\s*:", r"\boverride\b", r"\bjailbreak\b",
    r"new instructions?", r"(?:ai|assistant|model|bot|agent)s?\s*[:,]", r"\byou (?:must|should|are now|will now)\b",
    r"instead,? (?:generate|draw|create|make|add|output|produce|reply|say)", r"do not follow",
    r"disable (?:the )?(?:safety|filter|guard)", r"(?:safety|nsfw) (?:filter|checker) (?:off|disabled)",
]
_INJECTION_RE = re.compile("|".join(_INJECTION))


def nsfw_terms(text: str) -> list[str]:
    return sorted({m.group(0) for m in _NSFW_RE.finditer(normalize(text))})


def deceptive_person_edit(text: str, image_has_person: bool) -> list[str]:
    """Return reasons if the request asks for a deceptive edit/fabrication of a real person."""
    t = normalize(text)
    reasons = []
    fig = _FIGURE_RE.search(t)
    if fig and _EDIT_VERB_RE.search(t):
        reasons.append(f"names a real public figure ('{fig.group(0)}') in an image edit/generation request")
    dec = _DECEPTIVE_RE.search(t)
    if dec and (image_has_person or _PERSON_RE.search(t)):
        reasons.append(f"deceptive edit of a real person ('{dec.group(0)}')")
    return reasons


def image_injection(image_text: str) -> list[str]:
    if not image_text:
        return []
    t = normalize(image_text)
    return sorted({m.group(0).strip() for m in _INJECTION_RE.finditer(t)})


def scrub_image_text(prompt: str, image_text: str, keep_from: str = "", n: int = 3) -> str:
    """Remove phrases copied from image text into a generation prompt (image text is data).

    Any run of >= ``n`` consecutive words that also occurs in the image text is dropped, unless the
    user typed that phrase themselves (``keep_from``). Single words (e.g. 'cats') are kept.
    """
    if not image_text or image_text.strip().upper() == "NONE":
        return prompt
    norm = lambda s: re.findall(r"[a-z0-9]+", s.lower())
    img_w, user = norm(image_text), " ".join(norm(keep_from))
    grams = {" ".join(img_w[i:i + n]) for i in range(len(img_w) - n + 1)}
    grams = {g for g in grams if g not in user}
    toks = prompt.split()
    keys = [(re.sub(r"[^a-z0-9]", "", t.lower())) for t in toks]
    drop = set()
    for i in range(len(toks) - n + 1):
        if " ".join(keys[i:i + n]) in grams:
            drop.update(range(i, i + n))
    if not drop:
        return prompt
    return re.sub(r"\s+,", ",", " ".join(t for i, t in enumerate(toks) if i not in drop)).strip(" ,")
