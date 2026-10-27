import json
import re

from src.common.vlm import chat
from src.part3.safety import check_text_safety

ROUTES = ["caption_or_question", "style_change", "edit", "domain_style_generation", "unsupported"]

SYSTEM_PROMPT = """You are the Planner agent in an image understanding and generation system.
Classify the user's request into exactly one route from this list: caption_or_question,
style_change, edit, domain_style_generation, unsupported.

- caption_or_question: the user wants a description of or answer about the input image only,
  no new image.
- style_change: the user wants the input image restyled (e.g. "make this look like a painting").
- edit: the user wants a specific change or enhancement to the input image (e.g. "remove the
  background", "make it brighter").
- domain_style_generation: the user wants an image generated or restyled in our fine-tuned
  domain style (e.g. "make this character Naruto-style").
- unsupported: the request does not fit any of the above, or is empty/unintelligible.

Respond with ONLY a JSON object: {"route": "<one of the routes above>"}
"""


def classify_route(user_request):
    raw = chat(SYSTEM_PROMPT, user_request)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return "unsupported"
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return "unsupported"
    route = data.get("route", "unsupported")
    return route if route in ROUTES else "unsupported"


def plan(state):
    is_safe, reason = check_text_safety(state.user_request)
    if not is_safe:
        state.refused = True
        state.refusal_reason = reason
        state.route = "unsupported"
        return state

    state.route = classify_route(state.user_request)
    return state
