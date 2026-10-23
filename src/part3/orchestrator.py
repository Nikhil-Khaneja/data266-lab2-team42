import json
import time

from src.part3 import critique as critique_agent
from src.part3 import image_generation
from src.part3 import planner
from src.part3 import prompt_engineering
from src.part3 import vision
from src.part3.state import RequestState

MAX_REVISIONS = 2


def timed_step(state, agent_name, step_input, func, *args):
    start = time.time()
    try:
        func(*args)
        error = None
    except Exception as exc:
        error = str(exc)
    duration = time.time() - start
    step_output = error if error else "ok"
    state.log_step(agent_name, step_input, step_output, duration)
    return error


def run_request(user_request, input_image=None):
    state = RequestState(user_request, input_image)

    error = timed_step(state, "planner", user_request, planner.plan, state)
    if error:
        state.refused = True
        state.refusal_reason = "planner_error: {}".format(error)
        return state

    if state.refused:
        return state

    if state.route == "unsupported":
        state.refusal_reason = state.refusal_reason or "could_not_classify_request"
        return state

    if input_image is None:
        state.refused = True
        state.refusal_reason = "missing_or_broken_input_image"
        return state

    error = timed_step(state, "vision_understanding", "input_image", vision.understand, state)
    if error:
        state.refused = True
        state.refusal_reason = "vision_error: {}".format(error)
        return state

    if state.route == "caption_or_question":
        return state

    error = timed_step(state, "prompt_engineering", state.user_request, prompt_engineering.build_prompt, state)
    if error:
        state.refused = True
        state.refusal_reason = "prompt_engineering_error: {}".format(error)
        return state

    while True:
        error = timed_step(state, "image_generation", state.generation_prompt, image_generation.generate, state)
        if error:
            state.refused = True
            state.refusal_reason = "generation_error: {}".format(error)
            return state

        error = timed_step(state, "critique", state.generation_prompt, critique_agent.critique, state)
        if error:
            state.refused = True
            state.refusal_reason = "critique_error: {}".format(error)
            return state

        if state.critique_decision == "accept" or state.revision_count > MAX_REVISIONS:
            break

        error = timed_step(
            state, "prompt_engineering_revision", state.critique_feedback,
            prompt_engineering.build_prompt, state, state.critique_feedback
        )
        if error:
            state.refused = True
            state.refusal_reason = "revision_error: {}".format(error)
            return state

    return state


def save_trace(state, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "user_request": state.user_request,
            "route": state.route,
            "refused": state.refused,
            "refusal_reason": state.refusal_reason,
            "critique_decision": state.critique_decision,
            "revision_count": state.revision_count,
            "trace": state.trace,
        }, f, indent=2)
