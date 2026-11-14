import json
import os
import time

from src.part3 import critique as critique_agent
from src.part3 import image_generation
from src.part3 import planner
from src.part3 import prompt_engineering
from src.part3 import vision
from src.part3.state import RequestState

MAX_REVISIONS = 2


def summarize_output(state, agent_name):
    if agent_name == "planner":
        return {"route": state.route, "refused": state.refused, "refusal_reason": state.refusal_reason,
                "image_text": state.image_text}
    if agent_name == "vision_understanding":
        return {"caption": state.vision_caption, "objects": state.vision_objects,
                "text_in_image": state.vision_text_in_image, "qa": state.vision_qa}
    if agent_name in ("prompt_engineering", "prompt_engineering_revision"):
        return {"generation_prompt": state.generation_prompt}
    if agent_name == "image_generation":
        return {"mode": state.generation_mode, "model": state.generation_model,
                "seed": state.generation_seed, "steps": state.generation_steps,
                "safety_flagged": state.output_flagged}
    if agent_name == "critique":
        return {"score": state.critique_score, "decision": state.critique_decision,
                "feedback": state.critique_feedback}
    return "ok"


def timed_step(state, agent_name, step_input, func, *args):
    start = time.time()
    try:
        func(*args)
        error = None
    except Exception as exc:
        error = str(exc)
    duration = time.time() - start
    step_output = {"error": error} if error else summarize_output(state, agent_name)
    state.log_step(agent_name, step_input, step_output, duration)
    return error


def stop(state, reason):
    state.refused = True
    state.refusal_reason = reason
    state.final_status = "refused"
    return state


def run_request(user_request, input_image=None, use_critique=True):
    """Planner -> Vision -> Prompt Engineering -> Generation -> Critique, revising at most
    MAX_REVISIONS times. use_critique=False is the baseline: one generation, no critique loop."""
    state = RequestState(user_request, input_image, use_critique=use_critique)

    error = timed_step(state, "planner", user_request, planner.plan, state)
    if error:
        return stop(state, "planner_error: {}".format(error))
    if state.refused:
        return stop(state, state.refusal_reason)
    if state.route == "unsupported":
        return stop(state, state.refusal_reason or "could_not_classify_request")
    if input_image is None:
        return stop(state, "missing_or_broken_input_image")

    error = timed_step(state, "vision_understanding", "input_image", vision.understand, state)
    if error:
        return stop(state, "vision_error: {}".format(error))
    if state.route == "caption_or_question":
        state.final_status = "answered"
        return state

    error = timed_step(state, "prompt_engineering", state.user_request, prompt_engineering.build_prompt, state)
    if error:
        return stop(state, "prompt_engineering_error: {}".format(error))

    while True:
        error = timed_step(state, "image_generation", state.generation_prompt, image_generation.generate, state)
        if error:
            return stop(state, "generation_error: {}".format(error))
        if not use_critique:
            state.final_status = "no_critique"
            return state

        error = timed_step(state, "critique", state.generation_prompt, critique_agent.critique, state)
        if error:
            return stop(state, "critique_error: {}".format(error))
        if state.critique_decision == "accept":
            state.final_status = "accepted"
            return state
        if state.revision_count >= MAX_REVISIONS:
            # Still "revise" after the last allowed revision: keep the last image, say so honestly.
            state.final_status = "revision_limit_reached"
            return state

        state.revision_count += 1
        error = timed_step(
            state, "prompt_engineering_revision", state.critique_feedback,
            prompt_engineering.build_prompt, state, state.critique_feedback
        )
        if error:
            return stop(state, "revision_error: {}".format(error))


def save_trace(state, output_path, image_dir=None):
    """JSON trace of every agent step; with image_dir, the input and final output images too."""
    images = {}
    if image_dir:
        os.makedirs(image_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(output_path))[0]
        for kind, image in (("input", state.input_image), ("output", state.generated_image)):
            if image is not None:
                images[kind] = os.path.join(image_dir, "{}_{}.png".format(stem, kind))
                image.save(images[kind])
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({
            "user_request": state.user_request,
            "use_critique": state.use_critique,
            "route": state.route,
            "refused": state.refused,
            "refusal_reason": state.refusal_reason,
            "image_text": state.image_text,
            "final_status": state.final_status,
            "critique_history": state.critique_history,
            "revision_count": state.revision_count,
            "generation": {"mode": state.generation_mode, "model": state.generation_model,
                           "prompt": state.generation_prompt, "seed": state.generation_seed,
                           "steps": state.generation_steps, "safety_flagged": state.output_flagged},
            "images": images,
            "trace": state.trace,
        }, f, indent=2)
