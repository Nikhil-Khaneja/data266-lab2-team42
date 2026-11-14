import os
import sys

import gradio as gr

# Run as `python app/gradio_app.py` from part3/member-nikhil/: make `src` importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.part3.orchestrator import run_request


def handle_request(user_request, input_image, use_critique=True):
    state = run_request(user_request, input_image, use_critique=use_critique)

    if state.refused:
        return "Refused: {}".format(state.refusal_reason), None, format_trace(state)

    if state.route == "caption_or_question":
        summary = "Caption: {}\nObjects: {}\nText in image: {}\nQ&A: {}".format(
            state.vision_caption, state.vision_objects, state.vision_text_in_image, state.vision_qa
        )
        return summary, None, format_trace(state)

    summary = "Route: {}\nPrompt: {}\nModel: {}\nCritique decisions: {}\nResult: {} after {} revision(s)\nLast feedback: {}".format(
        state.route, state.generation_prompt, state.generation_model,
        ", ".join(state.critique_history) or "(critique off)", state.final_status,
        state.revision_count, state.critique_feedback
    )
    return summary, state.generated_image, format_trace(state)


def format_trace(state):
    lines = []
    for step in state.trace:
        lines.append("{}: {} ({:.2f}s)".format(step["agent"], step["output"], step["duration_seconds"]))
    return "\n".join(lines)


def build_app():
    with gr.Blocks() as demo:
        gr.Markdown("# Team 42 — Multimodal Multi-Agent Image System")
        with gr.Row():
            input_image = gr.Image(type="pil", label="Input image")
            output_image = gr.Image(type="pil", label="Generated image")
        request_box = gr.Textbox(label="Request")
        critique_box = gr.Checkbox(value=True, label="Critique loop (max 2 revisions)")
        submit_button = gr.Button("Run")
        summary_box = gr.Textbox(label="Summary", lines=6)
        trace_box = gr.Textbox(label="Agent trace", lines=10)

        submit_button.click(
            fn=handle_request,
            inputs=[request_box, input_image, critique_box],
            outputs=[summary_box, output_image, trace_box],
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
