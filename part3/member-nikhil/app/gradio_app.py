import gradio as gr

from src.part3.orchestrator import run_request


def handle_request(user_request, input_image):
    state = run_request(user_request, input_image)

    if state.refused:
        return "Refused: {}".format(state.refusal_reason), None, format_trace(state)

    if state.route == "caption_or_question":
        summary = "Caption: {}\nObjects: {}\nText in image: {}\nQ&A: {}".format(
            state.vision_caption, state.vision_objects, state.vision_text_in_image, state.vision_qa
        )
        return summary, None, format_trace(state)

    summary = "Route: {}\nPrompt: {}\nCritique: {} ({})\nRevisions: {}".format(
        state.route, state.generation_prompt, state.critique_decision, state.critique_feedback,
        state.revision_count
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
        submit_button = gr.Button("Run")
        summary_box = gr.Textbox(label="Summary", lines=6)
        trace_box = gr.Textbox(label="Agent trace", lines=10)

        submit_button.click(
            fn=handle_request,
            inputs=[request_box, input_image],
            outputs=[summary_box, output_image, trace_box],
        )
    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
