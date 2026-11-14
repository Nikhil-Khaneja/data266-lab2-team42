"""Gradio demo: shows every agent's output for one request.

Local:   python -m p3.app            (add --share for a public link, e.g. on Colab)
Notebook: from p3.app import build_demo; build_demo(orch).launch(share=True)
"""
from __future__ import annotations

import json
import os
import sys

import gradio as gr

from .config import Config
from .graph import Orchestrator

EXAMPLES = [
    ["Describe this image.", "data/images/cats.jpg"],
    ["Turn this photo into a watercolor painting.", "data/images/cats.jpg"],
    ["Turn the cats into tigers.", "data/images/cats.jpg"],
    ["Draw these two cats in Naruto anime style.", "data/images/cats.jpg"],
    ["Generate a Naruto-style ninja with a red scarf standing on a cliff at sunset.", None],
    ["Describe this image.", "data/images/inject_cats.png"],
    ["do something with this", "data/images/teddy_bears.jpg"],
]


def build_demo(orch: Orchestrator) -> gr.Blocks:
    def run(request, image_path, use_critique):
        s = orch.run(request or "", image_path, use_critique=bool(use_critique), tag="app")
        tj = s.to_json()
        status = f"### Status: `{s.status}`\n\n{s.message}\n\n*Request `{s.request_id}` - {s.total_seconds:.1f} s*"
        gallery = [(g.image_path, f"attempt {g.attempt}: seed {g.seed}, strength {g.strength}, "
                                  f"cfg {g.guidance_scale}") for g in s.generations if g.image_path]
        timing = "\n".join(f"| {t.agent} | {t.seconds:.2f} s |" for t in s.trace)
        timing = "| agent | time |\n|---|---|\n" + timing
        trace_path = os.path.join(orch.cfg.out_dir, "traces", f"{s.request_id}.json")
        return (s.final_image_path, status, tj["planner"], tj["vision"], tj["prompt"], tj["generations"],
                tj["critiques"], gallery, timing, trace_path)

    with gr.Blocks(title="DATA 266 Part 3 - Multi-Agent Image System") as demo:
        gr.Markdown("## Multimodal multi-agent system\nPlanner -> Vision -> Prompt Engineering -> Image Generation -> "
                    "Critique (max 2 revisions). Text inside images is treated as data, never as instructions.")
        with gr.Row():
            with gr.Column(scale=1):
                img = gr.Image(type="filepath", label="Input image (optional for Naruto-style text-to-image)")
                req = gr.Textbox(label="Request", lines=2, placeholder="e.g. Turn this photo into a watercolor painting")
                crit = gr.Checkbox(value=True, label="Use critique loop")
                btn = gr.Button("Run", variant="primary")
                gr.Examples(EXAMPLES, inputs=[req, img])
            with gr.Column(scale=1):
                out_img = gr.Image(label="Final image")
                status = gr.Markdown()
                timing = gr.Markdown()
        with gr.Tabs():
            with gr.Tab("1. Planner"):
                j_plan = gr.JSON()
            with gr.Tab("2. Vision"):
                j_vis = gr.JSON()
            with gr.Tab("3. Prompt Engineering"):
                j_prompt = gr.JSON()
            with gr.Tab("4. Image Generation"):
                j_gen = gr.JSON()
                gallery = gr.Gallery(label="All attempts", columns=3, height=300)
            with gr.Tab("5. Critique"):
                j_crit = gr.JSON()
            with gr.Tab("Trace file"):
                trace = gr.File(label="JSON trace")
        btn.click(run, [req, img, crit], [out_img, status, j_plan, j_vis, j_prompt, j_gen, j_crit, gallery, timing, trace])
    return demo


if __name__ == "__main__":
    build_demo(Orchestrator(Config())).launch(share="--share" in sys.argv)
