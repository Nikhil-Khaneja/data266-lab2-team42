"""Draws docs/p3_architecture.png (architecture diagram for the report)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
fig, ax = plt.subplots(figsize=(13, 6.2))
ax.set_xlim(0, 13); ax.set_ylim(0, 6.2); ax.axis("off")

AG = "#dbe8f7"; ST = "#f3f3f3"; RF = "#fde2e2"; OK = "#e1f3e1"


def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12", fc=color, ec="#444", lw=1.2))
    ax.text(x + w / 2, y + h - 0.22, title, ha="center", va="top", fontsize=10.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=6.9, linespacing=1.3)


def arrow(a, b, text="", color="#333", style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=13, color=color, lw=1.3,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.12, text, ha="center", fontsize=7.5, color=color)


box(0.1, 3.3, 1.6, 1.4, "User", "request text\n+ optional image", ST)
box(2.1, 3.0, 2.0, 2.0, "1 Planner", "validate image (empty, broken,\ntiny, blank)\nNSFW / real-person checks\nOCR scan: text in image\n= data, never instructions\nclassify + route", AG)
box(4.5, 3.0, 2.0, 2.0, "2 Vision", "Qwen2.5-VL-3B\ncaption, objects,\nvisible text,\n>= 2 VQA answers", AG)
box(6.9, 3.0, 2.0, 2.0, "3 Prompt Eng.", "request + caption ->\ngeneration prompt\nnegative prompt (NSFW)\nscrub image text\napply critic revision", AG)
box(9.3, 3.0, 2.0, 2.0, "4 Generation", "style/enhance: SD1.5 img2img\nedit: InstructPix2Pix\ndomain: SD1.5 + Part 2\nNaruto LoRA (img2img/txt2img)\nrecord model/prompt/seed/steps", AG)
box(9.3, 0.3, 2.0, 2.1, "5 Critique", "CLIP(image, prompt)\nCLIP preservation vs input\nVLM YES/NO judgement\nSD safety checker\naccept / revise + instruction", AG)
box(11.6, 3.3, 1.3, 1.4, "Output", "final image\n+ JSON trace", OK)
box(2.1, 0.4, 2.0, 1.4, "Refuse / ask", "unsafe -> refused\nbroken -> clear error\nambiguous -> clarify", RF)
box(4.5, 0.4, 2.0, 1.4, "Answer", "caption / question\nrequests stop here", OK)
box(6.3, 5.35, 3.2, 0.75, "Shared typed AgentState (pydantic)", "", ST)
ax.text(7.9, 5.55, "planner | vision | prompt | generations[] | critiques[] | trace[] | status", ha="center", fontsize=7.3)

arrow((1.7, 4.0), (2.1, 4.0))
arrow((4.1, 4.0), (4.5, 4.0), "safe")
arrow((6.5, 4.0), (6.9, 4.0), "generate")
arrow((8.9, 4.0), (9.3, 4.0))
arrow((10.3, 3.0), (10.3, 2.4))
arrow((11.3, 1.6), (12.25, 3.3), "accept", color="#2a7a2a", rad=0.25)
arrow((9.3, 1.3), (7.9, 3.0), "revise (max 2)", color="#b03030", rad=-0.25, ls="--")
arrow((3.1, 3.0), (3.1, 1.8), "stop", color="#b03030")
arrow((5.5, 3.0), (5.5, 1.8), "caption/question", color="#2a7a2a")
arrow((3.2, 5.0), (6.9, 4.9), "Naruto txt2img (no image)", color="#555", rad=-0.18, ls=":")
ax.set_title("Part 3: five-agent image understanding & generation system", fontsize=12, weight="bold")
out = ROOT / "docs" / "p3_architecture.png"
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print("wrote", out)
