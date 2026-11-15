"""Hugging Face Space entry point for the Part 3 multi-agent demo.

Free Spaces run on CPU, so the system uses its CPU settings (SmolVLM-500M, 384 px, 12 diffusion steps).
On a GPU Space the same code switches to Qwen2.5-VL-3B at 512 px automatically.
"""
import os
import threading

os.environ.setdefault("P3_OUT_DIR", "/tmp/p3_outputs")

from p3 import Config, Orchestrator  # noqa: E402
from p3.app import build_demo  # noqa: E402

orch = Orchestrator(Config())


def _warm_up() -> None:
    # load the models in the background so the first request does not wait for downloads
    try:
        orch.hub.vlm, orch.hub.clip, orch.hub.diffusion
        print("models loaded; LoRA loaded:", orch.hub.diffusion.lora_loaded)
    except Exception as e:  # the app still starts; the failing agent reports the error per request
        print("warm-up failed:", type(e).__name__, e)


threading.Thread(target=_warm_up, daemon=True).start()
demo = build_demo(orch)
demo.queue(max_size=8).launch()
