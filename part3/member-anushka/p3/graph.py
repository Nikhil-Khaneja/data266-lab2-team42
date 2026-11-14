"""Orchestrator: plain-Python state machine over the shared AgentState.

user request -> Planner (input safety) -> Vision -> Prompt Engineering -> Image Generation
             -> Critique (output safety) -> accept | revise (back to Prompt Engineering, max 2x)
Caption/question requests stop after Vision; unsafe requests stop at the Planner.
Every agent call is timed and written to a JSON trace.
"""
from __future__ import annotations

import json
import os
import time
import traceback
from typing import Callable, Optional

from . import agents
from .config import Config
from .models import ModelHub
from .state import AgentState, TraceEvent


def _summ(d: dict, limit: int = 400) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, str) and len(v) > limit:
            v = v[:limit] + "..."
        out[k] = v
    return out


class Orchestrator:
    def __init__(self, cfg: Optional[Config] = None, hub: Optional[ModelHub] = None):
        self.cfg = cfg or Config()
        self.hub = hub or ModelHub(self.cfg)

    def _step(self, name: str, fn: Callable, state: AgentState, inp: dict) -> bool:
        t0 = time.time()
        try:
            out = fn(state, self.hub)
            ok = True
        except Exception as e:  # never crash the app: turn failures into a clear message
            ok = False
            out = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc(limit=3)}
            state.status = "error"
            if name == "image_generation":
                msg = "Image generation failed"
                if "out of memory" in str(e).lower():
                    msg += " (GPU out of memory - try a smaller image or restart the runtime)"
                elif "LoRA" in str(e):
                    msg += " (the Part 2 Naruto LoRA could not be loaded)"
                state.message = f"{msg}: {type(e).__name__}. No image was produced."
            else:
                state.message = f"The {name.replace('_', ' ')} agent failed ({type(e).__name__}: {e}). No result was produced."
        state.trace.append(TraceEvent(agent=name, started=t0, seconds=round(time.time() - t0, 3),
                                      input=_summ(inp), output=_summ(out)))
        return ok

    def run(self, request: str, image_path: Optional[str] = None, use_critique: Optional[bool] = None,
            tag: str = "") -> AgentState:
        uc = self.cfg.use_critique if use_critique is None else use_critique
        s = AgentState(user_request=request or "", image_path=image_path, use_critique=uc)
        if tag:
            s.request_id = f"{tag}_{s.request_id[:6]}"

        self._step("planner", agents.planner, s, {"request": request, "image_path": image_path, "use_critique": uc})
        route = s.planner.route if s.planner else []
        if s.status == "running" and "vision" in route:
            self._step("vision", agents.vision, s, {"image_path": image_path, "intent": s.planner.intent})

        if s.status == "running" and "image_generation" in route:
            while True:
                if not self._step("prompt_engineering", agents.prompt_engineering, s,
                                  {"request": request, "caption": s.vision.caption if s.vision else None,
                                   "revision": s.critiques[-1].revision_instruction if s.critiques else None}):
                    break
                if not self._step("image_generation", agents.image_generation, s, s.prompt.model_dump()):
                    break
                g = s.generations[-1]
                if not uc:
                    # no critique loop: only the mandatory output safety check
                    if g.nsfw_flagged:
                        s.status, s.message = "refused", "The generated image failed the safety check and was withheld."
                    else:
                        s.status, s.final_image_path = "completed", g.image_path
                        s.message = f"Generated with {g.model} (mode {g.mode}, seed {g.seed}, {g.steps} steps)."
                    break
                if not self._step("critique", agents.critique, s, {"attempt": g.attempt, "image_path": g.image_path,
                                                                   "prompt": s.prompt.prompt}):
                    break
                c = s.critiques[-1]
                if c.decision == "revise":
                    s.revisions += 1
                    continue
                if c.decision == "reject_unsafe" and not any(cc.safe for cc in s.critiques):
                    s.status = "refused"
                    s.message = "Every attempt failed the output safety check, so no image is returned."
                    break
                # accept (possibly after running out of revisions): keep the best safe attempt
                if c.decision == "accept" and c.revision_instruction is None:   # genuine accept: return this attempt
                    best_c, best_g = c, g
                else:
                    safe = [(cc, gg) for cc, gg in zip(s.critiques, s.generations) if cc.safe and gg.image_path]
                    best_c, best_g = max(safe, key=lambda x: (x[0].vlm_verdict == "YES", x[0].clip_score))
                s.status, s.final_image_path = "completed", best_g.image_path
                if c.decision == "accept" and c.revision_instruction is None:
                    note = "accepted by the critic"
                elif c.decision == "reject_unsafe":
                    note = "the last attempt failed the safety check; returning the best earlier safe attempt"
                else:
                    note = "critic still not fully satisfied after 2 revisions; returning the best attempt"
                s.message = (f"Generated with {best_g.model} (mode {best_g.mode}, seed {best_g.seed}, "
                             f"{best_g.steps} steps) after {s.revisions} revision(s); {note}. "
                             f"CLIP={best_c.clip_score:.3f}, VLM judge={best_c.vlm_verdict}.")
                break

        if s.status == "running":
            s.status = "error"
            s.message = s.message or "The request did not finish."
        s.total_seconds = round(time.time() - s.started, 3)
        self.save_trace(s)
        return s

    def save_trace(self, s: AgentState) -> str:
        d = os.path.join(self.cfg.out_dir, "traces")
        os.makedirs(d, exist_ok=True)
        path = os.path.join(d, f"{s.request_id}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s.to_json(), f, indent=2, default=str)
        return path
