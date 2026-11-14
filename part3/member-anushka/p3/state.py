"""One shared, typed state passed between the five agents (pydantic models)."""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Intent = Literal["caption_question", "style_change", "edit", "enhance",
                 "domain_generation", "unsupported", "ambiguous", "unsafe"]
Status = Literal["running", "completed", "refused", "needs_clarification", "error"]


class PlannerOutput(BaseModel):
    intent: Intent
    route: list[str]                       # agents that will run, in order
    reason: str
    safety_flags: list[str] = []
    image_ok: Optional[bool] = None        # None = no image supplied
    image_issue: Optional[str] = None
    image_text_scan: Optional[str] = None  # text the planner found in the image (data, never instructions)


class VisionOutput(BaseModel):
    caption: str = ""
    objects: list[str] = []
    visible_text: str = ""
    qa: list[dict[str, str]] = []          # [{"question": ..., "answer": ...}]


class PromptOutput(BaseModel):
    mode: Literal["style", "edit", "enhance", "domain_img2img", "domain_txt2img"]
    prompt: str
    negative_prompt: str
    strength: Optional[float] = None
    guidance_scale: float = 7.5
    image_guidance_scale: Optional[float] = None
    applied_revision: Optional[str] = None


class GenerationRecord(BaseModel):
    attempt: int
    mode: str
    model: str
    lora: Optional[str] = None
    prompt: str
    negative_prompt: str
    seed: int
    steps: int
    strength: Optional[float] = None
    guidance_scale: float
    image_guidance_scale: Optional[float] = None
    resolution: list[int]
    nsfw_flagged: bool = False
    image_path: Optional[str] = None
    seconds: float = 0.0


class CritiqueOutput(BaseModel):
    attempt: int
    clip_score: float
    preservation: Optional[float] = None   # CLIP image-image similarity to the input
    vlm_verdict: str
    vlm_reason: str
    safe: bool
    decision: Literal["accept", "revise", "reject_unsafe"]
    revision_instruction: Optional[str] = None
    revision_params: dict[str, Any] = {}


class TraceEvent(BaseModel):
    agent: str
    started: float
    seconds: float
    input: dict[str, Any]
    output: dict[str, Any]


class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    user_request: str
    image_path: Optional[str] = None
    use_critique: bool = True

    image: Any = None                      # PIL.Image after validation (not serialised)
    planner: Optional[PlannerOutput] = None
    vision: Optional[VisionOutput] = None
    prompt: Optional[PromptOutput] = None
    generations: list[GenerationRecord] = []
    critiques: list[CritiqueOutput] = []
    revisions: int = 0

    status: Status = "running"
    message: str = ""
    final_image_path: Optional[str] = None
    trace: list[TraceEvent] = []
    started: float = Field(default_factory=time.time)
    total_seconds: float = 0.0

    def to_json(self) -> dict:
        return self.model_dump(exclude={"image"})
