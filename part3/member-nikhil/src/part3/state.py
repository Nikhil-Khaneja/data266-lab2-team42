from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class RequestState:
    """The one typed state every agent reads and writes, passed down the pipeline."""
    user_request: str
    input_image: Optional[Any] = None            # PIL.Image or None
    use_critique: bool = True                    # False = the no-critique baseline
    route: Optional[str] = None
    refused: bool = False
    refusal_reason: Optional[str] = None
    image_text: str = ""                         # text the Planner read in the image (data, never instructions)
    vision_caption: Optional[str] = None
    vision_objects: List[str] = field(default_factory=list)
    vision_text_in_image: Optional[str] = None
    vision_qa: List[dict] = field(default_factory=list)
    generation_prompt: Optional[str] = None
    generated_image: Optional[Any] = None
    generation_mode: Optional[str] = None
    generation_model: Optional[str] = None
    generation_seed: Optional[int] = None
    generation_steps: Optional[int] = None
    output_flagged: bool = False                 # Stable Diffusion safety checker hit on the output
    critique_score: Optional[float] = None
    critique_decision: Optional[str] = None      # "accept" or "revise", from the latest critique
    critique_feedback: Optional[str] = None
    critique_history: List[str] = field(default_factory=list)   # every decision, in order
    revision_count: int = 0
    final_status: Optional[str] = None           # accepted | revision_limit_reached | no_critique | refused
    trace: List[dict] = field(default_factory=list)

    def log_step(self, agent_name, step_input, step_output, duration_seconds):
        self.trace.append({
            "agent": agent_name,
            "input": step_input,
            "output": step_output,
            "duration_seconds": duration_seconds,
        })
