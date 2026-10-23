class RequestState:
    def __init__(self, user_request, input_image=None):
        self.user_request = user_request
        self.input_image = input_image
        self.route = None
        self.refused = False
        self.refusal_reason = None
        self.vision_caption = None
        self.vision_objects = []
        self.vision_text_in_image = None
        self.vision_qa = []
        self.generation_prompt = None
        self.generated_image = None
        self.generation_mode = None
        self.generation_model = None
        self.generation_seed = None
        self.generation_steps = None
        self.critique_score = None
        self.critique_decision = None
        self.critique_feedback = None
        self.revision_count = 0
        self.trace = []

    def log_step(self, agent_name, step_input, step_output, duration_seconds):
        self.trace.append({
            "agent": agent_name,
            "input": step_input,
            "output": step_output,
            "duration_seconds": duration_seconds,
        })
