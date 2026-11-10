"""Single place for every switch that changes what a Kaggle submission does.

Each flag is read from the environment, so one run = one explicit configuration, written to
run_manifest.json next to submission.csv and traceable to a leaderboard score.
"""
import os
from dataclasses import asdict, dataclass


@dataclass
class RagConfig:
    generation_top_k: int  # retrieved chunks shown to the model
    narrative_top_k: int   # passages shown for an explanatory question
    min_per_doc: int       # guarantee this many chunks from every document (conflicts need all sources)
    max_images: int        # chart images attached when a question names figures
    max_citations: int
    fact_top_k: int        # table cells shown for a question that names a year
    max_context_chars: int  # keeps the prompt near 4.5k tokens, so the T4 peak stays under its 15GB
    embedder: str
    reranker: str          # "" disables the cross-encoder
    model_name: str
    quantization: str      # "none" | "4bit"

    def as_dict(self):
        return asdict(self)


def load_config():
    return RagConfig(
        generation_top_k=int(os.environ.get("RAG_TOP_K", "6")),
        narrative_top_k=int(os.environ.get("RAG_NARRATIVE_TOP_K", "5")),
        min_per_doc=int(os.environ.get("RAG_MIN_PER_DOC", "2")),
        max_images=int(os.environ.get("RAG_MAX_IMAGES", "3")),
        max_citations=int(os.environ.get("RAG_MAX_CITATIONS", "6")),
        fact_top_k=int(os.environ.get("RAG_FACT_TOP_K", "6")),
        max_context_chars=int(os.environ.get("RAG_MAX_CONTEXT_CHARS", "9500")),
        embedder=os.environ.get("RAG_EMBEDDER", "BAAI/bge-small-en-v1.5"),
        reranker=os.environ.get("RAG_RERANKER", "BAAI/bge-reranker-v2-m3"),
        model_name=os.environ.get("RAG_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
        quantization=os.environ.get("RAG_QUANT", "4bit").strip().lower(),
    )
