import re

import chromadb
import numpy as np
import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from src.part1 import facts as facts_module

# Keeps letters, digits, apostrophes and hyphens inside numbers/labels, so "A-6", "China's"
# and "2011-2013" stay searchable, and trailing punctuation never blocks a match.
TOKEN_PATTERN = re.compile(r"[a-z0-9']+(?:-[a-z0-9]+)*")
# BGE models are trained with this instruction on the query side only.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
META_KEYS = ("doc_id", "page", "trust_level", "figure_id", "kind", "image_path")


def simple_tokenize(text):
    return TOKEN_PATTERN.findall(text.lower())


class HybridIndex:
    """Dense (ChromaDB) + BM25, fused with RRF, optionally reranked by a cross-encoder.

    Text passages, table chunks and figure descriptions live in one collection, so a value
    that only exists inside a chart can be found by an ordinary question.
    """

    def __init__(self, persist_dir, text_model_name="BAAI/bge-small-en-v1.5", reranker_name=None):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("kb_chunks", metadata={"hnsw:space": "cosine"})
        self.text_embedder = SentenceTransformer(text_model_name)
        self.reranker = None
        if reranker_name:
            half = {"dtype": torch.float16} if torch.cuda.is_available() else {}
            self.reranker = CrossEncoder(reranker_name, model_kwargs=half)
        self.chunks = []
        self.bm25 = None
        if self.collection.count() > 0:
            self._load_from_collection()

    def _load_from_collection(self):
        stored = self.collection.get(include=["documents", "metadatas"])
        self.chunks = [dict(meta, text=text) for text, meta in zip(stored["documents"], stored["metadatas"])]
        self.bm25 = BM25Okapi([simple_tokenize(c["text"]) for c in self.chunks])
        self._build_facts()

    def _build_facts(self):
        self.tables_by_label = {(c["doc_id"], c["figure_id"]): c for c in self.all_tables()}
        self.facts = [f for t in self.all_tables() for f in facts_module.keyed_table_facts(t)]
        self.prose_facts = [f for c in self.chunks if c.get("kind") == "text" for f in facts_module.prose_facts(c)]
        self.fact_embeddings = self.text_embedder.encode(
            [f["context"] for f in self.facts], show_progress_bar=False, normalize_embeddings=True)

    def search_facts(self, query, years, top_k=10):
        """Table cells most similar to the query. When the question names a year, only cells
        for that year qualify, so a value from a neighbouring column cannot be picked up. With
        no year (years=None), every cell competes, including the period averages."""
        if not self.facts:
            return []
        embedding = self.text_embedder.encode(
            [BGE_QUERY_PREFIX + query], show_progress_bar=False, normalize_embeddings=True)[0]
        scores = np.dot(self.fact_embeddings, embedding)
        eligible = [i for i, f in enumerate(self.facts) if not years or facts_module.fact_matches_years(f, years)]
        if self.reranker is not None and eligible:
            ce = self.reranker.predict([(query, self.facts[i]["context"]) for i in eligible],
                                       batch_size=16, show_progress_bar=False)
            scores = dict(zip(eligible, ce))
        ranked = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)[:top_k]
        return [self.facts[i] for i in ranked]

    def lookup(self, key):
        return facts_module.resolve_lookup(key, self.facts + self.prose_facts)

    def chunk_for_fact(self, fact):
        """The chunk a fact came from, for citation."""
        if fact.get("figure_id"):
            return self.tables_by_label[(fact["doc_id"], fact["figure_id"])]
        for c in self.chunks:
            if c.get("kind") == "text" and c["doc_id"] == fact["doc_id"] and c["page"] == fact["page"]:
                if fact["text"][:40] in c["text"].replace("\n", " ").replace("U.S.", "US"):
                    return c
        return None

    def facts_as_sources(self, facts):
        return facts_module.group_facts(facts, self.tables_by_label)

    def build(self, chunks):
        if not chunks:
            return
        texts = [c["text"] for c in chunks]
        embeddings = self.text_embedder.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
        metadatas = [{k: c.get(k, "") for k in META_KEYS} for c in chunks]
        ids = ["chunk_{}".format(i) for i in range(len(chunks))]
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        self._load_from_collection()

    def dense_search_text(self, query, top_k=5, exclude_low_trust=True):
        n = self.collection.count()
        if n == 0:
            return []
        embedding = self.text_embedder.encode(
            [BGE_QUERY_PREFIX + query], show_progress_bar=False, normalize_embeddings=True).tolist()
        kwargs = {"query_embeddings": embedding, "n_results": min(top_k, n)}
        if exclude_low_trust:
            # Filter BEFORE ranking: low-trust content is never evidence, so it must not
            # occupy a slot (it is also written to look relevant to what it tries to subvert).
            kwargs["where"] = {"trust_level": {"$ne": "low"}}
        results = self.collection.query(**kwargs)
        return [dict(meta, text=text) for text, meta in zip(results["documents"][0], results["metadatas"][0])]

    def bm25_search_text(self, query, top_k=5, exclude_low_trust=True):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(simple_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            if exclude_low_trust and self.chunks[i]["trust_level"] == "low":
                continue
            results.append(self.chunks[i])
            if len(results) >= top_k:
                break
        return results

    @staticmethod
    def key(item):
        return (item["doc_id"], item["page"], item.get("figure_id", ""), item["text"][:60])

    def fused_ranking(self, query, fetch_k=30, rrf_k=60):
        scores, lookup = {}, {}
        for results in (self.dense_search_text(query, top_k=fetch_k), self.bm25_search_text(query, top_k=fetch_k)):
            for rank, item in enumerate(results):
                key = self.key(item)
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                lookup.setdefault(key, item)
        return [lookup[k] for k in sorted(scores, key=scores.get, reverse=True)]

    def ranked(self, query, fetch_k=30):
        ranking = self.fused_ranking(query, fetch_k=fetch_k)
        if self.reranker is not None and ranking:
            scores = self.reranker.predict([(query, r["text"]) for r in ranking], batch_size=8, show_progress_bar=False)
            order = sorted(range(len(ranking)), key=lambda i: float(scores[i]), reverse=True)
            ranking = [ranking[i] for i in order]
        return ranking

    def hybrid_search_text(self, query, top_k=5):
        return self.ranked(query)[:top_k]

    def retrieve(self, query, top_k=8, min_per_doc=0):
        """Top-k, then top up any document with fewer than min_per_doc chunks. A conflict or
        revision is only visible if the model sees every document stating the value."""
        ranking = self.ranked(query)
        picked = ranking[:top_k]
        for doc_id in sorted({r["doc_id"] for r in ranking}):
            have = sum(1 for r in picked if r["doc_id"] == doc_id)
            for r in ranking[top_k:]:
                if have >= min_per_doc:
                    break
                if r["doc_id"] == doc_id:
                    picked.append(r)
                    have += 1
        return picked

    def all_tables(self):
        tables = [c for c in self.chunks if c.get("kind") == "table" and c["trust_level"] != "low"]
        return sorted(tables, key=lambda c: (c["doc_id"], c["page"]))

    def known_labels(self):
        return {c["figure_id"] for c in self.chunks if c.get("figure_id") and c["trust_level"] != "low"}

    def find_by_label(self, label):
        return [c for c in self.chunks if c.get("figure_id") == label and c["trust_level"] != "low"]

    def find_text_by_page(self, doc_id, page):
        return [c for c in self.chunks
                if c["doc_id"] == doc_id and c["page"] == page and c.get("kind") == "text" and c["trust_level"] != "low"]
