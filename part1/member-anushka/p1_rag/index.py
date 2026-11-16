"""Tasks 1.1.4 and 1.2.1/1.2.2 - vector database + hybrid retrieval.

* ChromaDB (open source) stores every unit with doc_id, page, figure_id, trust_level, kind.
* Dense: BAAI/bge-small-en-v1.5 sentence embeddings (cosine).
* Sparse: BM25 (rank_bm25) over the same units.
* Fusion: reciprocal-rank fusion of the two lists.
* Exact-label routing: if the question names "Figure A-5" / "Table 1-2", that exact object is
  put first; other figures are dropped from the context so near-identical charts are never mixed.
* Reranker (bonus): a cross-encoder (BAAI/bge-reranker-v2-m3) re-scores the top 20 hybrid candidates;
  named objects and structured-fact tables stay pinned first.
* Low-trust content (Appendix C) stays in the index (it is part of the KB) but is never returned
  as evidence.
"""
import re

import numpy as np

from .vocab import find_labels

EMB_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"     # open-source cross-encoder, 568M params (bonus: reranker)
RERANK_CANDIDATES = 20                       # hybrid candidates re-scored by the cross-encoder
RERANK_FUSE = False                          # fuse the reranker's ranking with the hybrid ranking (RRF)
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.\-][a-z0-9]+)*")
STOP = set("the a an of in on for to and or is was were what how did does do by with from at as be "
           "that this which it its into than then there their they you your are been has have had".split())


def tokenize(text):
    return [t for t in TOKEN_RE.findall((text or "").lower().replace("−", "-")) if t not in STOP]


class KBIndex:
    def __init__(self, units, persist_dir=None, embed_model=EMB_MODEL, device=None):
        import chromadb
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.units = units
        self.reranker, self.rerank_model = None, None
        self.by_uid = {u.uid: u for u in units}
        self.by_label = {}
        for u in units:
            if u.label:
                self.by_label.setdefault(u.label, []).append(u)
        self.embedder = SentenceTransformer(embed_model, device=device)
        docs = [u.text for u in units]
        embs = self.embedder.encode(docs, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
        client = chromadb.PersistentClient(path=persist_dir) if persist_dir else chromadb.EphemeralClient()
        try:
            client.delete_collection("p1_kb")
        except Exception:
            pass
        self.col = client.create_collection("p1_kb", metadata={"hnsw:space": "cosine"})
        self.col.add(ids=[u.uid for u in units], embeddings=[e.tolist() for e in embs], documents=docs,
                     metadatas=[{"doc_id": u.doc_id, "page": int(u.page), "figure_id": u.label or "",
                                 "trust_level": u.trust, "kind": u.kind,
                                 "section": u.extra.get("section", "")} for u in units])
        self.bm25 = BM25Okapi([tokenize(d) for d in docs])

    # ------------------------------------------------------------------ primitives
    def dense(self, query, k=20, exclude_low=True):
        q = self.embedder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
        where = {"trust_level": {"$ne": "low"}} if exclude_low else None
        res = self.col.query(query_embeddings=[q.tolist()], n_results=k, where=where)
        return [(uid, 1 - d) for uid, d in zip(res["ids"][0], res["distances"][0])]

    def sparse(self, query, k=20, exclude_low=True):
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(-scores)
        out = []
        for i in order:
            u = self.units[i]
            if exclude_low and u.trust == "low":
                continue
            if scores[i] <= 0:
                break
            out.append((u.uid, float(scores[i])))
            if len(out) >= k:
                break
        return out

    # ------------------------------------------------------------------ retrieval modes
    def retrieve_dense_only(self, query, k=5):
        """Baseline: dense retrieval only, no routing."""
        return [self.by_uid[uid] for uid, _ in self.dense(query, k)]

    def enable_reranker(self, model_name=RERANK_MODEL, device=None):
        """Bonus: cross-encoder reranker applied to the top hybrid candidates."""
        import torch
        from sentence_transformers import CrossEncoder
        kw = {"model_kwargs": {"torch_dtype": torch.float16}} if torch.cuda.is_available() else {}
        self.reranker = CrossEncoder(model_name, max_length=512, device=device, **kw)
        self.rerank_model = model_name
        return self

    def rerank(self, query, cands, fuse=None, rrf_k=60):
        """Re-order candidate units (given in hybrid order) by cross-encoder relevance. With fuse, the
        reranker's ranking and the hybrid ranking are combined by reciprocal-rank fusion."""
        if not cands:
            return cands
        fuse = RERANK_FUSE if fuse is None else fuse
        scores = self.reranker.predict([(query, u.text[:2000]) for u in cands], batch_size=16,
                                       show_progress_bar=False)
        order = sorted(range(len(cands)), key=lambda i: -float(scores[i]))
        if not fuse:
            return [cands[i] for i in order]
        rr_rank = {i: r for r, i in enumerate(order)}
        comb = {i: 1.0 / (rrf_k + i + 1) + 1.0 / (rrf_k + rr_rank[i] + 1) for i in range(len(cands))}
        return [cands[i] for i in sorted(comb, key=lambda i: (-comb[i], i))]

    def retrieve(self, query, k=6, rrf_k=60, boost_uids=(), rerank=None, n_cand=RERANK_CANDIDATES, fuse=None):
        """Final: hybrid dense + BM25 with RRF, exact-label routing, low-trust filter, and (when enabled)
        a cross-encoder reranker over the top hybrid candidates. Named objects and boosted fact tables stay
        pinned first; the reranker orders everything after them."""
        use_rr = (self.reranker is not None) if rerank is None else (rerank and self.reranker is not None)
        fused = {}
        for lst in (self.dense(query, 25), self.sparse(query, 25)):
            for rank, (uid, _) in enumerate(lst):
                fused[uid] = fused.get(uid, 0.0) + 1.0 / (rrf_k + rank + 1)
        boosted = [uid for uid in boost_uids if uid in self.by_uid]
        for uid in boosted:                          # e.g. tables holding a structured-lookup fact
            fused[uid] = fused.get(uid, 0.0) + 0.02
        ranked = sorted(fused, key=lambda x: -fused[x])
        named = self.named_objects(query)
        out = list(named)
        cands = []
        for uid in ranked:
            u = self.by_uid[uid]
            if u in out or u in cands or u.trust == "low":
                continue
            if named and u.kind == "figure":         # never mix other (near-identical) figures in
                continue
            if use_rr and uid in boosted:
                out.append(u)                        # pinned: holds the verified value
                continue
            cands.append(u)
            if len(cands) >= (n_cand if use_rr else k):
                break
        if use_rr:
            cands = self.rerank(query, cands, fuse=fuse)
        out += cands
        return out[:max(k, len(named))]

    def named_objects(self, query):
        """Units for every Figure/Table label named in the query. A plain label also brings its
        '(corrected)' version, because an erratum supersedes the printed figure."""
        out = []
        for lab in find_labels(query):
            for cand in (lab, lab + " (corrected)" if "(corrected)" not in lab else None):
                for u in self.by_label.get(cand or "", []):
                    if u not in out and u.trust != "low":
                        out.append(u)
        # corrected version first: it is the one to read values from
        out.sort(key=lambda u: 0 if "(corrected)" in u.label else 1)
        return out
