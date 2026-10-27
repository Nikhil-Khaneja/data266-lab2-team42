import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def simple_tokenize(text):
    return text.lower().split()


class HybridIndex:
    def __init__(self, persist_dir, text_model_name="BAAI/bge-small-en-v1.5", image_model_name="clip-ViT-B-32"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.text_collection = self.client.get_or_create_collection("text_chunks")
        self.figure_collection = self.client.get_or_create_collection("figures")
        self.text_embedder = SentenceTransformer(text_model_name)
        self.image_embedder = SentenceTransformer(image_model_name)
        self.bm25 = None
        self.bm25_chunks = []

    def build_text_index(self, chunks):
        self.bm25_chunks = chunks
        tokenized = [simple_tokenize(c["text"]) for c in chunks]
        self.bm25 = BM25Okapi(tokenized)

        texts = [c["text"] for c in chunks]
        embeddings = self.text_embedder.encode(texts, show_progress_bar=False).tolist()
        ids = ["text_{}".format(i) for i in range(len(chunks))]
        metadatas = [
            {"doc_id": c["doc_id"], "page": c["page"], "trust_level": c["trust_level"], "figure_id": ""}
            for c in chunks
        ]
        self.text_collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    def build_figure_index(self, figures):
        from PIL import Image

        if not figures:
            return
        images = [Image.open(f["image_path"]) for f in figures]
        embeddings = self.image_embedder.encode(images, show_progress_bar=False).tolist()
        ids = ["figure_{}".format(i) for i in range(len(figures))]
        documents = [f["caption"] for f in figures]
        metadatas = [
            {"doc_id": f["doc_id"], "page": f["page"], "trust_level": f["trust_level"], "figure_id": f["figure_id"]}
            for f in figures
        ]
        self.figure_collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    def dense_search_text(self, query, top_k=5):
        query_embedding = self.text_embedder.encode([query], show_progress_bar=False).tolist()
        results = self.text_collection.query(query_embeddings=query_embedding, n_results=top_k)
        return self._format_results(results)

    def bm25_search_text(self, query, top_k=5):
        if self.bm25 is None:
            return []
        scores = self.bm25.get_scores(simple_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for i in ranked:
            c = self.bm25_chunks[i]
            results.append({
                "text": c["text"],
                "doc_id": c["doc_id"],
                "page": c["page"],
                "trust_level": c["trust_level"],
                "figure_id": "",
                "score": float(scores[i]),
            })
        return results

    def hybrid_search_text(self, query, top_k=5):
        dense = self.dense_search_text(query, top_k=top_k)
        sparse = self.bm25_search_text(query, top_k=top_k)
        combined = {}
        for rank, item in enumerate(dense):
            key = (item["doc_id"], item["page"], item["text"][:50])
            combined[key] = combined.get(key, 0) + 1.0 / (rank + 1)
        for rank, item in enumerate(sparse):
            key = (item["doc_id"], item["page"], item["text"][:50])
            combined[key] = combined.get(key, 0) + 1.0 / (rank + 1)
        lookup = {}
        for item in dense + sparse:
            key = (item["doc_id"], item["page"], item["text"][:50])
            lookup[key] = item
        ranked_keys = sorted(combined.keys(), key=lambda k: combined[k], reverse=True)[:top_k]
        return [lookup[k] for k in ranked_keys]

    def find_figure_by_label(self, label):
        results = self.figure_collection.get(where={"figure_id": label})
        return self._format_get(results)

    def _format_results(self, results):
        out = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        for text, meta in zip(docs, metas):
            out.append({
                "text": text,
                "doc_id": meta["doc_id"],
                "page": meta["page"],
                "trust_level": meta["trust_level"],
                "figure_id": meta.get("figure_id", ""),
            })
        return out

    def _format_get(self, results):
        out = []
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        for text, meta in zip(docs, metas):
            out.append({
                "text": text,
                "doc_id": meta["doc_id"],
                "page": meta["page"],
                "trust_level": meta["trust_level"],
                "figure_id": meta.get("figure_id", ""),
            })
        return out
