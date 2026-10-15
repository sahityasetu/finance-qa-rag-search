"""Retrievers: sparse BM25, dense embeddings, hybrid fusion, and a cross-encoder re-ranker.

Every retriever returns, for each query, a ranked list of (doc_index, score).
"""
import re
from collections import Counter
from pathlib import Path

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "cache"

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# bge models are trained to embed short questions with this prefix
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

STOPWORDS = set("""a an the and or but if of at by for with about against between into through during before after
above below to from up down in out on off over under again further then once here there when where why how all any
both each few more most other some such no nor not only own same so than too very s t can will just don should now
i me my we our you your he him his she her it its they them their what which who whom this that these those am is
are was were be been being have has had having do does did doing would could""".split())


def tokenize(text: str) -> list[str]:
    """Lower-case words without stop words, with light suffix stripping so that
    'taxes'/'tax', 'companies'/'company' and 'investing'/'invest' match."""
    out = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in STOPWORDS:
            continue
        if len(t) > 5 and t.endswith("ing"):
            t = t[:-3]
        elif len(t) > 4 and t.endswith("ies"):
            t = t[:-3] + "y"
        elif len(t) > 3 and t.endswith("es") and t[:-2].endswith(("x", "s", "ch", "sh", "z")):
            t = t[:-2]
        elif len(t) > 3 and t.endswith("s") and not t.endswith(("ss", "us", "is")):
            t = t[:-1]
        out.append(t)
    return out


class BM25:
    """Okapi BM25 on a sparse term-document matrix (fast: one sparse product per query batch)."""

    def __init__(self, docs: list[str], k1: float = 0.9, b: float = 0.4):
        self.vocab: dict[str, int] = {}
        rows, cols, vals, lengths = [], [], [], []
        for i, doc in enumerate(docs):
            counts = Counter(tokenize(doc))
            lengths.append(sum(counts.values()))
            for term, c in counts.items():
                rows.append(i)
                cols.append(self.vocab.setdefault(term, len(self.vocab)))
                vals.append(c)
        tf = sp.csr_matrix((vals, (rows, cols)), shape=(len(docs), len(self.vocab)), dtype=np.float32)
        lengths = np.asarray(lengths, dtype=np.float32)
        df = np.bincount(tf.indices, minlength=len(self.vocab))
        self.idf = np.log(1 + (len(docs) - df + 0.5) / (df + 0.5)).astype(np.float32)
        # Pre-compute the BM25 weight of every (doc, term) pair.
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        tf = tf.tocoo()
        w = tf.data * (k1 + 1) / (tf.data + norm[tf.row]) * self.idf[tf.col]
        self.weights = sp.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # terms x docs

    def search(self, queries: list[str], k: int = 100):
        q = sp.lil_matrix((len(queries), len(self.vocab)), dtype=np.float32)
        for i, text in enumerate(queries):
            for t in set(tokenize(text)):
                if t in self.vocab:
                    q[i, self.vocab[t]] = 1.0
        scores = (q.tocsr() @ self.weights).toarray()
        return _top_k(scores, k)


class DenseRetriever:
    """Bi-encoder: embed documents once, then nearest-neighbour search by cosine similarity."""

    def __init__(self, docs: list[str], model_name: str = EMBED_MODEL, device: str | None = None):
        from sentence_transformers import SentenceTransformer
        import torch

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = SentenceTransformer(model_name, device=device)
        CACHE.mkdir(parents=True, exist_ok=True)
        path = CACHE / f"{model_name.replace('/', '_')}_{len(docs)}.npy"
        if path.exists():
            self.doc_emb = np.load(path)
        else:
            print(f"Embedding {len(docs):,} documents with {model_name} (one-time, cached)...")
            self.doc_emb = self.model.encode(docs, batch_size=64, normalize_embeddings=True,
                                             show_progress_bar=True).astype(np.float32)
            np.save(path, self.doc_emb)

    def search(self, queries: list[str], k: int = 100):
        q = self.model.encode([QUERY_PREFIX + t for t in queries], batch_size=64, normalize_embeddings=True)
        return _top_k(q @ self.doc_emb.T, k)


def reciprocal_rank_fusion(*runs, k: int = 100, c: int = 60):
    """Combine ranked lists: score = sum over retrievers of 1 / (c + rank). No score tuning needed."""
    fused = []
    for lists in zip(*runs):
        scores: dict[int, float] = {}
        for ranked in lists:
            for rank, (doc, _) in enumerate(ranked, start=1):
                scores[doc] = scores.get(doc, 0.0) + 1.0 / (c + rank)
        fused.append(sorted(scores.items(), key=lambda x: -x[1])[:k])
    return fused


class Reranker:
    """Cross-encoder reads (question, passage) together - slower but more accurate, so only for the top-N."""

    def __init__(self, model_name: str = RERANK_MODEL, device: str | None = None):
        from sentence_transformers import CrossEncoder
        import torch

        device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = CrossEncoder(model_name, device=device, max_length=512)

    def rerank(self, queries: list[str], runs, docs: list[str], top_n: int = 50):
        out = []
        for query, ranked in zip(queries, runs):
            cand = [d for d, _ in ranked[:top_n]]
            scores = self.model.predict([(query, docs[d]) for d in cand], batch_size=64)
            order = np.argsort(-scores)
            out.append([(cand[i], float(scores[i])) for i in order] + ranked[top_n:])
        return out


def _top_k(scores: np.ndarray, k: int):
    idx = np.argpartition(-scores, kth=min(k, scores.shape[1] - 1), axis=1)[:, :k]
    results = []
    for row, cand in zip(scores, idx):
        cand = cand[np.argsort(-row[cand])]
        results.append([(int(d), float(row[d])) for d in cand])
    return results
