"""Standard information-retrieval metrics (same definitions as the BEIR benchmark)."""
import numpy as np


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    gains = [relevant.get(d, 0) for d in ranked[:k]]
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    ideal = sorted(relevant.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg else 0.0


def recall_at_k(ranked: list[str], relevant: dict[str, int], k: int) -> float:
    rel = {d for d, s in relevant.items() if s > 0}
    return len(rel.intersection(ranked[:k])) / len(rel) if rel else 0.0


def mrr_at_k(ranked: list[str], relevant: dict[str, int], k: int = 10) -> float:
    for i, d in enumerate(ranked[:k]):
        if relevant.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def evaluate(run: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> dict[str, float]:
    qs = list(run)
    return {
        "nDCG@10": float(np.mean([ndcg_at_k(run[q], qrels[q], 10) for q in qs])),
        "MRR@10": float(np.mean([mrr_at_k(run[q], qrels[q], 10) for q in qs])),
        "Recall@10": float(np.mean([recall_at_k(run[q], qrels[q], 10) for q in qs])),
        "Recall@100": float(np.mean([recall_at_k(run[q], qrels[q], 100) for q in qs])),
    }
