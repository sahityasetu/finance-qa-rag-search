"""Benchmark every retrieval pipeline on FiQA. BM25 settings are tuned on the dev split; results are on test.

Run from the project root:  python -m src.evaluate
Outputs: reports/metrics.json, reports/figures/*.png
"""
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.data import download, load_corpus, load_queries
from src.metrics import evaluate, ndcg_at_k
from src.retrievers import BM25, DenseRetriever, Reranker, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"


def to_run(qids, results, doc_ids):
    return {q: [doc_ids[d] for d, _ in r] for q, r in zip(qids, results)}


def timed(fn, *args, **kw):
    t = time.perf_counter()
    out = fn(*args, **kw)
    return out, time.perf_counter() - t


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    download()
    corpus = load_corpus()
    docs, doc_ids = corpus["text"].tolist(), corpus["doc_id"].tolist()

    # ---- tune BM25 on the dev split ----
    dev_q, dev_qrels = load_queries("dev")
    best = None
    for k1 in (0.9, 1.2, 1.5):
        for b in (0.4, 0.75):
            res = BM25(docs, k1=k1, b=b).search(dev_q["text"].tolist(), k=100)
            score = evaluate(to_run(dev_q["query_id"], res, doc_ids), dev_qrels)["nDCG@10"]
            print(f"BM25 dev k1={k1} b={b}: nDCG@10={score:.4f}")
            if best is None or score > best[0]:
                best = (score, k1, b)
    _, k1, b = best

    queries, qrels = load_queries("test")
    qtext, qids = queries["text"].tolist(), queries["query_id"].tolist()
    n = len(qtext)

    bm25 = BM25(docs, k1=k1, b=b)
    dense = DenseRetriever(docs)
    reranker = Reranker()

    bm25_res, t_bm25 = timed(bm25.search, qtext, k=100)
    dense_res, t_dense = timed(dense.search, qtext, k=100)
    hybrid_res, t_fuse = timed(reciprocal_rank_fusion, bm25_res, dense_res, k=100)
    hybrid_rr, t_rr = timed(reranker.rerank, qtext, hybrid_res, docs, top_n=50)
    dense_rr, t_rr_dense = timed(reranker.rerank, qtext, dense_res, docs, top_n=50)

    runs = {
        "BM25 (keyword)": (bm25_res, t_bm25),
        "Dense (bge-small embeddings)": (dense_res, t_dense),
        "Hybrid (BM25 + dense, RRF)": (hybrid_res, t_bm25 + t_dense + t_fuse),
        "Dense + cross-encoder re-rank": (dense_rr, t_dense + t_rr_dense),
        "Hybrid + cross-encoder re-rank": (hybrid_rr, t_bm25 + t_dense + t_fuse + t_rr),
    }
    table = {}
    for name, (res, secs) in runs.items():
        m = evaluate(to_run(qids, res, doc_ids), qrels)
        m["latency_ms_per_query"] = 1000 * secs / n
        table[name] = {k: round(v, 4) for k, v in m.items()}
        print(f"{name:34s} " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))

    # Where does each method win? Per-query nDCG@10, BM25 vs dense.
    per_q = pd.DataFrame({
        "query": qtext,
        "bm25": [ndcg_at_k([doc_ids[d] for d, _ in r], qrels[q]) for q, r in zip(qids, bm25_res)],
        "dense": [ndcg_at_k([doc_ids[d] for d, _ in r], qrels[q]) for q, r in zip(qids, dense_res)],
        "final": [ndcg_at_k([doc_ids[d] for d, _ in r], qrels[q]) for q, r in zip(qids, hybrid_rr)],
    })
    wins = {
        "dense_better": int((per_q["dense"] > per_q["bm25"]).sum()),
        "bm25_better": int((per_q["bm25"] > per_q["dense"]).sum()),
        "tie": int((per_q["bm25"] == per_q["dense"]).sum()),
        "final_pipeline_finds_relevant_in_top10_pct": round(100 * float((per_q["final"] > 0).mean()), 1),
        "bm25_finds_relevant_in_top10_pct": round(100 * float((per_q["bm25"] > 0).mean()), 1),
    }
    print(wins)

    metrics = {
        "dataset": {"documents": len(docs), "test_queries": n, "dev_queries": len(dev_q)},
        "bm25_params_tuned_on_dev": {"k1": k1, "b": b},
        "models": {"embedding": "BAAI/bge-small-en-v1.5", "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
                   "rerank_top_n": 50},
        "test_results": table,
        "bm25_vs_dense_per_query": wins,
    }
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ---- figure ----
    plt.style.use("seaborn-v0_8-whitegrid")
    t = pd.DataFrame(table).T
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 2]})
    colors = ["#BBBBBB", "#8DA0CB", "#66C2A5", "#FC8D62", "#4C72B0"]
    axes[0].barh(t.index, t["nDCG@10"], color=colors)
    for i, v in enumerate(t["nDCG@10"]):
        axes[0].text(v + 0.005, i, f"{v:.3f}", va="center")
    axes[0].invert_yaxis()
    axes[0].set(xlabel="nDCG@10 (higher is better)", title="Retrieval quality on FiQA test (648 questions)")
    axes[1].barh(t.index, t["latency_ms_per_query"], color=colors)
    axes[1].invert_yaxis()
    axes[1].set_yticklabels([])
    axes[1].set(xlabel="ms per query (batched, Apple M-series GPU)", title="Cost: latency")
    fig.tight_layout(); fig.savefig(FIG / "retrieval_comparison.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
