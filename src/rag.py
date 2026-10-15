"""Retrieval-augmented answering: retrieve passages, then ask an LLM to answer ONLY from them, with citations.

The LLM step is optional and provider-agnostic. Any OpenAI-compatible chat endpoint works, for example:
  - local and free:  LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=llama3.1   (Ollama)
  - hosted:          LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=<model>  LLM_API_KEY=...
Without LLM_BASE_URL set, the pipeline returns the retrieved passages only.
"""
import json
import os
import urllib.request
from dataclasses import dataclass

from src.retrievers import BM25, DenseRetriever, Reranker, reciprocal_rank_fusion

SYSTEM_PROMPT = (
    "You are a careful personal-finance assistant. Answer the question using ONLY the numbered sources. "
    "Cite every claim like [1] or [2]. If the sources do not contain the answer, say "
    "\"I could not find this in the sources.\" Do not give personalised investment advice."
)


@dataclass
class Passage:
    rank: int
    doc_id: str
    text: str
    score: float


class FinanceQA:
    """Default mode is dense retrieval: it scored best on the FiQA benchmark (see reports/metrics.json).
    'hybrid' and 'hybrid_rerank' are kept so the pipelines can be compared on new data."""

    def __init__(self, doc_ids: list[str], docs: list[str], mode: str = "dense"):
        self.doc_ids, self.docs, self.mode = doc_ids, docs, mode
        self.dense = DenseRetriever(docs)
        self.bm25 = BM25(docs) if mode != "dense" else None
        self.reranker = Reranker() if mode == "hybrid_rerank" else None

    def retrieve(self, question: str, k: int = 5) -> list[Passage]:
        ranked = self.dense.search([question], 100)
        if self.bm25:
            ranked = reciprocal_rank_fusion(self.bm25.search([question], 100), ranked)
        if self.reranker:
            ranked = self.reranker.rerank([question], ranked, self.docs, top_n=50)
        return [Passage(i + 1, self.doc_ids[d], self.docs[d], s) for i, (d, s) in enumerate(ranked[0][:k])]

    def answer(self, question: str, k: int = 5) -> dict:
        passages = self.retrieve(question, k)
        result = {"question": question, "passages": passages, "answer": None}
        if os.getenv("LLM_BASE_URL"):
            result["answer"] = call_llm(build_prompt(question, passages))
        return result


def build_prompt(question: str, passages: list[Passage], max_chars: int = 1500) -> list[dict]:
    sources = "\n\n".join(f"[{p.rank}] {p.text[:max_chars]}" for p in passages)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
    ]


def call_llm(messages: list[dict], timeout: int = 60) -> str:
    body = json.dumps({"model": os.environ["LLM_MODEL"], "messages": messages, "temperature": 0}).encode()
    req = urllib.request.Request(
        os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.getenv('LLM_API_KEY', 'none')}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]
