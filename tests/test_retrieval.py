from src.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.rag import Passage, build_prompt
from src.retrievers import BM25, reciprocal_rank_fusion, tokenize

DOCS = [
    "A Roth IRA is funded with after-tax money and withdrawals in retirement are tax free.",
    "Index funds track the market and have low fees.",
    "Mortgage interest can be deductible if you itemize deductions.",
]


def test_tokenize_removes_stopwords_and_suffixes():
    assert tokenize("The taxes on investing") == ["tax", "invest"]
    assert tokenize("companies stocks ira") == ["company", "stock", "ira"]
    assert tokenize("tax company stock") == ["tax", "company", "stock"]


def test_bm25_finds_keyword_match():
    res = BM25(DOCS).search(["roth ira withdrawals", "low fee index fund"], k=2)
    assert res[0][0][0] == 0
    assert res[1][0][0] == 1


def test_rrf_rewards_agreement():
    a = [[(1, 9.0), (2, 8.0), (3, 7.0)]]
    b = [[(3, 0.9), (1, 0.8), (4, 0.7)]]
    fused = reciprocal_rank_fusion(a, b, k=4)[0]
    assert fused[0][0] == 1          # ranked 1st and 2nd -> best combined


def test_metrics():
    rel = {"d1": 1, "d2": 1}
    assert ndcg_at_k(["d1", "d2", "x"], rel) == 1.0
    assert recall_at_k(["x", "d1"], rel, 2) == 0.5
    assert mrr_at_k(["x", "d2"], rel) == 0.5


def test_prompt_numbers_sources_and_demands_citations():
    msgs = build_prompt("What is a Roth IRA?", [Passage(1, "0", DOCS[0], 1.0)])
    assert "ONLY the numbered sources" in msgs[0]["content"]
    assert msgs[1]["content"].startswith("Sources:\n[1] A Roth IRA")
