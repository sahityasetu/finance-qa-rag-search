"""Download and load the FiQA-2018 financial question answering benchmark (BEIR version)."""
import io
import json
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "fiqa"
URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/fiqa.zip"


def download():
    if (DATA / "corpus.jsonl").exists():
        return
    print("Downloading FiQA (~18 MB)...")
    with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read())) as z:
        z.extractall(DATA.parent)


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_corpus() -> pd.DataFrame:
    docs = pd.DataFrame(_read_jsonl(DATA / "corpus.jsonl"))[["_id", "text"]].rename(columns={"_id": "doc_id"})
    docs["text"] = docs["text"].str.strip()
    return docs


def load_queries(split: str = "test"):
    """Return (queries DataFrame, qrels dict query_id -> {doc_id: relevance})."""
    qrels_df = pd.read_csv(DATA / "qrels" / f"{split}.tsv", sep="\t", dtype=str)
    qrels = {}
    for q, d, s in qrels_df[["query-id", "corpus-id", "score"]].itertuples(index=False):
        qrels.setdefault(q, {})[d] = int(s)
    queries = pd.DataFrame(_read_jsonl(DATA / "queries.jsonl"))[["_id", "text"]].rename(columns={"_id": "query_id"})
    return queries[queries["query_id"].isin(qrels)].reset_index(drop=True), qrels
