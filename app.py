"""Streamlit demo: ask a personal-finance question, see the retrieved evidence (and an LLM answer if configured).

Run:  streamlit run app.py
"""
import os

import streamlit as st

from src.data import download, load_corpus
from src.rag import FinanceQA

st.set_page_config(page_title="Finance Q&A Search", layout="wide")


@st.cache_resource(show_spinner="Loading 57k finance answers and models (first run embeds the corpus)...")
def get_qa():
    download()
    corpus = load_corpus()
    return FinanceQA(corpus["doc_id"].tolist(), corpus["text"].tolist())


qa = get_qa()
st.title("Finance Q&A: Semantic Search for RAG")
st.caption("57,638 answers from the FiQA finance community dataset, searched with bge-small embeddings "
           "(the best of 5 pipelines benchmarked in this project).")

examples = ["Is it better to pay off my student loan or invest?", "How are stock options taxed when I exercise them?",
            "What happens to my 401k if I change jobs?"]
question = st.text_input("Your question", examples[0])
k = st.slider("Passages to show", 3, 10, 5)

if question:
    result = qa.answer(question, k)
    if result["answer"]:
        st.subheader("Answer (from the sources below)")
        st.write(result["answer"])
    elif not os.getenv("LLM_BASE_URL"):
        st.info("Showing retrieved evidence only. Set LLM_BASE_URL and LLM_MODEL to generate a cited answer.")
    st.subheader("Evidence")
    for p in result["passages"]:
        with st.expander(f"[{p.rank}] score {p.score:.2f} - {p.text[:120]}...", expanded=p.rank == 1):
            st.write(p.text)
