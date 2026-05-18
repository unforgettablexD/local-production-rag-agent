from __future__ import annotations

import json
import os
from typing import Any

import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Local Production RAG Agent", layout="wide")


def api_get(path: str) -> dict[str, Any]:
    response = requests.get(f"{BACKEND_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, payload: dict[str, Any] | None = None, files=None) -> dict[str, Any]:
    response = requests.post(f"{BACKEND_URL}{path}", json=payload, files=files, timeout=180)
    response.raise_for_status()
    return response.json()


def api_delete(path: str) -> dict[str, Any]:
    response = requests.delete(f"{BACKEND_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def render_sidebar() -> None:
    st.sidebar.title("Local Production RAG Agent")
    st.sidebar.caption("FastAPI + Streamlit + Ollama + Qdrant")
    try:
        health = api_get("/health")
        st.sidebar.success(f"Backend: {health['status']}")
        st.sidebar.write(f"Ollama reachable: `{health['ollama_reachable']}`")
        st.sidebar.write(f"Qdrant reachable: `{health['qdrant_reachable']}`")
    except Exception as exc:
        st.sidebar.error("Backend unavailable")
        st.sidebar.caption(str(exc))


def render_upload_section() -> None:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or Markdown files",
        type=["pdf", "docx", "txt", "md", "markdown"],
        accept_multiple_files=True,
    )
    if st.button("Index Uploaded Files", use_container_width=True) and uploaded_files:
        files = [("files", (file.name, file.getvalue(), file.type or "application/octet-stream")) for file in uploaded_files]
        with st.spinner("Indexing documents..."):
            result = api_post("/documents/upload", files=files)
        st.success(result["message"])
        st.json(result["indexed_documents"])


def render_documents_section() -> list[dict[str, Any]]:
    st.subheader("Indexed Documents")
    documents = api_get("/documents").get("documents", [])
    if not documents:
        st.info("No documents indexed yet.")
        return []
    for document in documents:
        cols = st.columns([4, 1])
        cols[0].markdown(
            f"**{document['filename']}**  \n"
            f"`{document['document_id']}` | chunks: `{document['chunk_count']}`"
        )
        if cols[1].button("Delete", key=f"delete-{document['document_id']}"):
            api_delete(f"/documents/{document['document_id']}")
            st.rerun()
    return documents


def render_chat_tab() -> None:
    st.subheader("Grounded RAG Chat")
    question = st.text_area(
        "Ask a question about your indexed documents",
        placeholder="Example: What does the security policy say about MFA for contractors?",
        height=120,
    )
    top_k = st.slider("Top-K retrieval", min_value=1, max_value=10, value=5)
    if st.button("Ask", use_container_width=True):
        with st.spinner("Retrieving context and generating answer..."):
            response = api_post("/chat", {"question": question, "top_k": top_k})
        st.markdown("### Answer")
        st.write(response["answer"])
        st.write(f"Confidence: `{response['confidence']:.2f}`")
        st.write(f"Needs retrieval: `{response['needs_retrieval']}`")
        if response.get("rewritten_query"):
            st.write(f"Rewritten query: `{response['rewritten_query']}`")

        st.markdown("### Citations")
        if response["citations"]:
            for citation in response["citations"]:
                page = f" page {citation['page_number']}" if citation["page_number"] else ""
                st.markdown(
                    f"- [{citation['index']}] **{citation['filename']}**{page} | `{citation['chunk_id']}`"
                )
                st.caption(citation["snippet"])
        else:
            st.info("No citations returned.")

        with st.expander("Retrieved Chunks", expanded=False):
            for chunk in response["retrieved_chunks"]:
                st.markdown(
                    f"**[{chunk['index']}] {chunk['filename']}** | score `{chunk['score']:.3f}` | `{chunk['chunk_id']}`"
                )
                st.write(chunk["text"])
                st.divider()


def render_summary_tab(documents: list[dict[str, Any]]) -> None:
    st.subheader("Document Summary")
    if not documents:
        st.info("Index documents first.")
        return
    options = {f"{doc['filename']} ({doc['document_id']})": doc["document_id"] for doc in documents}
    selected = st.selectbox("Select a document", list(options.keys()))
    if st.button("Summarize Document", use_container_width=True):
        result = api_post(f"/documents/{options[selected]}/summarize", {})
        st.write(result["summary"])
        st.json(result["citations"])


def render_compare_tab(documents: list[dict[str, Any]]) -> None:
    st.subheader("Compare Documents")
    if len(documents) < 2:
        st.info("Index at least two documents first.")
        return
    labels = {f"{doc['filename']} ({doc['document_id']})": doc["document_id"] for doc in documents}
    selected_a = st.selectbox("Document A", list(labels.keys()), key="doc_a")
    selected_b = st.selectbox("Document B", list(labels.keys()), index=1, key="doc_b")
    if st.button("Compare", use_container_width=True):
        result = api_post(
            "/documents/compare",
            {"document_id_a": labels[selected_a], "document_id_b": labels[selected_b]},
        )
        st.write(result["comparison"])
        st.json(result["citations"])


def render_evaluation_tab() -> None:
    st.subheader("Evaluation")
    top_k = st.slider("Eval Top-K", min_value=1, max_value=10, value=5, key="eval_top_k")
    if st.button("Run Evaluation", use_container_width=True):
        with st.spinner("Running evaluation set..."):
            result = api_post("/evaluate", {"top_k": top_k})
        st.markdown("### Metrics")
        st.json(result["metrics"])
        with st.expander("Detailed Results", expanded=False):
            st.code(json.dumps(result["results"], indent=2))


render_sidebar()
st.title("Local Production RAG Agent")
st.caption("A portfolio-grade local RAG system with citations, refusals, evaluation, and Dockerized deployment.")

render_upload_section()
documents = render_documents_section()
tabs = st.tabs(["Chat", "Summarize", "Compare", "Evaluate"])

with tabs[0]:
    render_chat_tab()
with tabs[1]:
    render_summary_tab(documents)
with tabs[2]:
    render_compare_tab(documents)
with tabs[3]:
    render_evaluation_tab()
