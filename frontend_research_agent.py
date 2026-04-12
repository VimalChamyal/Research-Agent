from __future__ import annotations
import json
from typing import Any, Dict, Iterator, Tuple, List
from datetime import datetime
import streamlit as st

# ==============================
# IMPORTS
# ==============================
from graphs.build_graph import build_app

# RAG
from rag.ingestion import load_from_uploaded_files
from rag.retrieval import build_retriever
from rag.graph import build_graph as build_rag_graph

from rag.evaluation.metrics import avg_retrieval_score
from rag.evaluation.judge import evaluate_answer
from rag.evaluation.report import generate_evaluation_report

from rag.config import UPPER_TH, LOWER_TH


# ==============================
# INIT
# ==============================
st.set_page_config(page_title="Research Corpus Agent", layout="wide")
st.title("🔎 Research Corpus Agent")

blog_app = build_app()


# ==============================
# SESSION STATE
# ==============================
if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "rag_app" not in st.session_state:
    st.session_state.rag_app = None

if "last_rag_result" not in st.session_state:
    st.session_state.last_rag_result = None

if "last_blog_out" not in st.session_state:
    st.session_state.last_blog_out = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "report" not in st.session_state:
    st.session_state.report = None


# ==============================
# SIDEBAR
# ==============================
with st.sidebar:
    st.header("Setup")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Build Knowledge Base"):
        if not uploaded_files:
            st.warning("Upload PDFs first")
        else:
            with st.spinner("Processing documents..."):
                chunks = load_from_uploaded_files(uploaded_files)
                retriever = build_retriever(chunks)

                st.session_state.retriever = retriever
                st.session_state.rag_app = build_rag_graph()

                st.success("Knowledge base ready!")

    st.divider()

    mode = st.radio(
        "Mode",
        ["🔍 RAG (Ask Questions)", "📝 Generate Content"]
    )


# =========================================================
# ====================== RAG MODE ==========================
# =========================================================

if mode == "🔍 RAG (Ask Questions)":

    st.header("Ask a Question")

    question = st.text_input("Enter your question")

    if st.button("Run Query"):

        if st.session_state.rag_app is None:
            st.warning("Build knowledge base first.")
        elif not question:
            st.warning("Enter a question.")
        else:
            st.session_state.last_question = question

            with st.spinner("Running RAG pipeline..."):
                res = st.session_state.rag_app.invoke({
                    "question": question,
                    "retriever": st.session_state.retriever,
                    "docs": [],
                    "good_docs": [],
                    "verdict": "",
                    "reason": "",
                    "doc_scores": [],
                    "web_query": "",
                    "web_docs": [],
                    "refined_context": "",
                    "answer": ""
                })

                st.session_state.last_rag_result = res

    # ===== DISPLAY =====
    if st.session_state.last_rag_result:

        res = st.session_state.last_rag_result

        st.subheader("Verdict")
        st.write(f"**{res['verdict']}**")
        st.write(res["reason"])

        # ===== EVALUATION =====
        doc_scores = res.get("doc_scores", [])
        retrieval_score = avg_retrieval_score(doc_scores)
        answer_eval = evaluate_answer(
            st.session_state.last_question,
            res["answer"]
        )

        st.subheader("Evaluation")
        st.write(f"Retrieval Score: {round(retrieval_score, 2)}")
        st.write(f"Answer Score: {answer_eval.score}/5")
        st.write(f"Thresholds → Upper: {UPPER_TH} | Lower: {LOWER_TH}")

        # ===== CHUNKS =====
        docs = res.get("docs", [])

        if docs:
            paired = list(zip(docs, doc_scores))
            paired.sort(
                key=lambda x: x[1] if isinstance(x[1], float) else 0,
                reverse=True
            )

            with st.expander("Retrieved Chunks"):
                for i, (doc, score) in enumerate(paired):

                    label = (
                        "HIGH" if score > UPPER_TH else
                        "MEDIUM" if score > LOWER_TH else
                        "LOW"
                    )

                    st.markdown(
                        f"**Chunk {i+1} | Score: {round(score,2)} | {label}**"
                    )
                    st.write(doc.page_content[:500] + "...")
                    st.divider()

        # ===== CONTEXT =====
        with st.expander("Refined Context"):
            st.write(res.get("refined_context", ""))

        # ===== ANSWER =====
        st.subheader("Answer")
        st.write(res.get("answer", ""))

        # ===== REPORT =====
        if st.button("Generate Evaluation Report"):

            report = generate_evaluation_report(
                st.session_state.rag_app,
                st.session_state.retriever,
                st.session_state.last_question
            )

            st.session_state.report = report

        if st.session_state.report:
            st.subheader("Evaluation Report")
            st.text_area("Report", st.session_state.report, height=400)

            st.download_button(
                "Download Report",
                st.session_state.report,
                file_name="rag_report.md"
            )


# =========================================================
# ==================== BLOG MODE ===========================
# =========================================================

else:

    st.header("Content Generation")

    topic = st.text_area("Topic")

    logs: List[str] = []

    def try_stream(graph_app, inputs):
        try:
            for step in graph_app.stream(inputs, stream_mode="updates"):
                yield ("updates", step)
            out = graph_app.invoke(inputs)
            yield ("final", out)
            return
        except:
            pass

        out = graph_app.invoke(inputs)
        yield ("final", out)

    def extract_latest_state(current_state, step_payload):
        if isinstance(step_payload, dict):
            if len(step_payload) == 1:
                inner = next(iter(step_payload.values()))
                current_state.update(inner)
            else:
                current_state.update(step_payload)
        return current_state

    if st.button("Generate Content"):

        if not topic.strip():
            st.warning("Enter topic")
            st.stop()

        if st.session_state.retriever is None:
            st.warning("Build knowledge base first")
            st.stop()

        inputs = {
            "topic": topic.strip(),
            "as_of": datetime.now().strftime("%Y-%m-%d"),
            "mode": "",
            "needs_research": False,
            "queries": [],
            "evidence": [],
            "plan": None,
            "sections": [],
            "merged_md": "",
            "final": "",
            "retriever": st.session_state.retriever
        }

        status = st.status("Running agent...", expanded=True)
        progress_area = st.empty()

        current_state = {}
        last_node = None

        for kind, payload in try_stream(blog_app, inputs):

            if kind == "updates":

                node_name = None
                if isinstance(payload, dict) and len(payload) == 1:
                    node_name = next(iter(payload.keys()))

                if node_name and node_name != last_node:
                    status.write(f"Node: `{node_name}`")
                    last_node = node_name

                current_state = extract_latest_state(current_state, payload)

                summary = {
                    "queries": current_state.get("queries"),
                    "evidence": len(current_state.get("evidence", [])),
                    "sections_done": len(current_state.get("sections", [])),
                }

                progress_area.json(summary)

            elif kind == "final":
                st.session_state.last_blog_out = payload
                status.update(label="Done", state="complete")

    # ===== OUTPUT =====
    out = st.session_state.get("last_blog_out")

    if out:

        tab1, tab2, tab3, tab4 = st.tabs([
            "Plan",
            "Evidence (Work In Progress)",
            "Content",
            "Logs (Work In Progress)"
        ])

        with tab1:
            st.subheader("Plan")
            st.json(out.get("plan"))

        with tab2:
            st.subheader("Evidence")
            st.json(out.get("evidence", []))

        with tab3:
            st.subheader("Generated Content")
            final_md = out.get("final", "")
            st.markdown(final_md)

            st.download_button(
                "Download Blog",
                final_md,
                file_name="blog.md"
            )

        with tab4:
            st.subheader("Logs")
            st.write("Basic logs (expand later)")