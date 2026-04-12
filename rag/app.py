import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from ingestion import load_from_uploaded_files
from retrieval import build_retriever
from graph import build_graph

# Evaluation imports
from evaluation.metrics import avg_retrieval_score
from evaluation.judge import evaluate_answer
from evaluation.report import generate_evaluation_report
from config import UPPER_TH, LOWER_TH


st.set_page_config(page_title="RAG Agent", layout="wide")

st.title("🔍 Research Corpus RAG Agent")

# -----------------------------
# Upload Section
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------
# Session State
# -----------------------------
if "app" not in st.session_state:
    st.session_state.app = None

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "last_question" not in st.session_state:
    st.session_state.last_question = None

if "report" not in st.session_state:
    st.session_state.report = None

# -----------------------------
# Build Knowledge Base
# -----------------------------
if st.button("Build Knowledge Base"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Processing documents..."):
            chunks = load_from_uploaded_files(uploaded_files)
            retriever = build_retriever(chunks)
            app = build_graph()

            st.session_state.app = app
            st.session_state.retriever = retriever

        st.success("Knowledge base ready!")

# -----------------------------
# Query Section
# -----------------------------
question = st.text_input("Ask a question")

if st.button("🔎 Run Query"):
    if st.session_state.app is None:
        st.warning("Please build the knowledge base first.")
    elif not question:
        st.warning("Enter a question.")
    else:
        st.session_state.last_question = question

        with st.spinner("Running RAG pipeline..."):
            res = st.session_state.app.invoke({
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

        # -----------------------------
        # Verdict
        # -----------------------------
        st.subheader("Verdict")
        st.write(f"**{res['verdict']}**")
        st.write(res["reason"])

        # -----------------------------
        # Evaluation
        # -----------------------------
        doc_scores = res.get("doc_scores", [])
        retrieval_score = avg_retrieval_score(doc_scores)

        answer_eval = evaluate_answer(question, res["answer"])

        failure_reason = None

        if retrieval_score < 0.4:
            failure_reason = "Low retrieval quality"
        elif res["verdict"] == "INCORRECT":
            failure_reason = "Retriever failed, used web"
        elif answer_eval.score <= 2:
            failure_reason = "Poor answer quality"

        st.subheader("📈 Evaluation")

        st.write(f"**Retrieval Score:** {round(retrieval_score, 2)}")
        st.write(f"**Answer Score:** {answer_eval.score}/5")

        # 🔥 Threshold line (NEW)
        st.write(f"**Thresholds → Upper:** {UPPER_TH} | **Lower:** {LOWER_TH}")

        if failure_reason:
            st.error(f"{failure_reason}")

        # -----------------------------
        # Retrieved Chunks (Sorted)
        # -----------------------------
        docs = res.get("docs", [])

        if docs:
            # Pair docs with scores and sort
            paired = list(zip(docs, doc_scores))
            paired.sort(key=lambda x: x[1] if isinstance(x[1], float) else 0, reverse=True)

            with st.expander("Retrieved Chunks (sorted by score)"):
                for i, (doc, score) in enumerate(paired):
                    label = "✅ HIGH" if isinstance(score, float) and score > UPPER_TH else \
                            "🟡 MEDIUM" if isinstance(score, float) and score > LOWER_TH else \
                            "❌ LOW"

                    st.markdown(
                        f"**Chunk {i+1} | Score: {round(score,2) if isinstance(score,float) else score} | {label}**"
                    )
                    st.write(doc.page_content[:500] + "...")
                    st.divider()

        # -----------------------------
        # Web Query
        # -----------------------------
        if res.get("web_query"):
            st.subheader("🌐 Web Query")
            st.write(res["web_query"])

        # -----------------------------
        # Context
        # -----------------------------
        with st.expander("Refined Context"):
            st.write(res["refined_context"])

        # -----------------------------
        # Answer
        # -----------------------------
        st.subheader("Answer")
        st.write(res["answer"])


# -----------------------------
# 🔥 Generate Evaluation Report
# -----------------------------
if st.button("Generate Evaluation Report"):
    if st.session_state.app is None:
        st.warning("Build knowledge base first.")
    elif not st.session_state.last_question:
        st.warning("Run at least one query first.")
    else:
        with st.spinner("Generating report..."):
            report = generate_evaluation_report(
                st.session_state.app,
                st.session_state.retriever,
                st.session_state.last_question
            )

            st.session_state.report = report

        st.success("Report generated!")

# -----------------------------
# Show Report
# -----------------------------
if st.session_state.report:
    st.subheader("Evaluation Report")

    st.text_area("Report", st.session_state.report, height=400)

    st.download_button(
        label="⬇️ Download Report",
        data=st.session_state.report,
        file_name="rag_evaluation_report.md",
        mime="text/markdown"
    )