from judge import evaluate_answer
from metrics import avg_retrieval_score, verdict_stats

from ingestion import load_and_split_from_paths
from retrieval import build_retriever
from graph import build_graph

from dotenv import load_dotenv
import os

# Load .env from parent folder (rag/)
load_dotenv(dotenv_path="../.env")

import glob

# -----------------------------
# Load documents
# -----------------------------
pdf_paths = glob.glob("../rag/documents/*.pdf")

chunks = load_and_split_from_paths(pdf_paths)
retriever = build_retriever(chunks)
app = build_graph()

# -----------------------------
# Queries (dynamic)
# -----------------------------
queries = [
    "What are common uses?",
    "What trends are discussed?",
    "Compare key concepts mentioned",
    "What is ntpc doing for flyash?",
]

results = []

# -----------------------------
# Run evaluation
# -----------------------------
for q in queries:
    print("\n======================")
    print("Q:", q)

    res = app.invoke({
        "question": q,
        "retriever": retriever,
        "docs": [],
        "good_docs": [],
        "verdict": "",
        "reason": "",
        "doc_scores": [],   # 🔥 IMPORTANT
        "web_query": "",
        "web_docs": [],
        "refined_context": "",
        "answer": ""
    })

    # -----------------------------
    # REAL retrieval score
    # -----------------------------
    doc_scores = res.get("doc_scores", [])
    retrieval_score = avg_retrieval_score(doc_scores)

    # -----------------------------
    # Answer evaluation
    # -----------------------------
    answer_eval = evaluate_answer(q, res["answer"])

    # -----------------------------
    # Failure detection
    # -----------------------------
    failure_reason = None

    if retrieval_score < 0.4:
        failure_reason = "Low retrieval quality"

    elif res["verdict"] == "INCORRECT":
        failure_reason = "Retriever failed, used web"

    elif answer_eval.score <= 2:
        failure_reason = "Poor answer quality"

    # -----------------------------
    # Print
    # -----------------------------
    print("VERDICT:", res["verdict"])
    print("RETRIEVAL SCORE:", round(retrieval_score, 2))
    print("ANSWER SCORE:", answer_eval.score)

    if failure_reason:
        print("⚠️ FAILURE:", failure_reason)

    # -----------------------------
    # Store results
    # -----------------------------
    results.append({
        "question": q,
        "verdict": res["verdict"],
        "retrieval_score": retrieval_score,
        "answer_score": answer_eval.score,
        "failure": failure_reason
    })


# -----------------------------
# Summary
# -----------------------------
print("\n\n===== FINAL SUMMARY =====")

avg_ret = sum(r["retrieval_score"] for r in results) / len(results)
avg_ans = sum(r["answer_score"] for r in results) / len(results)
verdict_distribution = verdict_stats(results)

print("Avg Retrieval Score:", round(avg_ret, 2))
print("Avg Answer Score:", round(avg_ans, 2))
print("Verdict Distribution:", verdict_distribution)


# -----------------------------
# Failure Cases
# -----------------------------
print("\n===== FAILURE CASES =====")

for r in results:
    if r["failure"]:
        print("\nQ:", r["question"])
        print("Issue:", r["failure"])