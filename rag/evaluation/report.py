# evaluation/report.py

from rag.evaluation.metrics import avg_retrieval_score, verdict_stats
from rag.evaluation.judge import evaluate_answer
from rag.config import llm

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate


# -----------------------------
# Query Expansion (Paraphrasing)
# -----------------------------
class QueryVariants(BaseModel):
    variants: list[str]


paraphrase_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Generate 2 alternative ways to ask the same question.\n"
            "Keep meaning same.\n"
            "Return JSON with key: variants"
        ),
        ("human", "Question: {question}")
    ]
)

paraphrase_chain = paraphrase_prompt | llm.with_structured_output(QueryVariants)


def generate_queries(user_query):
    try:
        out = paraphrase_chain.invoke({"question": user_query})
        variants = out.variants[:2]
    except:
        variants = []

    return [user_query] + variants


# -----------------------------
# Main Report Function
# -----------------------------
def generate_evaluation_report(app, retriever, user_query):

    # Step 1: Generate queries
    queries = generate_queries(user_query)

    results = []

    # Step 2: Run evaluation
    for q in queries:
        res = app.invoke({
            "question": q,
            "retriever": retriever,
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

        # Retrieval score
        doc_scores = res.get("doc_scores", [])
        retrieval_score = avg_retrieval_score(doc_scores)

        # Answer score
        answer_eval = evaluate_answer(q, res["answer"])

        # Failure detection
        failure_reason = None

        if retrieval_score < 0.4:
            failure_reason = "Low retrieval quality"

        elif res["verdict"] == "INCORRECT":
            failure_reason = "Retriever failed (web fallback)"

        elif answer_eval.score <= 2:
            failure_reason = "Poor answer quality"

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
    avg_ret = sum(r["retrieval_score"] for r in results) / len(results)
    avg_ans = sum(r["answer_score"] for r in results) / len(results)
    verdict_distribution = verdict_stats(results)

    # -----------------------------
    # Build Report
    # -----------------------------
    report = []

    report.append("RAG EVALUATION REPORT\n")

    report.append("🔢 Metrics\n")
    report.append(f"- Avg Retrieval Score: {round(avg_ret, 2)}")
    report.append(f"- Avg Answer Score: {round(avg_ans, 2)}\n")

    report.append("⚖️ Verdict Distribution\n")
    for k, v in verdict_distribution.items():
        report.append(f"- {k}: {v * 100}%")

    report.append("\nQueries Used\n")
    for q in queries:
        report.append(f"- {q}")

    report.append("\nFailure Cases\n")

    failure_found = False
    for r in results:
        if r["failure"]:
            failure_found = True
            report.append(f"\nQ: {r['question']}")
            report.append(f"- Issue: {r['failure']}")
            report.append(f"- Retrieval Score: {round(r['retrieval_score'],2)}")
            report.append(f"- Answer Score: {r['answer_score']}")

    if not failure_found:
        report.append("\nNo major failure cases detected.\n")

    report.append("\nSystem Note\n")
    report.append(
        "Evaluation uses user query + paraphrased variants to test robustness. "
        "Retrieval quality is measured using LLM-based scoring, and answer quality "
        "is judged independently."
    )

    return "\n".join(report)