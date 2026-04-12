from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from rag.config import llm, UPPER_TH, LOWER_TH
from langchain_openai import ChatOpenAI


class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "You are a strict retrieval evaluator for RAG.\n"
         "Return score [0.0, 1.0]. Be conservative.\n"
         "Output JSON."),
        ("human", "Question: {question}\n\nChunk:\n{chunk}")
    ]
)

llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0)
doc_eval_chain = doc_eval_prompt | llm.with_structured_output(DocEvalScore)


def eval_each_doc_node(state):
    q = state["question"]
    scores = []
    good = []

    for d in state["docs"]:
        out = doc_eval_chain.invoke({
            "question": q,
            "chunk": d.page_content
        })

        scores.append(out.score)

        if out.score > LOWER_TH:
            good.append(d)

    # -----------------------------
    # Base update (IMPORTANT)
    # -----------------------------
    state_update = {
        "doc_scores": scores
    }

    # -----------------------------
    # Routing logic
    # -----------------------------
    if any(s > UPPER_TH for s in scores):
        state_update.update({
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": "High relevance chunk found"
        })
        return state_update

    if len(scores) > 0 and all(s < LOWER_TH for s in scores):
        state_update.update({
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "All chunks irrelevant"
        })
        return state_update

    state_update.update({
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": "Mixed relevance"
    })

    return state_update