# evaluation/judge.py

from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from rag.config import llm


# -----------------------------
# Output schema
# -----------------------------
class AnswerEvaluation(BaseModel):
    score: int   # 1 to 5
    reason: str


# -----------------------------
# Prompt
# -----------------------------
judge_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert evaluator of AI-generated answers.\n\n"
            "Evaluate the answer based on:\n"
            "1. Correctness\n"
            "2. Completeness\n"
            "3. Grounding in the provided context\n\n"
            "Scoring:\n"
            "1 = Very poor (incorrect / hallucinated)\n"
            "3 = Acceptable but incomplete\n"
            "5 = Excellent (correct, complete, well-grounded)\n\n"
            "Return JSON with:\n"
            "- score (1 to 5)\n"
            "- reason (short explanation)"
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Answer:\n{answer}"
        )
    ]
)


# -----------------------------
# Chain
# -----------------------------
judge_chain = judge_prompt | llm.with_structured_output(AnswerEvaluation)


# -----------------------------
# Function
# -----------------------------
def evaluate_answer(question: str, answer: str):
    result = judge_chain.invoke({
        "question": question,
        "answer": answer
    })
    return result