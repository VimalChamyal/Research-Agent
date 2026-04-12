import re
from typing import List
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from config.llm import llm
from langchain_openai import ChatOpenAI

def decompose_to_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


class KeepOrDrop(BaseModel):
    keep: bool

llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0)

filter_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Return keep=true only if sentence helps answer."),
        ("human", "Question: {question}\n\nSentence:\n{sentence}")
    ]
)

llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0)
filter_chain = filter_prompt | llm.with_structured_output(KeepOrDrop)


def refine(state):
    q = state["question"]

    if state["verdict"] == "CORRECT":
        docs = state["good_docs"]
    elif state["verdict"] == "INCORRECT":
        docs = state["web_docs"]
    else:
        docs = state["good_docs"] + state["web_docs"]

    context = "\n\n".join(d.page_content for d in docs)

    strips = decompose_to_sentences(context)

    kept = []
    for s in strips:
        if filter_chain.invoke({"question": q, "sentence": s}).keep:
            kept.append(s)

    return {
        "refined_context": "\n".join(kept)
    }