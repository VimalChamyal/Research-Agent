from typing import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END


from rag.retrieval import retrieve_node
from rag.doc_evaluator import eval_each_doc_node
from rag.web import rewrite_query_node, web_search_node
from rag.refinement import refine
from rag.generation import generate

class State(TypedDict):
    question: str
    retriever: object

    docs: List[Document]
    good_docs: List[Document]

    verdict: str
    reason: str

    doc_scores: List[float]

    web_query: str
    web_docs: List[Document]

    refined_context: str
    answer: str


def route_after_eval(state):
    return "refine" if state["verdict"] == "CORRECT" else "rewrite_query"


def build_graph():
    g = StateGraph(State)

    g.add_node("retrieve", retrieve_node)
    g.add_node("eval_each_doc", eval_each_doc_node)
    g.add_node("rewrite_query", rewrite_query_node)
    g.add_node("web_search", web_search_node)
    g.add_node("refine", refine)
    g.add_node("generate", generate)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "eval_each_doc")

    g.add_conditional_edges(
        "eval_each_doc",
        route_after_eval,
        {
            "refine": "refine",
            "rewrite_query": "rewrite_query"
        }
    )

    g.add_edge("rewrite_query", "web_search")
    g.add_edge("web_search", "refine")
    g.add_edge("refine", "generate")
    g.add_edge("generate", END)

    return g.compile()