from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.documents import Document
from config.llm import llm
from langchain_openai import ChatOpenAI


tavily = TavilySearchResults(max_results=5)

class WebQuery(BaseModel):
    query: str

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Rewrite question to short search query."),
        ("human", "Question: {question}")
    ]
)

llm_local = ChatOpenAI(model="gpt-4o-mini", temperature=0)
rewrite_chain = rewrite_prompt | llm.with_structured_output(WebQuery)


def rewrite_query_node(state):
    out = rewrite_chain.invoke({"question": state["question"]})
    return {"web_query": out.query}


def web_search_node(state):
    q = state.get("web_query") or state["question"]
    results = tavily.invoke({"query": q})

    docs = []
    for r in results or []:
        text = f"{r.get('title')}\n{r.get('content')}"
        docs.append(Document(page_content=text))

    return {"web_docs": docs}