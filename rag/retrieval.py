from langchain_community.vectorstores import FAISS
from rag.config import embeddings

def build_retriever(chunks):
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )
    return retriever


def retrieve_node(state):
    q = state["question"]
    return {"docs": state["retriever"].invoke(q)}