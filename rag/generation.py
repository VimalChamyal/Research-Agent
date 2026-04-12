from langchain_core.prompts import ChatPromptTemplate
from config.llm import llm

answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Answer ONLY using context."),
        ("human", "Question: {question}\n\nContext:\n{context}")
    ]
)

def generate(state):
    out = (answer_prompt | llm).invoke({
        "question": state["question"],
        "context": state["refined_context"]
    })
    return {"answer": out.content}