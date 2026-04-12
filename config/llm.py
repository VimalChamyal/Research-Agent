from langchain_openai import ChatOpenAI
from config.settings import OPENAI_API_KEY

print("✅ Using OpenAI LLM")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7,
    api_key=OPENAI_API_KEY
)