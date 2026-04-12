from langchain_core.messages import SystemMessage, HumanMessage
from config.llm import llm
from schemas.models import Task, Plan, EvidenceItem

# ✅ NEW: import RAG graph
from rag.graph import build_graph

# ✅ Initialize RAG app (once)
rag_app = build_graph()


WORKER_SYSTEM = """You are a senior technical writer and developer advocate.

Write ONE section of a technical blog post in Markdown.

Hard constraints:
- Follow the provided Goal and cover ALL Bullets in order (do not skip or merge bullets).
- Stay close to Target words (±15%).
- Output ONLY the section content in Markdown (no blog title H1, no extra commentary).
- Start with a '## <Section Title>' heading.

Scope guard:
- If blog_kind == "news_roundup": focus on summarizing events and implications.

Grounding policy:
- Use provided context strictly when available.
- Do NOT hallucinate facts outside the context.

Code:
- If requires_code == true, include at least one correct code snippet.

Style:
- Short paragraphs
- Use bullets where useful
- Use code blocks properly
- Avoid fluff

-------------------------
OUTPUT RULES

- Output ONLY markdown
- Do NOT explain anything
"""


def worker_node(payload: dict) -> dict:

    task = Task(**payload["task"])
    plan = Plan(**payload["plan"])
    evidence = [EvidenceItem(**e) for e in payload.get("evidence", [])]

    topic = payload["topic"]

    # -------------------------
    # STEP 1: Build query for RAG
    # -------------------------
    rag_query = f"{topic} - {task.title}. {task.goal}"

    # -------------------------
    # STEP 2: Call RAG pipeline
    # -------------------------
    try:
        rag_result = rag_app.invoke({
            "question": rag_query,
            "retriever": payload.get("retriever") 
        })

        context = rag_result.get("refined_context", "")
        verdict = rag_result.get("verdict", "UNKNOWN")

    except Exception as e:
        print("RAG ERROR:", e)
        context = ""
        verdict = "FAILED"

    # -------------------------
    # STEP 3: Fallback handling
    # -------------------------
    if not context or verdict == "INCORRECT":
        context = "No reliable external context found. Use general knowledge carefully."

    # Optional: truncate context (avoid token overflow)
    context = context[:3000]

    # -------------------------
    # Prepare bullets
    # -------------------------
    bullets_text = "\n- " + "\n- ".join(task.bullets)

    # -------------------------
    # STEP 4: Generate section using context
    # -------------------------
    response = llm.invoke([
        SystemMessage(content=WORKER_SYSTEM),
        HumanMessage(content=f"""
Topic: {topic}

Section: {task.title}
Goal: {task.goal}
Words: {task.target_words}

Bullets:
{bullets_text}

Context (use this for grounding):
{context}

Verdict: {verdict}
""")
    ])

    section_md = response.content.strip()

    return {"sections": [(task.id, section_md)]}