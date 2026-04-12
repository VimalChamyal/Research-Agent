from langchain_core.messages import SystemMessage, HumanMessage
from config.llm import llm
from schemas.models import Plan, State
from utils.structured_output import parse_output
from langgraph.types import Send

ORCH_SYSTEM = """You are a senior technical writer and developer advocate.
Your job is to produce a highly actionable outline for a technical blog post.

Hard requirements:
- Create 5–9 sections (tasks) suitable for the topic and audience.
- Each task must include:
  1) goal (1 sentence)
  2) 3–6 bullets that are concrete, specific, and non-overlapping
  3) target word count (120–550)

Flexibility:
- Do NOT use a fixed taxonomy unless it naturally fits.
- You may tag tasks (tags field), but tags are flexible.

Quality bar:
- Assume the reader is a developer; use correct terminology.
- Bullets must be actionable: build/compare/measure/verify/debug.
- Ensure the overall plan includes at least 2 of these somewhere:
  * minimal code sketch / MWE (set requires_code=True for that section)
  * edge cases / failure modes
  * performance/cost considerations
  * security/privacy considerations (if relevant)
  * debugging/observability tips

Grounding rules:
- Mode closed_book: keep it evergreen; do not depend on evidence.
- Mode hybrid:
  - Use evidence for up-to-date examples (models/tools/releases) in bullets.
  - Mark sections using fresh info as requires_research=True and requires_citations=True.
- Mode open_book (weekly news roundup):
  - Set blog_kind = "news_roundup".
  - Every section is about summarizing events + implications.
  - DO NOT include tutorial/how-to sections.
  - If evidence is empty, say "insufficient sources".

-------------------------
OUTPUT FORMAT (VERY IMPORTANT)

Return ONLY valid JSON.

Example:
{
  "blog_title": "Example Title",
  "audience": "Developers",
  "tone": "Technical",
  "blog_kind": "explainer",
  "constraints": [],
  "tasks": [
    {
      "id": 1,
      "title": "Intro",
      "goal": "Understand concept",
      "bullets": ["point1", "point2", "point3"],
      "target_words": 200,
      "tags": [],
      "requires_research": false,
      "requires_citations": false,
      "requires_code": false
    }
  ]
}

STRICT RULES:
- Output ONLY JSON
- No explanation
- No markdown
- Must start with { and end with }
"""

def orchestrator_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=ORCH_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}")
    ]).content

    plan = parse_output(response, Plan)

    if state["mode"] == "open_book":
        plan.blog_kind = "news_roundup"

    return {"plan": plan}

def fanout(state: State):
    return [
        Send("worker", {
            "task": task.model_dump(),
            "topic": state["topic"],
            "plan": state["plan"].model_dump(),
            "evidence": [e.model_dump() for e in state["evidence"]],
        })
        for task in state["plan"].tasks
    ]