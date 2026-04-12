from langchain_core.messages import SystemMessage, HumanMessage
from config.llm import llm
from schemas.models import RouterDecision, State
from utils.structured_output import parse_output

ROUTER_SYSTEM = """You are a routing module for a technical blog planner.

Decide whether web research is needed BEFORE planning.

Modes:
- closed_book (needs_research=false):
  Evergreen topics where correctness does not depend on recent facts (concepts, fundamentals).
- hybrid (needs_research=true):
  Mostly evergreen but needs up-to-date examples/tools/models to be useful.
- open_book (needs_research=true):
  Mostly volatile: weekly roundups, "this week", "latest", rankings, pricing, policy/regulation.

If needs_research=true:
- Output 3–10 high-signal queries.
- Queries should be scoped and specific (avoid generic queries like just "AI" or "LLM").
- For open_book weekly roundup, include queries that reflect the last 7 days constraint.

-------------------------
OUTPUT FORMAT (VERY IMPORTANT)

Return ONLY valid JSON. No explanation. No markdown.

Example:
{
  "needs_research": true,
  "mode": "hybrid",
  "reason": "topic needs recent examples",
  "queries": ["query1", "query2"],
  "max_results_per_query": 5
}

STRICT RULES:
- Do NOT write anything before JSON
- Do NOT write anything after JSON
- Do NOT explain anything
- Output must start with { and end with }
"""

def router_node(state: State) -> dict:
    response = llm.invoke([
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=f"Topic: {state['topic']}\nAs-of date: {state['as_of']}")
    ]).content

    decision = parse_output(response, RouterDecision)

    if decision.mode == "open_book":
        recency_days = 7
    elif decision.mode == "hybrid":
        recency_days = 45
    else:
        recency_days = 3650

    return {
        "needs_research": decision.needs_research,
        "mode": decision.mode,
        "queries": decision.queries,
        "recency_days": recency_days,
    }

def route_next(state: State) -> str:
    return "research" if state["needs_research"] else "orchestrator"