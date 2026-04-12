from langchain_core.messages import SystemMessage, HumanMessage
from config.llm import llm
from schemas.models import EvidencePack, EvidenceItem, State
from utils.structured_output import parse_output
from tools.tavily_tool import tavily_search

from datetime import date, timedelta

RESEARCH_SYSTEM = """You are an expert research synthesizer for technical writing.

Given raw web search results, produce a deduplicated list of EvidenceItem objects.

Rules:
- Only include items with a non-empty url.
- Prefer relevant + authoritative sources.
- Extract published_at as ISO (YYYY-MM-DD) if possible.
- If unsure, set published_at=null.
- Deduplicate by URL.

-------------------------
OUTPUT FORMAT

Return ONLY valid JSON.

Example:
{
  "evidence": [
    {
      "title": "Example",
      "url": "https://example.com",
      "published_at": "2025-01-01",
      "snippet": "short text",
      "source": "website"
    }
  ]
}

STRICT RULES:
- Output ONLY JSON
- No explanation
- Must start with { and end with }
"""

def research_node(state: State) -> dict:
    queries = state.get("queries", [])[:10]

    raw_results = []
    for q in queries:
        raw_results.extend(tavily_search(q))

    if not raw_results:
        return {"evidence": []}

    response = llm.invoke([
        SystemMessage(content=RESEARCH_SYSTEM),
        HumanMessage(content=str(raw_results))
    ]).content

    pack = parse_output(response, EvidencePack)

    evidence = pack.evidence

    if state["mode"] == "open_book":
        as_of = date.fromisoformat(state["as_of"])
        cutoff = as_of - timedelta(days=state["recency_days"])

        evidence = [
            e for e in evidence
            if e.published_at and date.fromisoformat(e.published_at) >= cutoff
        ]

    return {"evidence": evidence}