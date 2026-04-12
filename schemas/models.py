from __future__ import annotations

import operator
from typing import TypedDict, List, Optional, Literal, Annotated

from pydantic import BaseModel, Field


# -----------------------------
# Task Schema
# -----------------------------
class Task(BaseModel):
    id: int
    title: str

    goal: str = Field(
        ...,
        description="One sentence describing what the reader should understand."
    )

    bullets: List[str] = Field(
        ...,
        min_length=3,
        max_length=6,
        description="3–6 concrete subpoints."
    )

    target_words: int = Field(
        ...,
        description="Target word count (120–550)."
    )

    tags: List[str] = Field(default_factory=list)
    requires_research: bool = False
    requires_citations: bool = False
    requires_code: bool = False


# -----------------------------
# Plan Schema
# -----------------------------
class Plan(BaseModel):
    blog_title: str
    audience: str
    tone: str

    blog_kind: Literal[
        "explainer",
        "tutorial",
        "news_roundup",
        "comparison",
        "system_design"
    ] = "explainer"

    constraints: List[str] = Field(default_factory=list)
    tasks: List[Task]


# -----------------------------
# Evidence Schema
# -----------------------------
class EvidenceItem(BaseModel):
    title: str
    url: str
    published_at: Optional[str] = None
    snippet: Optional[str] = None
    source: Optional[str] = None


# -----------------------------
# Router Output
# -----------------------------
class RouterDecision(BaseModel):
    needs_research: bool
    mode: Literal["closed_book", "hybrid", "open_book"]
    reason: str
    queries: List[str] = Field(default_factory=list)
    max_results_per_query: int = 5


# -----------------------------
# Evidence Pack
# -----------------------------
class EvidencePack(BaseModel):
    evidence: List[EvidenceItem] = Field(default_factory=list)


# -----------------------------
# Graph State
# -----------------------------
class State(TypedDict):
    topic: str

    mode: str
    needs_research: bool
    queries: List[str]
    evidence: List[EvidenceItem]
    plan: Optional[Plan]

    as_of: str
    recency_days: int

    sections: Annotated[List[tuple[int, str]], operator.add]
    final: str