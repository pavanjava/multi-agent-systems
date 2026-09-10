"""
Standalone medical research agent.

Capabilities
------------
1. Reasoning + planning before searching   -> ReasoningTools (think / analyze scratchpad)
2. Domain-restricted Tavily search         -> custom tool pinned to peer-reviewed / authoritative sources
3. Structured output with citations        -> Pydantic schema with per-finding evidence grade
4. Memory reuse                            -> ShortTermMemory (session/agentic memory) + LongTermMemory (insert on completion)
"""
import asyncio
import os
from typing import List, Literal, Optional

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.reasoning import ReasoningTools
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, Field
from tavily import TavilyClient

from semantic_memory.memory_util import LongTermMemory, ShortTermMemory

load_dotenv(find_dotenv())

# --------------------------------------------------------------------------- #
# Memory (same objects as the workflow file)
# --------------------------------------------------------------------------- #
short_term_memory = ShortTermMemory(time_to_live=120)
long_term_memory = LongTermMemory()
user_id = "7f3a9c2e8b1d4f6a"

# --------------------------------------------------------------------------- #
# Domain-restricted Tavily search tool
# --------------------------------------------------------------------------- #
TRUSTED_MEDICAL_DOMAINS = [
    "pubmed.ncbi.nlm.nih.gov",
    "pmc.ncbi.nlm.nih.gov",
    "nih.gov",
    "who.int",
    "cochranelibrary.com",
    "cdc.gov",
    "nice.org.uk",
    "bmj.com",
    "thelancet.com",
    "nejm.org",
    "jamanetwork.com",
]

_tavily = TavilyClient(api_key=os.environ.get("TAVILY_API_KEY"))


def search_medical_literature(query: str, max_results: int = 6) -> str:
    """
    Search ONLY trusted medical sources (PubMed, PMC, NIH, WHO, Cochrane, CDC, NICE,
    BMJ, Lancet, NEJM, JAMA). Use this for every factual medical claim.

    Args:
        query: A focused search query (one clinical question per call).
        max_results: Number of results to return (1-10).
    Returns:
        Numbered list of results with title, URL, published date and content snippet.
    """
    response = _tavily.search(
        query=query,
        search_depth="advanced",
        max_results=max(1, min(max_results, 10)),
        include_domains=TRUSTED_MEDICAL_DOMAINS,
        include_answer=False,
    )
    results = response.get("results", [])
    if not results:
        return "No results from trusted medical domains for this query. Reformulate and retry."

    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(
            f"[{i}] {r.get('title', 'Untitled')}\n"
            f"    URL: {r.get('url')}\n"
            f"    Published: {r.get('published_date', 'n/a')}\n"
            f"    Snippet: {(r.get('content') or '').strip()[:700]}"
        )
    return "\n\n".join(lines)


# --------------------------------------------------------------------------- #
# Structured output schema
# --------------------------------------------------------------------------- #
EvidenceGrade = Literal["A", "B", "C", "D"]


class Citation(BaseModel):
    title: str
    url: str
    source: str = Field(description="Publishing body, e.g. PubMed, WHO, Cochrane")
    year: Optional[int] = None
    study_type: Optional[str] = Field(
        default=None,
        description="e.g. systematic review, RCT, cohort, guideline, expert opinion",
    )


class Finding(BaseModel):
    claim: str = Field(description="One clinically meaningful statement")
    evidence_grade: EvidenceGrade = Field(
        description="A = systematic review / meta-analysis / major guideline; "
                    "B = RCT or large cohort; C = observational / small study; D = expert opinion / mechanistic"
    )
    citations: List[Citation] = Field(min_length=1)
    caveats: Optional[str] = None


class MedicalEvidenceReport(BaseModel):
    question: str
    search_plan: List[str] = Field(description="The sub-queries the agent decided to run")
    executive_summary: str
    findings: List[Finding]
    conflicting_evidence: Optional[str] = None
    limitations: str
    overall_confidence: Literal["high", "moderate", "low"]
    disclaimer: str = (
        "Informational synthesis of published literature; not a substitute for professional medical advice."
    )


# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
medical_research_agent = Agent(
    name="Medical Evidence Researcher",
    model=Gemini(id="gemini-3.1-flash-lite", api_key=os.environ.get("GEMINI_API_KEY")),
    tools=[
        ReasoningTools(add_instructions=True),
        search_medical_literature,
    ],
    description=(
        "An evidence-based medical research agent that plans, searches only authoritative "
        "sources, and grades the strength of every claim it reports."
    ),
    instructions=[
        "Workflow you MUST follow on every request:",
        "1. THINK: decompose the question into 2-5 precise sub-queries (population, intervention, "
        "comparator, outcome where applicable). Record them; they become `search_plan`.",
        "2. SEARCH: call `search_medical_literature` once per sub-query. Never answer from prior "
        "knowledge alone; every finding needs at least one returned citation.",
        "3. ANALYZE: assess study type, recency and consistency across sources. Assign an evidence "
        "grade A-D per finding using the schema definition. Flag disagreement explicitly.",
        "4. REPORT: fill the MedicalEvidenceReport schema. Prefer fewer, well-supported findings "
        "over many weak ones. Never fabricate URLs or study details.",
        "Use the user's stored context (name, prior questions, preferences) when relevant.",
    ],
    output_schema=MedicalEvidenceReport,
    db=short_term_memory.memory(),
    build_user_context=True,
    enable_agentic_memory=True,
    add_history_to_context=True,
    num_history_runs=3,
    user_id=user_id,
    markdown=False,
    debug_mode=False,
)


async def main() -> None:
    response = await medical_research_agent.arun(
        input=(
            "Trace my mempries historically about what i was asking and interacting"
        ),
        session_id="medical-session-001",
    )

    report: MedicalEvidenceReport = response.content
    print(report.model_dump_json(indent=2))

    long_term_memory.memory().insert(
        text=report.model_dump_json(),
        metadata={"user_id": user_id, "topic": report.question, "agent": "medical_research"},
    )


if __name__ == "__main__":
    asyncio.run(main())