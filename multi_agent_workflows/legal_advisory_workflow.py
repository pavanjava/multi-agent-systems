"""
use case of parallel execution of workflow
"""
import asyncio
import os

from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools
from agno.db.postgres import PostgresDb
from agno.workflow import Step, Workflow
from agno.workflow.parallel import Parallel
from dotenv import load_dotenv, find_dotenv

from semantic_memory.memory_util import ShortTermMemory, LongTermMemory

load_dotenv(find_dotenv())
db = PostgresDb(db_url=os.environ.get("DATABASE_URL"))

short_term_memory = ShortTermMemory(time_to_live=120)
long_term_memory = LongTermMemory()
user_id = '7f3a9c2e8b1d4f6a'

# Create specialized legal agents
case_researcher = Agent(
    name="Case Law Researcher",
    model=Gemini(id="gemini-3.1-flash", api_key=os.environ.get("GEMINI_API_KEY")),
    tools=[TavilyTools(api_key=os.environ.get("TAVILY_API_KEY"))],
    description="Specialized in finding relevant case law and legal precedents",
    instructions="Search for relevant case law, judicial opinions, and legal precedents",
    db=short_term_memory.memory(),
    build_user_context=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

statute_researcher = Agent(
    name="Statutory Researcher",
    model=Gemini(id="gemini-3.1-flash", api_key=os.environ.get("GEMINI_API_KEY")),
    tools=[TavilyTools(api_key=os.environ.get("TAVILY_API_KEY"))],
    description="Specialized in researching statutes, regulations, and legislative history",
    instructions="Search for applicable statutes, regulations, and legislative materials",
    db=short_term_memory.memory(),
    build_user_context=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

legal_analyst = Agent(
    name="Legal Analyst",
    model=Gemini(id="gemini-3.1-flash", api_key=os.environ.get("GEMINI_API_KEY")),
    description="Analyzes legal research and synthesizes findings into coherent legal arguments",
    instructions="Synthesize research findings into a comprehensive legal memorandum with clear arguments and citations",
    db=short_term_memory.memory(),
    build_user_context=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

compliance_reviewer = Agent(
    name="Compliance Reviewer",
    model=Gemini(id="gemini-3.1-flash", api_key=os.environ.get("GEMINI_API_KEY")),
    description="Reviews legal documents for accuracy, completeness, and ethical compliance",
    instructions="Review the legal memorandum for accuracy, cite-checking, and compliance with professional standards",
    db=short_term_memory.memory(),
    build_user_context=True,
    enable_agentic_memory=True,
    user_id=user_id,
)

# Create individual steps
case_research_step = Step(
    name="Research Case Law",
    agent=case_researcher
)

statute_research_step = Step(
    name="Research Statutes",
    agent=statute_researcher
)

analysis_step = Step(
    name="Legal Analysis",
    agent=legal_analyst
)

review_step = Step(
    name="Compliance Review",
    agent=compliance_reviewer
)

# Create workflow with parallel research phase
legal_workflow = Workflow(
    name="Legal Research & Memorandum Pipeline",
    steps=[
        Parallel(
            case_research_step,
            statute_research_step,
            name="Legal Research Phase"
        ),
        analysis_step,
        review_step,
    ],
    db=short_term_memory.memory(),
)

if __name__ == "__main__":
    response = asyncio.run(
        legal_workflow.arun(input="My name is Advocate. Pavan Mantha, Analyze the legal implications of using AI-generated content "
                                  "in commercial products, focusing on copyright and liability issues. "
                                  "Focus exclusive in Indian copy right and content act also any other laws from India."
                            )
    )

    print(response.content)
    long_term_memory.memory().insert(text=response.content, metadata={'user_id': user_id})