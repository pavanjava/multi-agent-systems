from agno.agent import Agent
from agno.knowledge.embedder.ollama import OllamaEmbedder
from agno.knowledge.knowledge import Knowledge
from agno.models.openai import OpenAIResponses
from agno.team import Team
from agno.tools.reasoning import ReasoningTools
from agno.vectordb.qdrant import Qdrant
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
knowledge = Knowledge(
    vector_db=Qdrant(
        collection="mortgage_1",
        url="http://localhost:6333",
        api_key="th3s3cr3tk3y",
        embedder=OllamaEmbedder(id="embeddinggemma:latest",dimensions=768,)
    ),
)
# only uncomment for the first time load
# knowledge.insert(path="/Users/pavanmantha/Pavans/Workshops/multi_agent_systems/data/Leukemia Guide.pdf")

# ---------------------------------------------------------------------------
# Create Members
# ---------------------------------------------------------------------------
information_gatherer = Agent(
    name="Information Gatherer",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Gather comprehensive information from knowledge sources",
    knowledge=knowledge,
    search_knowledge=True,
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Search the knowledge base thoroughly for all relevant information.",
        "Use reasoning tools to plan your search strategy.",
        "Gather comprehensive context and supporting details.",
        "Document all sources and evidence found.",
    ],
    markdown=True,
)

reasoning_analyst = Agent(
    name="Reasoning Analyst",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Apply logical reasoning to analyze gathered information",
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Analyze information using structured reasoning approaches.",
        "Identify logical connections and relationships.",
        "Apply deductive and inductive reasoning where appropriate.",
        "Break down complex topics into logical components.",
        "Use reasoning tools to structure your analysis.",
    ],
    markdown=True,
)

evidence_evaluator = Agent(
    name="Evidence Evaluator",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Evaluate evidence quality and identify information gaps",
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Evaluate the quality and reliability of gathered evidence.",
        "Identify gaps in information or reasoning.",
        "Assess the strength of logical connections.",
        "Highlight areas needing additional clarification.",
        "Use reasoning tools to structure your evaluation.",
    ],
    markdown=True,
)

response_coordinator = Agent(
    name="Response Coordinator",
    model=OpenAIResponses(id="gpt-5.2"),
    role="Coordinate team findings into comprehensive reasoned response",
    tools=[ReasoningTools(add_instructions=True)],
    instructions=[
        "Synthesize all team member contributions into a coherent response.",
        "Ensure logical flow and consistency across the response.",
        "Include proper citations and evidence references.",
        "Present reasoning chains clearly and transparently.",
        "Use reasoning tools to structure the final response.",
    ],
    markdown=True,
)

# ---------------------------------------------------------------------------
# Create Team
# ---------------------------------------------------------------------------
coordinated_reasoning_team = Team(
    name="Coordinated Reasoning RAG Team",
    model=OpenAIResponses(id="gpt-5.2"),
    members=[
        information_gatherer,
        reasoning_analyst,
        evidence_evaluator,
        response_coordinator,
    ],
    instructions=[
        "Work together to provide comprehensive, well-reasoned responses.",
        "Information Gatherer: First search and gather all relevant information.",
        "Reasoning Analyst: Then apply structured reasoning to analyze the information.",
        "Evidence Evaluator: Evaluate the evidence quality and identify any gaps.",
        "Response Coordinator: Finally synthesize everything into a clear, reasoned response.",
        "All agents should use reasoning tools to structure their contributions.",
        "Show your reasoning process transparently in responses.",
    ],
    show_members_responses=True,
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run Team
# ---------------------------------------------------------------------------
async def async_reasoning_demo() -> None:
    print("Async Coordinated Reasoning RAG Team Demo")
    print("=" * 60)

    query = "What two criteria must be met for an exchange of related Classes within a Series to be permitted?"

    await coordinated_reasoning_team.aprint_response(
        query,
        stream=True,
        show_full_reasoning=True,
    )


def sync_reasoning_demo() -> None:
    print("Coordinated Reasoning RAG Team Demo")
    print("=" * 50)

    query = "What two criteria must be met for an exchange of related Classes within a Series to be permitted?"

    coordinated_reasoning_team.print_response(
        query,
        stream=True,
        show_full_reasoning=True,
    )


if __name__ == "__main__":
    # asyncio.run(async_reasoning_demo())
    sync_reasoning_demo()