"""
LTM recall check: run after clinical_diagnostic_support.py has stored a report.
Asks a paraphrased question and shows whether the team pulled it from Qdrant.
"""

import os
import asyncio
from pathlib import Path

from agno.agent import Agent
from agno.models.google import Gemini
from agno.skills import LocalSkills, Skills
from agno.team import Team
from dotenv import find_dotenv, load_dotenv

from semantic_memory.ltm_tools import LongTermMemoryTools
from semantic_memory.memory_util import ShortTermMemory

load_dotenv(find_dotenv())

user_id = "7f3a9c2e8b1d4f6a"  # same user_id as clinical_diagnostic_support.py
SKILLS_DIR = Path(__file__).parent / "skills"

short_term_memory = ShortTermMemory()

clinical_guidelines_agent = Agent(
    name="Clinical Guidelines Agent",
    model=Gemini(id="gemini-2.5-flash", api_key=os.environ.get("GEMINI_API_KEY")),
    role="Extract evidence-based guidelines, dosage protocols, and contraindications",
    db=short_term_memory.memory(),
    user_id=user_id,
)

medical_research_team = Team(
    name="Medical Research Team",
    model=Gemini(id="gemini-2.5-flash", api_key=os.environ.get("GEMINI_API_KEY")),
    members=[clinical_guidelines_agent],
    tools=[LongTermMemoryTools(user_id=user_id, default_limit=3)],
    skills=Skills(loaders=[LocalSkills(str(SKILLS_DIR))]),
    instructions=[
        "Before delegating or answering, call search_long_term_memory with the user's question.",
        "Load the memory-recall-format skill to present anything recalled.",
    ],
    db=short_term_memory.memory(),
    user_id=user_id,
    markdown=True,
    debug_mode=True,
)

if __name__ == "__main__":
    # Paraphrased on purpose - avoids the wording used in the stored report
    query = (
        "What did we conclude earlier for my middle-aged diabetic patient "
        "with B-symptoms and a family history of blood cancer?"
    )

    response = asyncio.run(medical_research_team.arun(input=query, user_id=user_id))

    tool_calls = [t.tool_name for t in (getattr(response, "tools", None) or [])]
    print("\n[leader tool calls]", tool_calls)
    print("[LTM used]", "search_long_term_memory" in tool_calls)
    print("\n", response.content)