"""
Agent With Instructions
=============================

Agent With Instructions Quickstart.
"""

from agno.agent import Agent
from agno.models.openai import OpenAIResponses
from dotenv import load_dotenv, find_dotenv
# from temporal import MemoryManager
#
# memory_manager = MemoryManager()
# memory_manager.remember()

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Agent Instructions
# ---------------------------------------------------------------------------
instructions = """\
You are a concise assistant.
Answer with exactly 3 bullet points when possible.\
"""

# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
agent = Agent(
    name="Instruction-Tuned Agent",
    model=OpenAIResponses(id="gpt-5.2"),
    instructions=instructions,
)

# ---------------------------------------------------------------------------
# Run Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    agent.print_response("How can I improve my Python debugging workflow?", stream=True)