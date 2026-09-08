import os
from agno.agent import Agent
from agno.models.google import Gemini
from agno.run.base import RunStatus
from agno.tools.hackernews import HackerNewsTools
from dotenv import load_dotenv, find_dotenv

from temporal import MemoryManager, format_memories_for_prompt

load_dotenv(find_dotenv())

USER_ID = "user-123"
AGENT_ID = "hn-digest-agent"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def make_agent(instructions: str) -> Agent:
    return Agent(
        model=Gemini(id="gemini-2.5-flash-lite", api_key=GEMINI_API_KEY, thinking_budget=512),
        tools=[HackerNewsTools()],
        instructions=instructions,
        markdown=True,
    )


def run(agent: Agent, task: str):
    response = agent.run(task)
    if response.status == RunStatus.error:
        print(f"run failed: {response.content}")
        return None
    return response


memory = MemoryManager()

print("=== session 1: stating a preference ===")
preference = (
    "I only care about AI infra, databases, and distributed systems on HN. "
    "Skip crypto and NFT stuff entirely."
)
session1 = run(make_agent("Acknowledge the user's preference briefly."), preference)
if session1:
    print(session1.content)
    memory.remember(user_id=USER_ID, agent_id=AGENT_ID, content=preference)

print("\n=== session 2: fresh agent, no restated preference ===")
task2 = "What's trending on HN today?"
recalled = memory.recall(query_text=task2, user_id=USER_ID, agent_id=AGENT_ID, limit=5)
instructions = "Write a short digest of trending topics."
if recalled:
    instructions += (
            "\n\nYou have access to memory from past interactions with this user, retrieved "
            "below. Treat it as ground truth and apply it; do not claim you lack memory.\n\n"
            + format_memories_for_prompt(recalled)
    )
session2 = run(make_agent(instructions), task2)
if session2:
    print(session2.content)
    memory.remember(user_id=USER_ID, agent_id=AGENT_ID, content=session2.content)

memory.close()
