from __future__ import annotations

import os
import asyncio
from pathlib import Path

from agno.agent import Agent
from agno.context.fs import FilesystemContextProvider
from agno.models.openai import OpenAIResponses
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# Create the provider
# ---------------------------------------------------------------------------
fs = FilesystemContextProvider(
    id="filesystem-context",
    root=Path(__file__).resolve().parent,
    model=OpenAIResponses(id="gpt-5.6-luna"),
)

# ---------------------------------------------------------------------------
# Create the Agent
# ---------------------------------------------------------------------------
agent = Agent(
    model=OpenAIResponses(id="gpt-5.4"),
    tools=fs.get_tools(),
    instructions=fs.instructions(),
    markdown=True,
)


# ---------------------------------------------------------------------------
# Run the Agent
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\nfs.status() = {fs.status()}\n")
    prompt = (
        "Give me the treatment guidelines for AML."
    )
    print(f"> {prompt}\n")
    asyncio.run(agent.aprint_response(prompt))