"""
Confirmation Required MCP Toolkit (fixed)
==========================================
NOTE: Before this code make sure the postgresql_mcp server is up and running.
Human-in-the-Loop: Adding User Confirmation to Tool Calls with MCP Servers.
"""

import asyncio

from agno.agent import Agent
from agno.db.sqlite import SqliteDb
from agno.models.google import Gemini
from agno.tools.mcp import MCPTools
from rich.console import Console
from rich.prompt import Prompt
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

console = Console()


async def main():
    # Keep ONE MCP session alive across the pause + continuation.
    async with MCPTools(
            transport="streamable-http",
            url="http://127.0.0.1:8000/mcp",
            requires_confirmation_tools=[
                "ListSchemas",
                "ListDatabases",
                "ListTables",
                "ListTableColumns",
                "ListTableRelations",
                "ListSchemaRelations",
            ],
    ) as mcp_tools:
        agent = Agent(
            model=Gemini(id="gemini-3.1-flash-lite"),
            tools=[mcp_tools],
            markdown=True,
            db=SqliteDb(db_file="tmp/confirmation_required_toolkit.db"),
        )
        # What is the table that saves the information about runs in mlflow?
        stream = agent.arun(
            "get me all registered models in mlflow?",
            stream=True,
            stream_events=True,
        )

        while True:
            paused = None
            async for event in stream:
                if event.event == "RunContent" and event.content:
                    print(event.content, end="", flush=True)
                elif event.event == "RunPaused":
                    paused = event
                elif event.event == "RunError":
                    print(f"\nRun failed: {event.content}")

            if paused is None:
                break  # run finished, no more pauses

            for requirement in paused.active_requirements:
                if not requirement.needs_confirmation:
                    raise RuntimeError("Unexpected requirement type; not handled here")
                execution = requirement.tool_execution
                console.print(
                    f"Tool name [bold blue]{execution.tool_name}({execution.tool_args})[/] requires confirmation."
                )
                choice = (
                    Prompt.ask("Do you want to continue?", choices=["y", "n"], default="y")
                    .strip()
                    .lower()
                )
                if choice == "y":
                    requirement.confirm()
                else:
                    requirement.reject()

            # Resume with the SAME session; loop back in case another pause follows
            stream = agent.acontinue_run(
                run_id=paused.run_id,
                requirements=paused.requirements,
                stream=True,
                stream_events=True,
            )

        print()  # final newline


if __name__ == "__main__":
    asyncio.run(main())