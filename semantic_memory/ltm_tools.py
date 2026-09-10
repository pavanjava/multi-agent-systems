"""
Agno Toolkit exposing the Qdrant-backed SemanticLongTermMemory as tools.

Follows https://docs.agno.com/reference/tools/toolkit : subclass Toolkit,
register bound methods, and let Agno derive the tool schema from docstrings.
"""

import json
from typing import Optional

from agno.tools import Toolkit

from semantic_memory.memory_util import LongTermMemory


class LongTermMemoryTools(Toolkit):
    def __init__(
            self,
            user_id: Optional[str] = None,
            default_limit: int = 5,
            **kwargs,
    ):
        """
        Args:
            user_id: If set, every search/store is scoped to this user via payload filter.
            default_limit: Default number of memories returned by a search.
        """
        self.user_id = user_id
        self.default_limit = default_limit
        self.ltm = LongTermMemory().memory()  # SemanticLongTermMemory, configured from .env

        super().__init__(
            name="long_term_memory_tools",
            tools=[self.search_long_term_memory, self.store_long_term_memory],
            instructions=(
                "Call search_long_term_memory BEFORE answering whenever the user refers to "
                "earlier cases, prior findings, previous conversations, or asks a follow-up. "
                "Treat returned memories as prior context, not as new evidence."
            ),
            add_instructions=True,
            **kwargs,
        )

    def search_long_term_memory(self, query: str, limit: Optional[int] = None) -> str:
        """
        Semantically search long-term memory for prior notes, findings, or conversations
        related to the query. Uses hybrid (dense + sparse) retrieval over Qdrant.

        Args:
            query: Natural-language description of what to recall (e.g. "earlier findings on the patient with night sweats").
            limit: Maximum number of memories to return.

        Returns:
            JSON string: list of {id, text, score, metadata}. Empty list if nothing relevant.
        """
        filters = {"user_id": self.user_id} if self.user_id else None
        results = self.ltm.retrieve(
            query=query,
            limit=limit or self.default_limit,
            filters=filters,
        )
        return json.dumps(results, default=str)

    def store_long_term_memory(self, text: str, tags: Optional[str] = None) -> str:
        """
        Persist an important finding, decision, or summary to long-term memory so it can be
        recalled in future sessions.

        Args:
            text: The content to remember. Write it as a self-contained note.
            tags: Optional comma-separated tags (e.g. "diagnosis,follow-up").

        Returns:
            JSON string with the stored memory id.
        """
        metadata = {}
        if self.user_id:
            metadata["user_id"] = self.user_id
        if tags:
            metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]
        point_id = self.ltm.insert(text=text, metadata=metadata or None)
        return json.dumps({"stored": True, "id": point_id})