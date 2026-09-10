import os

from agno.db.redis import RedisDb
from dotenv import load_dotenv, find_dotenv

from semantic_memory.qdrant_db import SemanticLongTermMemory

load_dotenv(find_dotenv())


class ShortTermMemory:
    def __init__(self, time_to_live: int | None = None):
        # Setup Redis - configured from .env (falls back to defaults)
        self.redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379")
        self.time_to_live = (
            time_to_live
            if time_to_live is not None
            else int(os.environ.get("STM_TTL_SECONDS", "60"))
        )
        self.stm_db = RedisDb(db_url=self.redis_url, expire=self.time_to_live)

    def memory(self) -> RedisDb:
        return self.stm_db


class LongTermMemory:
    def __init__(self):
        # All Qdrant / model settings are read from .env inside SemanticLongTermMemory
        self.ltm_db = SemanticLongTermMemory()

    def memory(self) -> SemanticLongTermMemory:
        return self.ltm_db