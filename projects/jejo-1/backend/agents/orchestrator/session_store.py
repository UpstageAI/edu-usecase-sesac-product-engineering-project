# backend/agents/orchestrator/session_store.py

import json
import uuid
import time
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod

import redis

from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationStore(ABC):
    """Abstract base class for conversation storage"""

    @abstractmethod
    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session state by session_id"""
        pass

    @abstractmethod
    def save_state(self, session_id: str, state: Dict[str, Any]):
        """Save session state"""
        pass

    @abstractmethod
    def delete_state(self, session_id: str):
        """Delete session state"""
        pass

    @abstractmethod
    def create_new_session_id(self) -> str:
        """Generate a new unique session ID"""
        pass


class InMemoryConversationStore(ConversationStore):
    """
    In-memory session store using Python dict.

    WARNING: All sessions are lost on server restart.
    Use only for development/testing.
    """

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(session_id)

    def save_state(self, session_id: str, state: Dict[str, Any]):
        self._store[session_id] = state

    def delete_state(self, session_id: str):
        if session_id in self._store:
            del self._store[session_id]

    def create_new_session_id(self) -> str:
        return str(uuid.uuid4())


class RedisConversationStore(ConversationStore):
    """
    Redis-based session store with automatic expiration.

    Features:
    - Persistent storage across server restarts
    - Automatic session TTL (time-to-live)
    - Connection pooling for performance
    - JSON serialization for complex state
    """

    def __init__(self, settings=None):
        if settings is None:
            settings = get_settings()

        self.settings = settings
        self.ttl = settings.session_ttl  # seconds

        # Initialize Redis connection
        try:
            if settings.redis_url:
                # Use redis_url if provided (e.g., for cloud services)
                self.redis_client = redis.from_url(
                    settings.redis_url,
                    decode_responses=True,
                    encoding='utf-8'
                )
            else:
                # Use individual connection parameters.
                # NOTE:
                # Passing `ssl` to low-level ConnectionPool can break on 일부 redis-py
                # connection classes. Construct Redis client directly for compatibility.
                redis_kwargs = {
                    "host": settings.redis_host,
                    "port": settings.redis_port,
                    "db": settings.redis_db,
                    "decode_responses": True,
                    "encoding": "utf-8",
                    "max_connections": 10,
                }
                if settings.redis_password:
                    redis_kwargs["password"] = settings.redis_password
                if settings.redis_ssl:
                    redis_kwargs["ssl"] = True

                self.redis_client = redis.Redis(**redis_kwargs)

            # Test connection
            self.redis_client.ping()
            logger.info("Redis connection established (TTL=%ss)", self.ttl)

        except redis.ConnectionError as e:
            logger.warning("Redis connection failed: %s", e)
            logger.warning("Falling back to InMemoryConversationStore")
            raise

    def _make_key(self, session_id: str) -> str:
        """Create Redis key with namespace prefix"""
        return f"session:{session_id}"

    def get_state(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            key = self._make_key(session_id)
            data = self.redis_client.get(key)

            if data is None:
                return None

            # Deserialize JSON
            state = json.loads(data)
            return state

        except redis.RedisError as e:
            logger.warning("Redis get_state error for %s: %s", session_id, e)
            return None
        except json.JSONDecodeError as e:
            logger.warning("JSON decode error for session %s: %s", session_id, e)
            return None

    def save_state(self, session_id: str, state: Dict[str, Any]):
        try:
            key = self._make_key(session_id)

            # Serialize to JSON
            data = json.dumps(state, ensure_ascii=False)

            # Save with TTL
            self.redis_client.setex(
                name=key,
                time=self.ttl,
                value=data
            )

        except redis.RedisError as e:
            logger.warning("Redis save_state error for %s: %s", session_id, e)
        except (TypeError, ValueError) as e:
            logger.warning("JSON encode error for session %s: %s", session_id, e)

    def delete_state(self, session_id: str):
        try:
            key = self._make_key(session_id)
            self.redis_client.delete(key)

        except redis.RedisError as e:
            logger.warning("Redis delete_state error for %s: %s", session_id, e)

    def create_new_session_id(self) -> str:
        return str(uuid.uuid4())

    def extend_ttl(self, session_id: str):
        """
        Extend the TTL of an existing session.
        Useful for keeping active sessions alive.
        """
        try:
            key = self._make_key(session_id)
            self.redis_client.expire(key, self.ttl)
        except redis.RedisError as e:
            logger.warning("Redis extend_ttl error for %s: %s", session_id, e)

    def get_all_session_ids(self) -> list[str]:
        """
        Get all active session IDs.
        WARNING: Use with caution in production (can be slow with many sessions).
        """
        try:
            keys = self.redis_client.keys("session:*")
            return [key.replace("session:", "") for key in keys]
        except redis.RedisError as e:
            logger.warning("Redis get_all_session_ids error: %s", e)
            return []


def create_conversation_store() -> ConversationStore:
    """
    Factory function to create the appropriate conversation store.

    Returns:
        - RedisConversationStore if use_redis_session=True and Redis is available
        - InMemoryConversationStore otherwise (fallback for dev/testing)
    """
    settings = get_settings()

    if settings.use_redis_session:
        try:
            store = RedisConversationStore(settings)
            logger.info("Using RedisConversationStore for session management")
            return store
        except Exception as e:
            logger.warning("Failed to initialize Redis: %s", e)
            logger.info("Falling back to InMemoryConversationStore")

    logger.info("Using InMemoryConversationStore (development mode)")
    return InMemoryConversationStore()
