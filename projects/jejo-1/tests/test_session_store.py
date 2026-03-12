# tests/test_session_store.py

import pytest
import time
import json
from typing import Dict, Any

from backend.agents.orchestrator.session_store import (
    InMemoryConversationStore,
    RedisConversationStore,
    create_conversation_store,
)
from backend.config import Settings


class TestInMemoryConversationStore:
    """Test cases for InMemoryConversationStore"""

    def test_create_and_get_session(self):
        store = InMemoryConversationStore()
        session_id = store.create_new_session_id()

        # Initially, session should not exist
        assert store.get_state(session_id) is None

        # Save session state
        state = {
            "active_agent": "quiz",
            "conversation_history": [{"role": "User", "content": "Hello"}],
            "agent_specific_state": {},
            "last_interaction_timestamp": time.time(),
        }
        store.save_state(session_id, state)

        # Retrieve session state
        retrieved = store.get_state(session_id)
        assert retrieved is not None
        assert retrieved["active_agent"] == "quiz"
        assert len(retrieved["conversation_history"]) == 1

    def test_update_session(self):
        store = InMemoryConversationStore()
        session_id = store.create_new_session_id()

        # Save initial state
        state = {"active_agent": "quiz", "conversation_history": []}
        store.save_state(session_id, state)

        # Update state
        state["conversation_history"].append({"role": "User", "content": "New message"})
        store.save_state(session_id, state)

        # Verify update
        retrieved = store.get_state(session_id)
        assert len(retrieved["conversation_history"]) == 1

    def test_delete_session(self):
        store = InMemoryConversationStore()
        session_id = store.create_new_session_id()

        state = {"active_agent": "quiz"}
        store.save_state(session_id, state)
        assert store.get_state(session_id) is not None

        # Delete session
        store.delete_state(session_id)
        assert store.get_state(session_id) is None

    def test_nonexistent_session(self):
        store = InMemoryConversationStore()
        assert store.get_state("nonexistent-id") is None


class TestRedisConversationStore:
    """Test cases for RedisConversationStore"""

    @pytest.fixture
    def redis_store(self):
        """Fixture to create a Redis store for testing"""
        # Create test settings
        settings = Settings(
            use_redis_session=True,
            redis_host="localhost",
            redis_port=6379,
            redis_password="",
            redis_db=1,  # Use a separate DB for testing
            session_ttl=10,  # Short TTL for testing
        )

        try:
            store = RedisConversationStore(settings)
            yield store
            # Cleanup: delete all test sessions after each test
            # (Redis DB 1 should only contain test data)
        except Exception as e:
            pytest.skip(f"Redis not available: {e}")

    def test_create_and_get_session(self, redis_store):
        session_id = redis_store.create_new_session_id()

        # Initially, session should not exist
        assert redis_store.get_state(session_id) is None

        # Save session state
        state = {
            "active_agent": "quiz",
            "conversation_history": [{"role": "User", "content": "Hello"}],
            "agent_specific_state": {},
            "last_interaction_timestamp": time.time(),
        }
        redis_store.save_state(session_id, state)

        # Retrieve session state
        retrieved = redis_store.get_state(session_id)
        assert retrieved is not None
        assert retrieved["active_agent"] == "quiz"
        assert len(retrieved["conversation_history"]) == 1

        # Cleanup
        redis_store.delete_state(session_id)

    def test_session_ttl(self, redis_store):
        """Test that sessions expire after TTL"""
        session_id = redis_store.create_new_session_id()

        state = {"active_agent": "quiz", "data": "test"}
        redis_store.save_state(session_id, state)

        # Session should exist immediately
        assert redis_store.get_state(session_id) is not None

        # Wait for TTL to expire (11 seconds > 10 second TTL)
        time.sleep(11)

        # Session should be expired
        assert redis_store.get_state(session_id) is None

    def test_extend_ttl(self, redis_store):
        """Test TTL extension"""
        session_id = redis_store.create_new_session_id()

        state = {"active_agent": "quiz"}
        redis_store.save_state(session_id, state)

        # Wait 5 seconds (half of TTL)
        time.sleep(5)

        # Extend TTL
        redis_store.extend_ttl(session_id)

        # Wait another 7 seconds (total 12s, but TTL was extended at 5s)
        time.sleep(7)

        # Session should still exist (extended to 5s + 10s = 15s total)
        assert redis_store.get_state(session_id) is not None

        # Cleanup
        redis_store.delete_state(session_id)

    def test_complex_state_serialization(self, redis_store):
        """Test that complex state (nested dicts, lists) is preserved"""
        session_id = redis_store.create_new_session_id()

        state = {
            "active_agent": "riskmanaging",
            "conversation_history": [
                {"role": "User", "content": "안녕하세요"},
                {"role": "Agent", "content": "네, 무엇을 도와드릴까요?"},
            ],
            "agent_specific_state": {
                "analysis_in_progress": True,
                "extracted_data": {
                    "contract_amount": 100000,
                    "supplier": "A사",
                },
            },
            "last_interaction_timestamp": 1234567890.123,
        }

        redis_store.save_state(session_id, state)
        retrieved = redis_store.get_state(session_id)

        assert retrieved["active_agent"] == "riskmanaging"
        assert len(retrieved["conversation_history"]) == 2
        assert retrieved["agent_specific_state"]["analysis_in_progress"] is True
        assert retrieved["agent_specific_state"]["extracted_data"]["contract_amount"] == 100000

        # Cleanup
        redis_store.delete_state(session_id)


class TestConversationStoreFactory:
    """Test factory function"""

    def test_creates_inmemory_by_default(self):
        """Test that factory creates InMemory store when use_redis_session=False"""
        # Default settings should have use_redis_session=False
        store = create_conversation_store()
        assert isinstance(store, InMemoryConversationStore)

    # NOTE: Testing Redis creation requires Redis to be running
    # and use_redis_session=True in environment


if __name__ == "__main__":
    # Quick manual test
    print("Testing InMemoryConversationStore...")
    store = InMemoryConversationStore()
    sid = store.create_new_session_id()
    print(f"Created session ID: {sid}")

    state = {"active_agent": "quiz", "data": "test"}
    store.save_state(sid, state)
    retrieved = store.get_state(sid)
    print(f"Retrieved state: {retrieved}")

    print("\n✅ InMemoryConversationStore test passed!")

    print("\nTesting factory function...")
    factory_store = create_conversation_store()
    print(f"Factory created: {type(factory_store).__name__}")
    print("✅ Factory test passed!")
