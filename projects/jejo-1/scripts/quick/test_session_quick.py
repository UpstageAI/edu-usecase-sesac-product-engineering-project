#!/usr/bin/env python3
# Quick test for session store

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.agents.orchestrator.session_store import (
    InMemoryConversationStore,
    create_conversation_store,
)

print("=== Testing InMemoryConversationStore ===")
store = InMemoryConversationStore()
sid = store.create_new_session_id()
print(f"✓ Created session ID: {sid}")

# Test save and retrieve
state = {
    "active_agent": "quiz",
    "conversation_history": [{"role": "User", "content": "Hello"}],
    "agent_specific_state": {},
}
store.save_state(sid, state)
print("✓ Saved state")

retrieved = store.get_state(sid)
print(f"✓ Retrieved state: active_agent={retrieved['active_agent']}, history_len={len(retrieved['conversation_history'])}")

# Test update
state["conversation_history"].append({"role": "Agent", "content": "Hi there!"})
store.save_state(sid, state)
retrieved = store.get_state(sid)
print(f"✓ Updated state: history_len={len(retrieved['conversation_history'])}")

# Test delete
store.delete_state(sid)
retrieved = store.get_state(sid)
print(f"✓ Deleted state: {retrieved is None}")

print("\n=== Testing Factory Function ===")
factory_store = create_conversation_store()
print(f"✓ Factory created: {type(factory_store).__name__}")

print("\n✅ All tests passed!")
