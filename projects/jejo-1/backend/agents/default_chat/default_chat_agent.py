from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional, Tuple
import asyncio
from openai import OpenAI
from backend.config import get_settings

class DefaultChatAgent:
    """
    Enhanced DefaultChatAgent using LLM (Solar Pro) for natural conversations
    and system-specific knowledge retrieval.
    """

    agent_type: str = "default_chat"

    def __init__(self):
        self.settings = get_settings()
        self.client = None
        if self.settings.upstage_api_key:
            self.client = OpenAI(
                api_key=self.settings.upstage_api_key,
                base_url="https://api.upstage.ai/v1/solar"
            )
        self.system_prompt = self._load_system_prompt()

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the prompts directory."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Adjusted path to go from backend/agents/default_chat up to project root
        project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
        prompt_path = os.path.join(project_root, 'backend', 'prompts', 'default_chat_prompt.txt')

        if not os.path.exists(prompt_path):
            return "당신은 무역 업무 지원 플랫폼의 스마트 지식 가이드입니다."

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            print(f"Error loading default_chat_prompt: {e}")
            return "당신은 무역 업무 지원 플랫폼의 스마트지식 가이드입니다."

    async def run(
        self,
        user_input: str,
        conversation_history: List[Dict[str, str]],
        analysis_in_progress: bool = False,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Runs the LLM-based default chat interaction."""
        
        # Prepare messages including history
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add filtered history (last N turns)
        for turn in conversation_history[-10:]:
            # Map history roles to LLM roles
            role = "assistant" if turn.get("role") in ["Agent", "assistant"] else "user"
            messages.append({"role": role, "content": turn.get("content", "")})
            
        # Add current user input
        messages.append({"role": "user", "content": user_input})
        
        try:
            if not self.client:
                raise ValueError("LLM client not initialized (check API key)")

            response = self.client.chat.completions.create(
                model="solar-pro",
                messages=messages,
                temperature=0.7
            )
            
            response_message = response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"DefaultChatAgent Error: {e}")
            response_message = "죄송합니다. 현재 대화 시스템에 일시적인 문제가 발생했습니다. 무역 실무 관련 작업(리스크 분석, 이메일, 퀴즈)은 상단 메뉴를 통해 계속 이용하실 수 있습니다."

        # Keep history format as expected by Orchestrator
        updated_history = list(conversation_history)
        updated_history.append({"role": "User", "content": user_input})
        updated_history.append({"role": "Agent", "content": response_message})

        return {
            "response": {
                "response": response_message,
                "agent_type": self.agent_type,
                "metadata": {"mode": "llm_natural_chat"},
            },
            "conversation_history": updated_history,
            "analysis_in_progress": False,
        }

if __name__ == "__main__":
    # Simple self-test
    agent = DefaultChatAgent()
    test_q = "안녕! 리스크 점수가 15점이면 위험한 거야?"
    print(f"Query: {test_q}")
    result = asyncio.run(agent.run(test_q, []))
    print(f"Response: {result['response']['response']}")
