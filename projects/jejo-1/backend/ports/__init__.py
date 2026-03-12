"""
Ports (Interfaces)

Clean Architecture의 내부 레이어.
비즈니스 로직이 외부 프레임워크에 의존하지 않도록 추상화 제공.
"""
from backend.ports.llm_gateway import LLMGateway
from backend.ports.document_retriever import DocumentRetriever

__all__ = ["LLMGateway", "DocumentRetriever"]
