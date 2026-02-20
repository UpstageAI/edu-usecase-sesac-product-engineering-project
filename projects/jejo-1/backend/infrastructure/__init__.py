"""
Infrastructure (Adapters)

Clean Architecture의 외부 레이어.
Port 인터페이스의 구체적 구현체 (Upstage, ChromaDB 등).
"""
from backend.infrastructure.upstage_llm import UpstageLLMGateway
from backend.infrastructure.chroma_retriever import ChromaDocumentRetriever

__all__ = ["UpstageLLMGateway", "ChromaDocumentRetriever"]
