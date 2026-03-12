"""
Document Retriever Port (Interface)

RAG 문서 검색을 추상화하여 비즈니스 로직이 특정 Vector DB에 의존하지 않도록 함.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class RetrievedDocument:
    """
    검색된 문서

    Attributes:
        content: 문서 내용
        metadata: 문서 메타데이터 (document_type, source 등)
        distance: 유사도 거리 (낮을수록 유사)
    """
    content: str
    metadata: Dict[str, Any]
    distance: float

    def is_relevant(self, threshold: float = 1.0) -> bool:
        """
        관련성 있는 문서인지 판단

        Args:
            threshold: 거리 임계값 (이하이면 관련성 있음)

        Returns:
            bool: True if distance <= threshold
        """
        return self.distance <= threshold


class DocumentRetriever(ABC):
    """
    문서 검색 추상화 인터페이스

    구현체:
        - ChromaDocumentRetriever: ChromaDB 기반
        - PineconeDocumentRetriever: Pinecone 기반
        - MockDocumentRetriever: 테스트용

    Example:
        retriever = ChromaDocumentRetriever(settings)
        docs = retriever.search(
            query="FOB 조건",
            k=3,
            document_type="email"
        )
        for doc in docs:
            print(doc.content, doc.distance)
    """

    @abstractmethod
    def search(
        self,
        query: str,
        k: int = 5,
        document_type: Optional[str] = None,
        **filters
    ) -> List[RetrievedDocument]:
        """
        유사도 기반 문서 검색

        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수
            document_type: 문서 타입 필터 (email, common_mistake 등)
            **filters: 추가 메타데이터 필터

        Returns:
            List[RetrievedDocument]: 검색된 문서 리스트 (유사도 순)

        Raises:
            RetrievalError: 검색 실패 시
        """
        pass

    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 정보 반환

        Returns:
            Dict: {"total_documents": int, "document_types": List[str]}
        """
        pass


class RetrievalError(Exception):
    """문서 검색 실패"""
    pass
