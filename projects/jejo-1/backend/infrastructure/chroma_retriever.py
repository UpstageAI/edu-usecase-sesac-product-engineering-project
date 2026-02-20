"""
ChromaDB Document Retriever Implementation

ChromaDB를 사용한 DocumentRetriever 구현체.
"""
import logging
from typing import List, Dict, Any, Optional
import chromadb
import requests

from backend.ports.document_retriever import (
    DocumentRetriever,
    RetrievedDocument,
    RetrievalError
)
from backend.config import Settings, get_settings


logger = logging.getLogger(__name__)


class UpstageEmbeddingFunction:
    """
    Upstage Solar Embedding Function for ChromaDB

    ChromaDB 검색 시 Upstage API를 사용하여 쿼리를 임베딩.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.upstage.ai/v1/embeddings"
        self.model = "embedding-query"

    def name(self) -> str:
        """ChromaDB가 요구하는 이름 반환"""
        return "upstage-solar-embedding"

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB가 호출하는 임베딩 함수

        Args:
            input: 임베딩할 텍스트 리스트

        Returns:
            List[List[float]]: 각 텍스트에 대한 임베딩 벡터
        """
        embeddings = []

        for text in input:
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "input": text,
                    "model": self.model
                }

                response = requests.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=10
                )
                response.raise_for_status()

                data = response.json()
                if data and "data" in data and len(data["data"]) > 0:
                    embedding = data["data"][0]["embedding"]
                    embeddings.append(embedding)
                else:
                    logger.error(f"Unexpected API response: {data}")
                    raise RetrievalError("Failed to get embedding from Upstage API")

            except Exception as e:
                logger.error(f"Embedding API call failed: {e}")
                raise RetrievalError(f"Embedding generation failed: {e}")

        return embeddings


class ChromaDocumentRetriever(DocumentRetriever):
    """
    ChromaDB를 사용한 문서 검색

    Features:
        - 메타데이터 필터링 (document_type 등)
        - 유사도 기반 검색
        - 컬렉션 통계

    Example:
        retriever = ChromaDocumentRetriever(get_settings())
        docs = retriever.search("FOB 조건", k=3, document_type="email")
    """

    def __init__(self, settings: Settings):
        """
        Args:
            settings: 애플리케이션 설정 (API 키, ChromaDB 경로 등)
        """
        self._settings = settings

        try:
            # Upstage Embedding Function 초기화 (쿼리 임베딩용)
            self._embedding_function = UpstageEmbeddingFunction(
                api_key=settings.upstage_api_key
            )

            # ChromaDB 클라이언트 초기화 (rag/chroma_client.py와 동일한 설정)
            self._client = chromadb.PersistentClient(
                path="backend/vectorstore"  # 설정 파라미터 제거 (singleton 충돌 방지)
            )

            # 컬렉션 로드 (embedding function은 지정하지 않음 - 기존 설정 유지)
            self._collection = self._client.get_or_create_collection(
                name="trade_coaching_knowledge"  # rag/chroma_client.py와 동일한 컬렉션명
            )

            doc_count = self._collection.count()
            logger.info(f"ChromaDocumentRetriever initialized: {doc_count} documents")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise RetrievalError(f"ChromaDB initialization failed: {e}")

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
            List[RetrievedDocument]: 검색된 문서 (유사도 순)

        Raises:
            RetrievalError: 검색 실패
        """
        try:
            # 메타데이터 필터 구성
            where = {}
            if document_type:
                where["document_type"] = document_type
            where.update(filters)

            # 쿼리를 Upstage API로 임베딩
            logger.debug(f"Embedding query: '{query[:50]}...'")
            query_embeddings = self._embedding_function([query])

            # 검색 실행 (임베딩 벡터로 직접 검색)
            logger.debug(f"Searching: k={k}, where={where}")

            results = self._collection.query(
                query_embeddings=query_embeddings,
                n_results=k,
                where=where if where else None
            )

            # RetrievedDocument 객체로 변환
            documents = []
            if results["documents"] and results["documents"][0]:
                for i in range(len(results["documents"][0])):
                    doc = RetrievedDocument(
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i] if results.get("metadatas") else {},
                        distance=results["distances"][0][i] if results.get("distances") else 0.0
                    )
                    documents.append(doc)

            logger.info(f"Found {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise RetrievalError(f"Document search failed: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 반환

        Returns:
            Dict: {"total_documents": int, "document_types": List[str]}
        """
        try:
            total_docs = self._collection.count()

            # 모든 메타데이터 가져오기 (샘플링)
            sample = self._collection.get(limit=1000)
            document_types = set()

            if sample.get("metadatas"):
                for metadata in sample["metadatas"]:
                    if "document_type" in metadata:
                        document_types.add(metadata["document_type"])

            return {
                "total_documents": total_docs,
                "document_types": sorted(list(document_types))
            }

        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_documents": 0,
                "document_types": [],
                "error": str(e)
            }

    def __repr__(self) -> str:
        try:
            count = self._collection.count()
            return f"<ChromaDocumentRetriever docs={count}>"
        except:
            return "<ChromaDocumentRetriever (not initialized)>"
