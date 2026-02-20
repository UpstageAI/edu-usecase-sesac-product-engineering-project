import sys
import os
import json
from typing import List, Dict, Any, Optional

# Ensure backend directory is in path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.rag.chroma_client import get_or_create_collection
from backend.rag.embedder import get_embedding
from backend.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

def search(query: str, k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a similarity search against the Chroma vector store.

    Args:
        query (str): The search query.
        k (int): The number of top results to retrieve.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a retrieved document
                               with its content and metadata.
    """
    collection = get_or_create_collection()
    query_embedding = get_embedding(query)

    if query_embedding is None:
        logger.warning("Could not generate embedding for query in search()")
        return []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=['documents', 'metadatas', 'distances']
    )

    retrieved_docs = []
    if results and results['documents']:
        for i in range(len(results['documents'][0])):
            doc = {
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            }
            retrieved_docs.append(doc)
    return retrieved_docs

def search_with_filter(
    query: str,
    k: int = 5,
    category: Optional[str] = None,
    priority: Optional[str] = None,
    level: Optional[str] = None,
    role: Optional[str] = None,
    document_type: Optional[str] = None,
    topic: Optional[str] = None,
    situation: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Performs a similarity search with optional metadata filtering.

    Args:
        query (str): The search query.
        k (int): The number of top results to retrieve.
        category (Optional[str]): Filter by original_category.
        priority (Optional[str]): Filter by priority (e.g., "high", "critical").
        level (Optional[str]): Filter by level (e.g., "beginner", "manager").
        role (Optional[str]): Filter by role (e.g., "sales", "logistics").
        document_type (Optional[str]): Filter by document_type.
        topic (Optional[str]): Filter by topic (checks if topic is in the list of topics).
        situation (Optional[str]): Filter by situation (checks if situation is in the list of situations).

    Returns:
        List[Dict[str, Any]]: A list of dictionaries, each representing a retrieved document
                               with its content and metadata.
    """
    collection = get_or_create_collection()
    query_embedding = get_embedding(query)

    if query_embedding is None:
        logger.warning("Could not generate embedding for query in search_with_filter()")
        return []

    # where_clause = {}
    # if category:
    #     where_clause["original_category"] = category
    # if priority:
    #     where_clause["priority"] = priority
    # if level:
    #     where_clause["level"] = level
    # if document_type:
    #     where_clause["document_type"] = document_type
    # if role:
    #     where_clause["role"] = {"$contains": role} # Assuming role can be a list in metadata
    # if topic:
    #     where_clause["topic"] = {"$contains": topic} # Assuming topic can be a list in metadata
    # if situation:
    #     where_clause["situation"] = {"$contains": situation} # Assuming situation can be a list in metadata
        
        
        
    # 1. 일단 조건에 맞는 필터들을 리스트에 '수집'만 합니다.   #debug : docs where_clause 딕셔너리 만드는 부분을 아래처럼 바꿈
    filters = []
    
    if category:
        filters.append({"original_category": category})
    if priority:
        filters.append({"priority": priority})
    if level:
        filters.append({"level": level})
    if document_type:
        filters.append({"document_type": document_type})
    
    # 리스트형 데이터($contains) 처리도 동일하게 리스트에 추가
    if role:
        filters.append({"role": {"$contains": role}})
    if topic:
        filters.append({"topic": {"$contains": topic}})
    if situation:
        filters.append({"situation": {"$contains": situation}})

    # 2. 여기서 딱 한 번만 판단해서 where_clause를 만듭니다.
    if len(filters) > 1:
        # 필터가 여러 개면 $and 상자에 담기
        where_clause = {"$and": filters}
    elif len(filters) == 1:
        # 필터가 하나면 그 필터 그대로 사용
        where_clause = filters[0]
    else:
        # 필터가 없으면 None
        where_clause = None    

        
        
        


    logger.debug("Executing search_with_filter where=%s", where_clause if where_clause else "None")

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where_clause if where_clause else None,
        include=['documents', 'metadatas', 'distances']
    )

    retrieved_docs = []
    if results and results['documents']:
        for i in range(len(results['documents'][0])):
            doc = {
                "document": results['documents'][0][i],
                "metadata": results['metadatas'][0][i],
                "distance": results['distances'][0][i]
            }
            retrieved_docs.append(doc)
    return retrieved_docs


if __name__ == '__main__':
    print("--- Retriever Test ---")

    # Ensure environment variable is set for embedding
    settings = get_settings()
    if not settings.upstage_api_key:
        print("Please set the UPSTAGE_API_KEY in your .env file to run retriever tests.")
        print("Example: UPSTAGE_API_KEY='your_api_key'")
        sys.exit(1)

    # Test basic search
    print("\n--- Basic Search Test ---")
    query_basic = "선적 지연 발생시 어떻게 해야 하나요?"
    results_basic = search(query_basic, k=3)
    if results_basic:
        print(f"\nTop 3 results for '{query_basic}':")
        for i, doc in enumerate(results_basic):
            print(f"--- Result {i+1} (Distance: {doc['distance']:.2f}) ---")
            print(f"Document: {doc['document']}")
            print(f"Metadata: {json.dumps(doc['metadata'], indent=2, ensure_ascii=False)}")
    else:
        print("No results found for basic search.")

    # Test filtered search by category and priority
    print("\n--- Filtered Search Test (Category: mistakes, Priority: critical) ---")
    query_filtered = "BL에 오류가 있을 때 어떻게 처리해야 하나요?"
    results_filtered = search_with_filter(
        query_filtered,
        k=2,
        category="mistakes",
        priority="critical"
    )
    if results_filtered:
        print(f"\nTop 2 results for '{query_filtered}' (filtered: category=mistakes, priority=critical):")
        for i, doc in enumerate(results_filtered):
            print(f"--- Result {i+1} (Distance: {doc['distance']:.2f}) ---")
            print(f"Document: {doc['document']}")
            print(f"Metadata: {json.dumps(doc['metadata'], indent=2, ensure_ascii=False)}")
    else:
        print("No results found for filtered search.")
    
    # Test filtered search by role and level
    print("\n--- Filtered Search Test (Role: manager, Level: executive) ---")
    query_filtered_role_level = "CEO의 의사결정 스타일은 무엇인가요?"
    results_filtered_role_level = search_with_filter(
        query_filtered_role_level,
        k=2,
        role="executive",
        level="executive",
        document_type="ceo_guideline"
    )
    if results_filtered_role_level:
        print(f"\nTop 2 results for '{query_filtered_role_level}' (filtered: role=executive, level=executive, document_type=ceo_guideline):")
        for i, doc in enumerate(results_filtered_role_level):
            print(f"--- Result {i+1} (Distance: {doc['distance']:.2f}) ---")
            print(f"Document: {doc['document']}")
            print(f"Metadata: {json.dumps(doc['metadata'], indent=2, ensure_ascii=False)}")
    else:
        print("No results found for filtered search by role and level.")

    # Test filtered search by topic and situation
    print("\n--- Filtered Search Test (Topic: incoterms, Situation: negotiation) ---")
    query_filtered_topic_sit = "FOB 조건 변경에 대한 협상 전략은?"
    results_filtered_topic_sit = search_with_filter(
        query_filtered_topic_sit,
        k=2,
        topic="incoterms",
        situation="negotiation"
    )
    if results_filtered_topic_sit:
        print(f"\nTop 2 results for '{query_filtered_topic_sit}' (filtered: topic=incoterms, situation=negotiation):")
        for i, doc in enumerate(results_filtered_topic_sit):
            print(f"--- Result {i+1} (Distance: {doc['distance']:.2f}) ---")
            print(f"Document: {doc['document']}")
            print(f"Metadata: {json.dumps(doc['metadata'], indent=2, ensure_ascii=False)}")
    else:
        print("No results found for filtered search by topic and situation.")
