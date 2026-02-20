from copy import deepcopy

from langchain_core.documents import Document
from qdrant_client import models as qdrant_models
from rich import print as rprint

from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings
from app.agents.document_parser.nodes.vector_store import setup_vector_store
from app.agents.rag_agent.constants import TOP_K
from app.agents.rag_agent.state.rag_state import RagState


def _build_search_filter() -> qdrant_models.Filter:
    return qdrant_models.Filter(
        must=[
            # qdrant_models.FieldCondition(
            #     key="metadata.doc.insurer_code",
            #     match=qdrant_models.MatchValue(value="samsung"),
            # ),
            # qdrant_models.FieldCondition(
            #     key="metadata.term_type",
            #     match=qdrant_models.MatchValue(value="basic"),
            # ),
            qdrant_models.FieldCondition(
                key="metadata.clause.clause_type",
                match=qdrant_models.MatchValue(value="coverage"),
            ),
        ]
    )


def _document_key(document: Document) -> tuple[str, tuple[tuple[str, str], ...]]:
    metadata_items = tuple(
        sorted(
            (str(key), str(value)) for key, value in (document.metadata or {}).items()
        )
    )
    return document.page_content, metadata_items


def _collect_query_texts(state: RagState) -> list[str]:
    query_texts = [
        query.strip() for query in (state.query_texts or []) if query and query.strip()
    ]
    if query_texts:
        return query_texts
    raise ValueError("invalid query_texts !")


def _collect_embeddings(state: RagState) -> list[list[float]]:
    if state.query_texts_embeddings:
        return state.query_texts_embeddings
    return []


def _search_by_embeddings(
    collection_name: str,
    query_embeddings: list[list[float]],
    k: int = TOP_K,
    search_filter: qdrant_models.Filter | None = None,
) -> list[Document]:
    underlying_embeddings = load_underlying_embeddings()
    vector_store = setup_vector_store(
        underlying_embeddings=underlying_embeddings,
        collection_name=collection_name,
    )

    merged_docs: list[Document] = []
    filter_metadata = (
        search_filter.model_dump(exclude_none=True)
        if search_filter is not None
        else None
    )

    for embedding in query_embeddings:
        docs = vector_store.similarity_search_by_vector(
            embedding, k=k, filter=search_filter
        )
        for doc in docs:
            metadata = dict(doc.metadata or {})
            metadata["filter"] = {
                "applied": search_filter is not None,
                "query_filter": deepcopy(filter_metadata),
            }
            merged_docs.append(
                Document(
                    page_content=doc.page_content,
                    metadata=metadata,
                )
            )

    return merged_docs


def retrieve_normal_by_multi_query_texts(state: RagState) -> RagState:
    return _retrieve_by_multi_query_texts(
        state,
        collection_name="terms_normal_tag_dense",
        collection_label="normal tag",
        filtered_result_key="terms_normal_tag_dense",
        unfiltered_result_key="terms_normal_tag_dense_unfiltered",
    )


def retrieve_simple_by_multi_query_texts(state: RagState) -> RagState:
    return _retrieve_by_multi_query_texts(
        state,
        collection_name="terms_simple_tag_dense",
        collection_label="simple tag",
        filtered_result_key="terms_simple_tag_dense",
        unfiltered_result_key="terms_simple_tag_dense_unfiltered",
    )


def _retrieve_by_multi_query_texts(
    state: RagState,
    *,
    collection_name: str,
    collection_label: str,
    filtered_result_key: str,
    unfiltered_result_key: str,
) -> RagState:
    rprint(f"🔎{collection_label} collection: retrieve start")
    query_texts_embeddings = _collect_embeddings(state)
    if not query_texts_embeddings:
        query_texts = _collect_query_texts(state)
        query_texts_embeddings = load_underlying_embeddings().embed_documents(
            query_texts
        )
    search_filter = _build_search_filter()
    filtered_search_result = _search_by_embeddings(
        collection_name,
        query_texts_embeddings,
        k=TOP_K,
        search_filter=search_filter,
    )
    # unfiltered_search_result = _search_by_embeddings(
    #     collection_name, query_texts_embeddings, k=TOP_K, search_filter=None
    # )
    unfiltered_search_result = []
    rprint(
        f"✅{collection_label} collection: retrieve complete "
        f"(top_k={TOP_K}, filtered={len(filtered_search_result)}, "
        f"unfiltered={len(unfiltered_search_result)})"
    )
    return {
        filtered_result_key: filtered_search_result,
        unfiltered_result_key: unfiltered_search_result,
    }
