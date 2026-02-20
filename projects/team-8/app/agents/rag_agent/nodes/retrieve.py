from app.agents.rag_agent.state.rag_state import RagState

from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings
from app.agents.document_parser.nodes.vector_store import setup_vector_store
from app.agents.rag_agent.constants import TOP_K


def retrieve_normal_by_single_query(state: RagState) -> RagState:
    # rprint("retrieve input state", state)

    underlying_embeddings = load_underlying_embeddings()

    vector_store = setup_vector_store(
        underlying_embeddings=underlying_embeddings,
        collection_name="terms_normal_tag_dense",
    )

    search_result = vector_store.similarity_search_by_vector(
        state.user_query_embedding, k=TOP_K
    )
    # rprint(">>> search_result", [document.page_content for document in search_result])

    return {"terms_normal_tag_dense": search_result}


def retrieve_simple_by_single_query(state: RagState) -> RagState:
    # rprint("retrieve input state", state)

    underlying_embeddings = load_underlying_embeddings()

    vector_store = setup_vector_store(
        underlying_embeddings=underlying_embeddings,
        collection_name="terms_simple_tag_dense",
    )

    search_result = vector_store.similarity_search_by_vector(
        state.user_query_embedding, k=TOP_K
    )
    # rprint(">>> search_result", [document.page_content for document in search_result])

    return {"terms_simple_tag_dense": search_result}
