from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings

from app.agents.rag_agent.state.rag_state import RagState


def embed_query_texts(state: RagState) -> RagState:
    # rprint(">>> embed_query_texts input state", state)

    query_texts = [
        query.strip() for query in (state.query_texts or []) if query and query.strip()
    ]
    if not query_texts:
        raise ValueError("invalid query_texts !")

    underlying_embeddings = load_underlying_embeddings()
    query_texts_embeddings = underlying_embeddings.embed_documents(query_texts)
    return {"query_texts_embeddings": query_texts_embeddings}
