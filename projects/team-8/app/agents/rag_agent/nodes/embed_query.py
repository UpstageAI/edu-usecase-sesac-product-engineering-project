from rich import print as rprint

from app.agents.document_parser.nodes.embeddings import load_underlying_embeddings

from app.agents.rag_agent.state.rag_state import RagState


def embed_query(state: RagState) -> RagState:
    # rprint(">>> embed_query input state", state)

    if not state.user_query:
        raise ValueError("invalid user_query !")

    underlying_embeddings = load_underlying_embeddings()
    user_query_embedding = underlying_embeddings.embed_query(state.user_query)
    return {"user_query_embedding": user_query_embedding}
