from langgraph.graph.state import CompiledStateGraph
from langgraph.graph import StateGraph, START, END

from app.agents import utils

from app.agents.rag_agent.nodes.embed_query_texts import embed_query_texts
from app.agents.rag_agent.nodes.generate_query_texts import generate_query_texts
from app.agents.rag_agent.nodes.generate_user_query import generate_user_query
from app.agents.rag_agent.nodes.retrieve_multi import (
    retrieve_normal_by_multi_query_texts,
    retrieve_simple_by_multi_query_texts,
)
from app.agents.rag_agent.nodes.summary_multi import summary_multi
from app.agents.rag_agent.state.rag_state import RagState

from app.agents.vet_agent.state import VetAgentState, DiseaseInfo
from app.agents.rag_agent.nodes.sparse_query_term_score import sparse_qt_score


class RagGraphState(VetAgentState, RagState): ...


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(
        RagGraphState,
        # input_schema=VetAgentState,
        # output_schema=RagState,
    )

    workflow.add_node("generate_query_texts", generate_query_texts)
    # workflow.add_node("generate_query_texts", generate_user_query)
    workflow.add_node("embed_query_texts", embed_query_texts)
    # workflow.add_node("retrieve_normal", retrieve_normal_by_multi_query_texts)
    workflow.add_node("retrieve_simple", retrieve_simple_by_multi_query_texts)
    workflow.add_node("summary", summary_multi)
    workflow.add_node("sparse_scoring", sparse_qt_score)

    workflow.add_edge(START, "generate_query_texts")
    workflow.add_edge("generate_query_texts", "embed_query_texts")
    # workflow.add_edge("embed_query_texts", "retrieve_normal")
    workflow.add_edge("embed_query_texts", "retrieve_simple")
    # workflow.add_edge("retrieve_normal", "summary")
    workflow.add_edge("retrieve_simple", "summary")
    workflow.add_edge("summary", "sparse_scoring")
    workflow.add_edge("sparse_scoring", END)

    return workflow.compile()


graph = build_graph()

if __name__ == "__main__":
    retrieve_graph = build_graph()

    utils.create_graph_image(
        retrieve_graph,
        utils.get_current_file_name(__file__, True),
        utils.get_parent_path(__file__),
    )

    result = retrieve_graph.invoke(
        VetAgentState(
            species="강아지",
            breed="치와와",
            age=10,
            gender="male",
            weight=10,
            health_condition={
                "frequent_illness_area": "다리를 자주 다치고 피부병이 가끔씩 생기는 것 같아요",
                "disease_surgery_history": "수술 이력은 없어요",
            },
            diseases=[
                DiseaseInfo(
                    name="슬개골 탈구",
                    incidence_rate="높음",
                    onset_period="전 연령",
                ),
                DiseaseInfo(
                    name="심장판막증",
                    incidence_rate="중간",
                    onset_period="7세 이상",
                ),
                DiseaseInfo(
                    name="치과 질환",
                    incidence_rate="중간",
                    onset_period="5세 이상",
                ),
                DiseaseInfo(
                    name="간질",
                    incidence_rate="낮음",
                    onset_period="1-5세",
                ),
            ],
        )
    )


# uv run python -m app.agents.rag_agent.rag_graph
