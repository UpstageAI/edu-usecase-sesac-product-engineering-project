import argparse

from langgraph.graph.state import StateGraph, CompiledStateGraph, START, END

from app.agents.document_parser.nodes import document_parser
from app.agents.document_parser.nodes.splitter import page_splitter, text_splitter
from app.agents.document_parser.nodes.tagger import chunk_loader
from app.agents.document_parser.nodes.tagger import tagger_normal, tagger_simple
from app.agents.document_parser.nodes import vector_store
from app.agents.document_parser.state.document_parser_state import DocumentParserState

from app.agents import utils


def build_graph() -> CompiledStateGraph:
    workflow = StateGraph(DocumentParserState)

    workflow.add_node("document_parser", document_parser.document_parser_node)

    workflow.add_edge(START, "document_parser")
    workflow.add_edge("document_parser", END)

    graph = workflow.compile()

    utils.create_graph_image(
        graph,
        utils.get_current_file_name(__file__, True),
        utils.get_parent_path(__file__),
    )

    return graph


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument(
        "--file-name",
        required=True,
        help="PDF file name  with extension",
    )
    parser.add_argument(
        "--tag-type",
        required=True,
        choices=["normal", "simple"],
        help="metadata tag type",
    )
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="해당 옵션이 포함된 경우 Vector DB 적재까지 진행",
    )
    parser.add_argument(
        "--basic-term-start",
        type=int,
        required=True,
        help="보통약관 시작 페이지 번호(footer 기준)",
    )
    parser.add_argument(
        "--basic-term-end",
        type=int,
        required=True,
        help="보통약관 종료 페이지 번호(footer 기준)",
    )
    parser.add_argument(
        "--special-term-start",
        type=int,
        required=True,
        help="특별약관 시작 페이지 번호(footer 기준)",
    )
    parser.add_argument(
        "--special-term-end",
        type=int,
        required=True,
        help="특별약관 종료 페이지 번호(footer 기준)",
    )
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    file_name: str = args.file_name
    tag_type: str = args.tag_type

    tagged_chunks = chunk_loader.load_chunk_files(
        file_name=file_name, tag_type=tag_type
    )
    if tagged_chunks:
        print(f"✅load existing chunks: {len(tagged_chunks)} (tag_type={tag_type})")
    else:
        dp_result = document_parser.parse_document(file_name)

        pages = page_splitter.split_pages_and_add_metadata(
            dp_result,
            file_name,
            basic_term_start=args.basic_term_start,
            basic_term_end=args.basic_term_end,
            special_term_start=args.special_term_start,
            special_term_end=args.special_term_end,
        )

        chunks = text_splitter.split(pages)

        if tag_type == "normal":
            tagged_chunks = tagger_normal.tag_chunks(chunks)
        elif tag_type == "simple":
            tagged_chunks = tagger_simple.tag_chunks(chunks)

    if args.ingest:
        if tag_type == "normal":
            vector_store.ingest_chunks("terms_normal_tag_dense", tagged_chunks)
        elif tag_type == "simple":
            vector_store.ingest_chunks("terms_simple_tag_dense", tagged_chunks)


# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_1_maum_pet_1_21_22_50.pdf --basic-term-start 1 --basic-term-end 21 --special-term-start 22 --special-term-end 50 --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_2_petpermint_cat_family_45_82_83_206.pdf --basic-term-start 45 --basic-term-end 82 --special-term-start 83 --special-term-end 206 --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name meritz_3_petpermint_dog_family_51_86_87_230.pdf --basic-term-start 51 --basic-term-end 86 --special-term-start 87 --special-term-end 230 --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_1_dog_anypet_3_20_21_47.pdf --basic-term-start 3 --basic-term-end 20 --special-term-start 21 --special-term-end 47 --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_2_cat_anypet_3_20_21_37.pdf --basic-term-start 3 --basic-term-end 20 --special-term-start 21 --special-term-end 37 --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_3_direct_good_pet_28_42_45_105.pdf --basic-term-start 28 --basic-term-end 42 --special-term-start 45 --special-term-end 105 --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_4_pet_majestic_dog_33_49_52_139.pdf --basic-term-start 33 --basic-term-end 49 --special-term-start 52 --special-term-end 139 --tag-type simple
# uv run python -m app.agents.document_parser.dp_graph --file-name samsung_5_pet_majestic_cat_31_47_50_159.pdf --basic-term-start 31 --basic-term-end 47 --special-term-start 50 --special-term-end 159 --tag-type simple
