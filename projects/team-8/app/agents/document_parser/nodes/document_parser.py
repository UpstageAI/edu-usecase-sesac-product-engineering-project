import argparse
import threading
import time


from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_upstage.document_parse import OutputFormat
from langchain_upstage import UpstageDocumentParseLoader

from rich import print as rprint

from app.agents.document_parser.constants import TERMS_DIR
from app.agents.document_parser.state.document_parser_state import DocumentParserState


load_dotenv()


def document_parser_node(state: DocumentParserState):
    parse_document(state.file_name)


def parse_document(file_name: str, output_format: OutputFormat = "html") -> Document:
    TERM_FILE_PATH = TERMS_DIR / file_name
    rprint("🔗parse_document TERM_FILE_PATH:", TERM_FILE_PATH)

    dp_loader = UpstageDocumentParseLoader(
        file_path=str(TERM_FILE_PATH),
        output_format=output_format,
        coordinates=False,
    )

    rprint("🚀document parsing start. output format:", output_format)
    start_time = time.perf_counter()
    stop_event = threading.Event()

    def _print_progress():
        while not stop_event.wait(5):
            elapsed = time.perf_counter() - start_time
            rprint(f"⏳document parsing in progress... (elapsed: {elapsed:.2f}s)")

    progress_thread = threading.Thread(target=_print_progress, daemon=True)
    progress_thread.start()

    try:
        dp_result = (
            dp_loader.load()
        )  # UpstageDocumentParse는 전체 문서를 하나의 Document로 추출함
    finally:
        stop_event.set()
        progress_thread.join()

    elapsed = time.perf_counter() - start_time
    rprint(
        f"✅document parsing done. result length: {len(dp_result)} (elapsed: {elapsed:.2f}s)"
    )

    return dp_result[0]


def create_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run document parser graph.")
    parser.add_argument("--file-name", help="PDF file name with extension")
    return parser


if __name__ == "__main__":
    args = create_arg_parser().parse_args()
    file_name = args.file_name
    parse_document(file_name)


# uv run python -m app.agents.document_parser.nodes.document_parser --file-name meritz_1_maum_pet_12_16.pdf
