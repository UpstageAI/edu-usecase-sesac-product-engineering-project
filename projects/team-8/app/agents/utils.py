import io
import os
from pathlib import Path

from PIL import Image as PILImage

from langgraph.graph.state import CompiledStateGraph


def get_current_file_name(
    target_file_path: str, without_extension: bool = False
) -> str:
    file_name = os.path.basename(target_file_path)

    if without_extension:
        name_only, extension = os.path.splitext(file_name)
        return name_only
    else:
        return file_name


def get_parent_path(target_file_path: str):
    return Path(target_file_path).resolve().parent


def create_graph_image(
    graph: CompiledStateGraph,
    file_name: str,
    base_dir: str = None,
):
    png_data = graph.get_graph().draw_mermaid_png()

    img = PILImage.open(
        io.BytesIO(png_data)
    )  # 바이너리 데이터를 메모리 상의 이미지로 변환합니다.

    BASE_DIR = base_dir if base_dir else get_parent_path(__file__)
    img.save(f"{BASE_DIR}/{file_name}.png")

    # img.show()
