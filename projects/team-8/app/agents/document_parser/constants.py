from app.agents.utils import get_parent_path

BASE_DIR = get_parent_path(__file__)  # app/agents/document_parser
TERMS_DIR = BASE_DIR / "data" / "terms"
