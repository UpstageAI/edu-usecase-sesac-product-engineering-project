# sparse embedding model - simple version for inference only
from typing import Dict, Any
from collections import defaultdict
from collections import Counter
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from app.agents.rag_agent.state.rag_state import RagState
from app.agents.rag_agent_gs.sparse import (
    match_predefined_words,
    tokenize_korean,
)

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(ENV_PATH)

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


BM25_K1 = 1.5
BM25_B = 0.75
VOCAB_BM25_WEIGHT = 0.6
PREDEFINED_BM25_WEIGHT = 1.8


def _get_terms_dir() -> Path:
    try:
        from app.agents.document_parser.constants import TERMS_DIR

        return Path(TERMS_DIR)
    except Exception:
        return (
            Path(__file__).resolve().parents[2] / "document_parser" / "data" / "terms"
        )


def _get_vocab_jsond_path_for_doc(file_name: str) -> Path | None:
    if not file_name:
        return None

    terms_dir = _get_terms_dir()
    doc_stem = Path(file_name).stem
    vocab_path = terms_dir / doc_stem / "vocab.jsond"
    if vocab_path.exists():
        return vocab_path

    return None


def _load_vocab_jsond(vocab_path: Path) -> Dict[str, Any]:
    try:
        with vocab_path.open("r", encoding="utf-8") as fp:
            return json.load(fp)
    except Exception:
        return {}


def _safe_avgdl(lengths: list[int]) -> float:
    if not lengths:
        return 1.0
    avgdl = sum(lengths) / len(lengths)
    return avgdl if avgdl > 0 else 1.0


def _bm25_tf_component(tf: int, dl: int, avgdl: float) -> float:
    if tf <= 0:
        return 0.0
    denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * (dl / max(avgdl, 1e-9)))
    if denominator <= 0:
        return 0.0
    return (tf * (BM25_K1 + 1)) / denominator


def _resolve_idf(term: str, idf: Dict[str, float]) -> float:
    term_lower = term.lower()
    if term_lower in idf:
        return float(idf[term_lower])

    split_terms = [token for token in term_lower.split() if token]
    if not split_terms:
        return 0.0
    split_idf = [float(idf[token]) for token in split_terms if token in idf]
    if not split_idf:
        return 0.0
    return sum(split_idf) / len(split_idf)


def _count_phrase_occurrences(text: str, phrase: str) -> int:
    text_lower = (text or "").lower()
    phrase_lower = (phrase or "").lower().strip()
    if not phrase_lower:
        return 0

    direct_count = text_lower.count(phrase_lower)

    text_no_space = text_lower.replace(" ", "")
    phrase_no_space = phrase_lower.replace(" ", "")
    compact_count = text_no_space.count(phrase_no_space) if phrase_no_space else 0

    return max(direct_count, compact_count)


def sparse_qt_score(state: RagState) -> Dict[str, Any]:
    """
    Extract insurance product information from retrieved documents.

    Processes the top 5 retrieved documents from RAG retrieval and extracts
    insurance product and insurer metadata for recommendation.

    Args:
        state: RagState containing retrieved_documents from vector DB search

    Returns:
        Updated state dict with:
        - "sparse_scoring_results": List of top 5 products with:
          {
            "product_name": str,
            "insurer": str,
            "metadata": dict
          }
    """

    vocab_cache: Dict[str, Dict[str, Any]] = {}

    tokenized_chunk_cache: list[list[str]] = []
    doc_length_by_file: Dict[str, list[int]] = defaultdict(list)
    for document in state.retrieved_documents:
        chunk_tokens = tokenize_korean(document.page_content or "")
        tokenized_chunk_cache.append(chunk_tokens)
        doc_meta = document.metadata.get("doc", {})
        file_name = doc_meta.get("file_name", "")
        doc_length_by_file[file_name].append(max(len(chunk_tokens), 1))

    avgdl_by_file: Dict[str, float] = {
        file_name: _safe_avgdl(lengths)
        for file_name, lengths in doc_length_by_file.items()
    }

    for idx, document in enumerate(state.retrieved_documents):
        evaluation = document.metadata.setdefault("evaluation", {})
        query_text = evaluation.get("query_context", "")
        doc_meta = document.metadata.get("doc", {})
        file_name = doc_meta.get("file_name", "")
        vocab_path = _get_vocab_jsond_path_for_doc(file_name)

        if vocab_path is None:
            evaluation["sparse_score"] = 0.00
            evaluation["predefined_match_count"] = 0
            continue

        query_tokens = tokenize_korean(query_text)
        chunk_tokens = tokenized_chunk_cache[idx]
        chunk_token_counter = Counter(chunk_tokens)
        dl = max(len(chunk_tokens), 1)
        avgdl = avgdl_by_file.get(file_name, float(dl))

        vocab_key = str(vocab_path)
        if vocab_key not in vocab_cache:
            vocab_cache[vocab_key] = _load_vocab_jsond(vocab_path)

        vocab_data = vocab_cache[vocab_key]
        vocab = vocab_data.get("vocab", {})
        idf = vocab_data.get("idf")
        predefined_words = vocab_data.get("predefined_words", [])

        if not isinstance(idf, dict):
            idf = {token: 1.0 for token in vocab.keys()}

        matched_vocab_tokens = [t for t in query_tokens if t in vocab]
        unmatched_vocab_tokens = [t for t in query_tokens if t not in vocab]
        vocab_query_tf = Counter(token.lower() for token in matched_vocab_tokens)
        vocab_total = 0.0
        for token, qtf in vocab_query_tf.items():
            tf = chunk_token_counter.get(token, 0)
            if tf <= 0:
                continue
            token_idf = _resolve_idf(token, idf)
            bm25_tf = _bm25_tf_component(tf=tf, dl=dl, avgdl=avgdl)
            vocab_total += qtf * token_idf * bm25_tf

        matched_predefined = match_predefined_words(query_text, predefined_words)
        predefined_query_tf = Counter(term.lower() for term in matched_predefined)
        predefined_total = 0.0
        for term, qtf in predefined_query_tf.items():
            tf = _count_phrase_occurrences(document.page_content or "", term)
            if tf <= 0:
                continue
            term_idf = _resolve_idf(term, idf)
            bm25_tf = _bm25_tf_component(tf=tf, dl=dl, avgdl=avgdl)
            predefined_total += qtf * term_idf * bm25_tf

        sparse_score = (
            VOCAB_BM25_WEIGHT * vocab_total + PREDEFINED_BM25_WEIGHT * predefined_total
        )
        predefined_match_count = len(matched_predefined)

        evaluation["sparse_score"] = round(sparse_score, 2)
        evaluation["predefined_match_count"] = predefined_match_count
        # evaluation["sparse_debug"] = {
        #     "query_text": query_text,
        #     "query_tokens": query_tokens,
        #     "vocab_path": str(vocab_path),
        #     "matched_vocab_tokens": matched_vocab_tokens,
        #     "unmatched_vocab_tokens": unmatched_vocab_tokens,
        #     "matched_predefined": matched_predefined,
        #     "matched_predefined_count": predefined_match_count,
        #     "vocab_bm25_total": vocab_total,
        #     "predefined_bm25_total": predefined_total,
        #     "weights": {
        #         "vocab": VOCAB_BM25_WEIGHT,
        #         "predefined": PREDEFINED_BM25_WEIGHT,
        #     },
        #     "bm25": {"k1": BM25_K1, "b": BM25_B, "dl": dl, "avgdl": avgdl},
        # }

    if state.retrieved_documents:
        # print("\n[Sparse score results - top 5 chunks]")
        first_query_printed = False
        for idx, document in enumerate(state.retrieved_documents[:5], start=1):
            evaluation = document.metadata.get("evaluation", {})
            debug_info = evaluation.get("sparse_debug", {})
            doc_meta = document.metadata.get("doc", {})
            indexing_meta = document.metadata.get("indexing", {})
            file_name = doc_meta.get("file_name", "")
            page = doc_meta.get("page", "")
            chunk_id = doc_meta.get("chunk_id") or indexing_meta.get("chunk_id") or ""

            if not first_query_printed:
                # print(f"query: {debug_info.get('query_text', '')}")
                first_query_printed = True

            # print(f"\n{idx}. file={file_name} page={page} chunk_id={chunk_id}")
            chunk_content = " ".join(document.page_content.split())
            # print(f"  chunk_content: {chunk_content}")

            if isinstance(debug_info, dict):
                query_tokens = debug_info.get("query_tokens", [])
                matched_vocab = debug_info.get("matched_vocab_tokens", [])
                unmatched_vocab = debug_info.get("unmatched_vocab_tokens", [])
                query_tokens_str = ", ".join(query_tokens)
                matched_vocab_str = ", ".join(matched_vocab)
                unmatched_vocab_str = ", ".join(unmatched_vocab)
                # print(f"  query_tokens: [{query_tokens_str}]")
                # print(f"  matched_vocab_tokens: [{matched_vocab_str}]")
                # print(f"  unmatched_vocab_tokens: [{unmatched_vocab_str}]")

            total_score = evaluation.get("total_score", "N/A")
            sparse_score = evaluation.get("sparse_score", "N/A")
            # print(f"  evaluation: ({total_score}, {sparse_score})")

    return {"retrieved_documents": state.retrieved_documents}
