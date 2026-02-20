import numpy as np
from typing import List, Dict, Any
from query_emb import embed_texts, qvec
from tc_emb import chunk_texts, chunk_metas, embeddings

from app.agents.rag_agent.constants import TOP_K


def cosine_topk(query_vec: np.ndarray, doc_vecs: np.ndarray, top_k: int = TOP_K):
    scores = doc_vecs @ query_vec

    # 점수가 큰 순서대로 top_k개의 인덱스를 구하세요. np.argsort는 오름차순이므로 내림차순이 되게 조정해야 합니다.
    top_idx = np.argsort(-scores)[:top_k]
    return [(int(i), float(scores[i])) for i in top_idx]


def retrieve_manual(query: str, top_k: int) -> List[Dict[str, Any]]:
    # qvec = np.array(embed_texts([query])[0], dtype=np.float32) # 사용자 질문에 대한 임베딩 벡터 생성
    # 위의 cosine_topk를 활용하여 top 결과를 받으세요.
    top = cosine_topk(qvec, embeddings, top_k=top_k)
    # 만들어둔 문서 텍스트 청크의 임베딩과의 유사도 점수 계산 후 top_k 개의 문서 텍스트 청크 인덱스와 점수 반환
    results = []
    for i, s in top:
        results.append(
            {
                "rank": len(results) + 1,
                "score": s,
                "text": chunk_texts[i],
                "meta": chunk_metas[i],
            }
        )
    return results


# cosine_topk는 질문 벡터(query_vec)과 문서 벡터(doc_vec) 간의 유사도를 계산합니다.
# top_k에 해당하는 문서의 인덱스와, 질문과의 유사도 점수를 반환합니다.
