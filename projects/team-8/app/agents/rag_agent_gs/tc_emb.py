from tc_chunk import chunks
from query_emb import embed_texts
import numpy as np

chunk_texts = [c["text"] for c in chunks] # 이전에 만든 청크들을 불러옵니다.
chunk_metas = [{"id": c["id"], "page": c["page"]} for c in chunks] # 각 청크의 메타 데이터(아이디, 페이지 번호)를 불러옵니다.

#위에서 만든 임베딩 함수를 이용하여, pdf 문서를 청킹해서 만든 각각의 텍스트 청크에 대한 임베딩 수행합니다.
embeddings = embed_texts(chunk_texts)
embeddings = np.array(embeddings, dtype=np.float32)
# print("embeddings shape:", embeddings.shape)