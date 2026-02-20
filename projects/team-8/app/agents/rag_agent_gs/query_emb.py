import os, getpass
from dotenv import load_dotenv
from typing import List, Dict, Any
from langchain_upstage import UpstageEmbeddings
from input_query import build_insurance_query
from mock import create_mock_vet_agent_state
import numpy as np

mock_state = create_mock_vet_agent_state()
query = build_insurance_query(mock_state)
# print(query)
load_dotenv()    
EMBED_MODEL_NAME = "solar-embedding-1-large" 
upstage_embeddings = UpstageEmbeddings(
    api_key=os.getenv("UPSTAGE_API_KEY"),
    model=EMBED_MODEL_NAME
    )

def embed_texts(texts: List[str]):
    ## UpstageEmbeddings는 리스트 입력(texts)을 embed_documents로 처리합니다.
    return upstage_embeddings.embed_documents(texts)
#query = input("반려동물 보험에 관해 질문을 입력하세요: ")

#vec = upstage_embeddings.embed_query(query) 
qvec = np.array(embed_texts([query])[0], dtype=np.float32) # 사용자 질문에 대한 임베딩 벡터 생성    
# print("query embedding:", vec)
# print("embedding dim:", len(vec))
#print(vec, len(vec))
#print(qvec, qvec.shape)