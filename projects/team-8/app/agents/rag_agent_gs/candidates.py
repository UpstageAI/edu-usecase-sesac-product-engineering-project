# !pip install -U pip setuptools wheel
# !pip install -U "langchain>=0.2" "langchain-core>=0.2" "langchain-community>=0.2" "langchain-text-splitters>=0.2"
# !pip install -U pypdf
# !pip install langchain-upstage
# !pip install -U chromadb "langchain-chroma>=0.1"
# import os, getpass
# from dotenv import load_dotenv
#
# from typing import List, Dict, Any
# from pypdf import PdfReader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_upstage import ChatUpstage
# from langchain_core.prompts import PromptTemplate
# from langchain_chroma import Chroma
# from langchain_core.documents import Document

# from enum import Enum
# from pydantic import BaseModel, Field
import textwrap
from cos_top import retrieve_manual
from input_query import build_insurance_query
from mock import create_mock_vet_agent_state
import time

from app.agents.rag_agent.constants import TOP_K

# start_time = time.time()  # 측정 시작


mock_state = create_mock_vet_agent_state()
query = build_insurance_query(mock_state)
print("query:", query)
hits = retrieve_manual(
    query=query, top_k=TOP_K
)  # 위의 함수를 사용하여 코사인 유사도 검사를 수행,top-k retrieve
for h in hits:
    print(
        f"\n[Rank {h['rank']}] score={h['score']:.4f} page={h['meta']['page']} id={h['meta']['id']}"
    )
    print(textwrap.shorten(h["text"].replace("\n", " "), width=250, placeholder=" ..."))

    # ================================================================
#     end_time = time.time()    # 측정 종료
# print(f"실행 시간: {end_time - start_time:.5f} 초")
# load_dotenv()
# # assert UPSTAGE_API_KEY, "UPSTAGE_API_KEY가 설정되지 않았습니다. 위 셀을 다시 실행하세요."
# # print("UPSTAGE_API_KEY set ✅ (앞 4글자):", UPSTAGE_API_KEY[:4], "****")
# EMBED_MODEL_NAME = "solar-embedding-1-large"
# upstage_embeddings = UpstageEmbeddings(
#     api_key=os.getenv("UPSTAGE_API_KEY"),
#     model=EMBED_MODEL_NAME
#     )
# llm = ChatUpstage(
#     api_key=os.getenv("UPSTAGE_API_KEY"),
#       model="solar-pro2"
#       )
# 여기에 PDF 경로를 지정하세요.

# print(f"생성된 청크 개수: {len(chunks)}")
# print("sample chunk:\n")
# print("id:", chunks[7]["id"], "page:", chunks[7]["page"])
# print(chunks[7]["text"][:300])


# ## UpstageEmbeddings는 리스트 입력(texts)을 embed_documents로 처리합니다.
# def embed_texts(texts: List[str]):
#     return upstage_embeddings.embed_documents(texts)
# #query = input("반려동물 보험에 관해 질문을 입력하세요: ")

# vec = upstage_embeddings.embed_query(query)
# UpstageEmbeddings는 사용자 질문을(texts)을 embed_query로 처리합니다.
# print("query embedding:", vec)
# print("embedding dim:", len(vec))
