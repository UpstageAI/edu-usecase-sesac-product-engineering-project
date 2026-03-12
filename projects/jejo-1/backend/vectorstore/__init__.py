"""
persist_directory="backend/vectorstore"
content: "partial damage","bag tearing 발생","납기 지연으로 클레임 발생"


category
situation
topic
role
priority
source_dataset


claims_001
emails_002


metadata
{
  "category": "claims",
  "situation": ["damage","claim","quality_issue"],
  "topic": ["logistics","trade","customer_issue"],
  "role": ["seller","buyer","forwarder"],
  "priority": "high",
  "level": "working",
  "source_dataset": "claims.json"
}


collection -> trade_coaching_knowledge


retrieval 구조
category = claims
priority = high

basic search
coaching search
agent routing search


ingest.py
1. dataset 폴더 json 전부 읽기
2. 각 entry 분해
3. metadata 정리
4. solar embedding
5. chroma insert
6. persist
"""