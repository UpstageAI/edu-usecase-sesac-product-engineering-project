from langgraph.graph import StateGraph, START, END
import operator
from typing import TypedDict, Annotated, List

# Reducer를 사용한 State 정의 (값 누적)
class AccumulatorState(TypedDict):
    messages: Annotated[List[str], operator.add]  # 리스트에 값 추가

import nest_asyncio
nest_asyncio.apply() # mermaid 랜더링을 위해 중첩 이벤트 루프 허용

from IPython.display import Image, display
from langchain_core.runnables.graph import MermaidDrawMethod
from term_image.image import AutoImage

# 1. 그래프 빌더 생성
builder = StateGraph(AccumulatorState)

# 노드 함수 정의
def query_vectorization(state: AccumulatorState) -> AccumulatorState:
    """노드 A: 상태에 'A' 추가"""
    print(f"  [Node A] 현재 상태: {state['messages']}")
    return {"messages": ["A 처리완료"]}

def retriever(state: AccumulatorState) -> AccumulatorState:
    """노드 B: 상태에 'B' 추가"""
    print(f"  [Node B] 현재 상태: {state['messages']}")
    return {"messages": ["B 처리완료"]}

def generator(state: AccumulatorState) -> AccumulatorState:
    """노드 C: 상태에 'C' 추가"""
    print(f"  [Node C] 현재 상태: {state['messages']}")
    return {"messages": ["C 처리완료"]}

# 2. 노드 추가
builder.add_node("query_vectorization", query_vectorization)
builder.add_node("retriever", retriever)
builder.add_node("generator", generator)

# 3. 엣지 추가 (실행 순서 정의)
builder.add_edge(START, "query_vectorization")  # 시작 → A
builder.add_edge("query_vectorization", "retriever")     # A → B
builder.add_edge("retriever", "generator")     # B → C
builder.add_edge("generator", END)     # C → 종료

# 4. 그래프 컴파일
graph = builder.compile()

# 5. 그래프 시각화
img_bytes = graph.get_graph().draw_mermaid_png()

with open("graph.png", "wb") as f:
    f.write(img_bytes)
# print("\n✅ 순차 실행 그래프 (START → A → B → END)"