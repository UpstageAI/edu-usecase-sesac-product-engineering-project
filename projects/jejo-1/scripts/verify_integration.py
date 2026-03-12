import os
import sys
import json

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from backend.agents.riskmanaging.graph import RiskManagingAgent

def test_risk_knowledge():
    print("--- [Test 1] Risk Knowledge Recognition (M001) ---")
    agent = RiskManagingAgent()
    # Provide ALL info the prompt wants: contract_amount, penalty_info, loss_estimate
    user_input = "M001 리스크(수하인 이름 오타)가 발생했어. 계약금액은 10만 달러고, 페널티 조항은 따로 없으며, 예상 손실은 약 1,000달러 내외로 보여. 사수처럼 조언해줘."
    
    print(f"Query: {user_input}")
    result = agent.run(
        user_input=user_input,
        conversation_history=[],
        analysis_in_progress=False
    )
    
    response_data = result['response']['response']
    print("\n[AI Response Summary]")
    # Printing just a part to avoid long output
    print(str(response_data)[:1000]) 
    
    # Check for keywords from mistakes_master.json M001
    # We want to see if RAG pulled the master data
    keywords = ["철자 오류", "$500", "물건을 못 찾고", "재발행"]
    found = [kw for kw in keywords if kw in str(response_data)]
    print(f"\nFound Keywords: {found}")
    
    if len(found) >= 1:
        print("RESULT: PASS (Integrated Information Found)")
    else:
        print("RESULT: FAIL (Integrated Information Missing)")

def test_user_persona():
    print("\n--- [Test 2] User Persona Adaptation (U01) ---")
    user_profile = {
        "user_id": "U01",
        "role_level": "junior",
        "experience_months": 1,
        "weak_topics": ["payment", "documentation"],
        "risk_tolerance": "low",
        "preferred_style": "checklist"
    }
    
    agent = RiskManagingAgent()
    # Also provide enough info here
    user_input = "품목 분류 HS Code를 잘못 입력했어. 계약금액은 5000만원이고 페널티는 딱히 없어. 예상 손실은 가산세 포함 1000만원 정도야. 리스크 분석해줘."
    
    print(f"User: U01 (Junior, Checklist, Low Risk)")
    print(f"Query: {user_input}")
    
    # We pass user_profile via context
    result = agent.run(
        user_input=user_input,
        conversation_history=[],
        analysis_in_progress=False,
        context={"user_profile": user_profile}
    )
    
    response_data = result['response']['response']
    print("\n[AI Response Summary]")
    print(str(response_data)[:1000]) 
    
    # Check for checklist style (1. or -) or explicit mention of checklist
    is_checklist = "1." in str(response_data) or "- " in str(response_data) or "체크리스트" in str(response_data)
    print(f"Checklist Format: {is_checklist}")
    
    if is_checklist:
        print("RESULT: PASS (Style Adapted)")
    else:
        print("RESULT: FAIL (Style NOT Adapted)")

def test_hypothesis_analysis():
    print("\n--- [Test 3] Hypothesis-based Analysis (Missing Info) ---")
    agent = RiskManagingAgent()
    # Provide MINIMAL info and explicitly say "don't know/just do it"
    user_input = "선적 지연이 발생했는데, 아직 정확한 계약 규모나 페널티 조항은 몰라. 그래도 아는 범위 내에서 리스크 분석해줄 수 있어?"
    
    print(f"Query: {user_input}")
    result = agent.run(
        user_input=user_input,
        conversation_history=[],
        analysis_in_progress=False
    )
    
    response_data = result['response']['response']
    # If it's a full analysis, the response will be a JSON-like string (or object) containing risk_scoring
    is_analytical = "risk_scoring" in str(response_data) or "loss_simulation" in str(response_data)
    
    print("\n[AI Response Summary]")
    print(str(response_data)[:1000]) 
    
    if is_analytical:
        print("\nRESULT: PASS (Hypothesis-based Analysis Performed)")
    else:
        print("\nRESULT: FAIL (Still asking for info or stuck)")

if __name__ == "__main__":
    test_risk_knowledge()
    test_user_persona()
    test_hypothesis_analysis()
