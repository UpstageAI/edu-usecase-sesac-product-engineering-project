import json
from pathlib import Path

def pick_meta(rid):
    if rid in ["M001", "M002", "M003"]:
        return {"topic": "documentation", "situation": "bl_review", "doc_type": "BL"}
    if rid in ["M004", "M005", "M006"]:
        return {"topic": "customs", "situation": "customs_preparation", "doc_type": "Invoice"}
    if rid in ["M007", "M008", "M009"]:
        return {"topic": "logistics", "situation": "shipment_preparation", "doc_type": "Packing List"}
    if rid in ["M010", "M011"]:
        return {"topic": "payment", "situation": "remittance_execution", "doc_type": "Bank Data"}
    if rid in ["M012", "M013"]:
        return {"topic": "contract", "situation": "contract_review", "doc_type": "Contract"}
    if rid in ["M014", "M015"]:
        return {"topic": "email", "situation": "email_communication", "doc_type": "Email"}
    if rid in ["M016", "M017"]:
        return {"topic": "risk_management", "situation": "issue_handling", "doc_type": "Internal Data"}
    return {"topic": "general", "situation": "work_review", "doc_type": "Document"}

def get_scenario_variants(item):
    rid = item["id"]
    risk_type = item["risk_type"]
    
    # Pre-defined high quality scenarios from raw data
    variants = {
        "M001": [
            "BL 수하인 이름 철자가 계약서랑 한 글자 달라요. 그냥 진행해도 되나요?",
            "Consignee name 오타 같아요. 수정 안 하면 어떤 문제가 생기죠?",
            "서류상 수하인 표기가 애매해서 현지에서 거절될까 걱정돼요."
        ],
        "M005": [
            "인보이스 초안인데 품목별 합계랑 총액이 안 맞아요. 그냥 보내도 되나요?",
            "Invoice line total 합계가 final amount랑 달라요. 어떤 실수가 생길 수 있죠?",
            "견적서 기반으로 인보이스 만들었는데 숫자가 애매하게 10~20달러 정도 차이나요."
        ],
        "M010": [
            "송금 정보 확인 중인데 은행은 UK인데 SWIFT 코드가 KR로 시작해요. 괜찮나요?",
            "Bank name이 HSBC London인데 SWIFT가 KODBKRSE로 나왔어요. 이대로 보내도 되나요?",
            "벤더가 준 계좌정보에 국가가 섞여 보이는데, 송금 진행 전에 뭐부터 확인해야 하나요?"
        ],
        "M019": [
            "CIF 조건인데 보험 서류가 아직 없어요. 선적 먼저 진행해도 되나요?",
            "CIF인데 insurance certificate 누락 상태예요. 지금 단계에서 리스크가 뭐죠?",
            "운송보험 가입을 깜빡했는데 출항 전까지 꼭 해야 하나요?"
        ]
    }
    
    if rid in variants:
        return variants[rid]
    
    # Generic templates for others
    return [
        f"{risk_type} 관련해서 지금 상황이 불안합니다. {item.get('situation_context', '')} 중에 이런 일이 생기면 어쩌죠?",
        f"{risk_type} 케이스에서 발생 가능한 결과와 예방책을 알고 싶습니다.",
        f"실무에서 {risk_type} 실수는 보통 왜 발생하고 어떻게 수습하나요?"
    ]

def build_scenarios_master():
    mistakes_path = Path("dataset/mistakes_master.json")
    output_path = Path("dataset/scenarios_master.json")
    
    if not mistakes_path.exists():
        print("mistakes_master.json not found.")
        return

    mistakes = json.loads(mistakes_path.read_text(encoding="utf-8"))
    scenarios = []
    
    for m in mistakes:
        meta = pick_meta(m["id"])
        variants = get_scenario_variants(m)
        
        for i, text in enumerate(variants, start=1):
            scenarios.append({
                "scenario_id": f"{m['id']}-S{i}",
                "base_case_id": m["id"],
                "content": text, # Added for RAG compatibility
                "input_text": text,
                "expected_answers": {
                    "risk_type": m["risk_type"],
                    "min_severity": m["risk_level"],
                    "must_contain_keywords": m["impact"]["consequences"][:2] # Top 2 consequences as keywords
                },
                "context_metadata": {
                    "topic": meta["topic"],
                    "situation": meta["situation"],
                    "doc_type": meta["doc_type"]
                }
            })

    output_path.write_text(json.dumps(scenarios, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Successfully created: {output_path}")
    print(f"Total scenarios: {len(scenarios)}")

if __name__ == "__main__":
    build_scenarios_master()
