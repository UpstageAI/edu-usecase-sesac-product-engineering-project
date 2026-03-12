import json
import csv
import io
import os
from pathlib import Path

def build_mistakes_master():
    # Paths
    raw_dir = Path("dataset/raw/raw_situation")
    output_path = Path("dataset/mistakes_master.json")
    
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist.")
        return

    # 1. Load message.txt (Master data source)
    message_path = raw_dir / "message.txt"
    if not message_path.exists():
        print(f"Error: {message_path} does not exist.")
        return
        
    try:
        message_data = json.loads(message_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error parsing message.txt: {e}")
        return

    # 2. Load 가상사례 11 (1).txt (Feedback and Expected Loss source)
    case_path = raw_dir / "가상사례 11 (1).txt"
    feedback_dict = {}
    if case_path.exists():
        case_text = case_path.read_text(encoding="utf-8")
        # Extract CSV part starting from header
        header = "id,risk_type,supervisor_feedback,expected_loss"
        # Search for header ignoring trailing whitespace
        import re
        match = re.search(r"id,risk_type,supervisor_feedback,expected_loss\s*", case_text)
        if match:
            csv_part = case_text[match.end():].split("\n\n")[0].strip() # Stop at next double newline or end
            # Use csv reader to handle quotes correctly
            f = io.StringIO(header + "\n" + csv_part)
            reader = csv.DictReader(f)
            for row in reader:
                mid = (row.get('id') or '').strip()
                if not mid: continue
                feedback_dict[mid] = {
                    "risk_type": (row.get('risk_type') or '').strip(),
                    "supervisor_feedback": (row.get('supervisor_feedback') or '').strip().strip('"""'),
                    "expected_loss_dynamic": (row.get('expected_loss') or '').strip()
                }

    # 3. Merge and Transform to Master Mistake Schema
    master_data = []
    for entry in message_data:
        mid = entry.get("id")
        feedback = feedback_dict.get(mid, {})
        
        # Searchable content for RAG
        content_text = f"ID: {mid}\n유형: {feedback.get('risk_type') or entry.get('mistake')}\n상태: {entry.get('situation')}\n설명: {entry.get('description')}\n사수 조언: {feedback.get('supervisor_feedback') or entry.get('mistake')}\n예상 손실: {feedback.get('expected_loss_dynamic') or entry.get('estimated_loss')}"

        # Mapping according to '스키마 예시.txt'
        master_entry = {
            "id": mid,
            "category": entry.get("category"),
            "risk_type": feedback.get("risk_type") or entry.get("mistake"),
            "content": content_text, # Added for RAG ingestion
            "situation_context": entry.get("situation"),
            "mistake_details": entry.get("description"),
            "risk_level": entry.get("risk_level"),
            "impact": {
                "expected_loss": feedback.get("expected_loss_dynamic") or entry.get("estimated_loss"),
                "consequences": entry.get("consequences", [])
            },
            "coaching": {
                "supervisor_feedback": feedback.get("supervisor_feedback") or f"{entry.get('mistake')}는 중요한 리스크입니다. 주의가 필요합니다.",
                "prevention_checklist": entry.get("prevention", []),
                "recovery_action": entry.get("real_case") or "상사에게 즉시 보고하고 매뉴얼에 따라 대응하십시오."
            },
            "metadata": {
                "topic": entry.get("category"),
                "doc_type": "risk_knowledge"
            }
        }
        master_data.append(master_entry)

    # 4. Save to dataset/mistakes_master.json
    try:
        output_path.write_text(json.dumps(master_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Successfully created: {output_path}")
        print(f"Total entries: {len(master_data)}")
    except Exception as e:
        print(f"Error saving to {output_path}: {e}")

if __name__ == "__main__":
    build_mistakes_master()
