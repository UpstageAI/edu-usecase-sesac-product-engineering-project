import json
from pathlib import Path

def build_users_master():
    raw_dir = Path("dataset/raw/raw_situation")
    output_path = Path("dataset/users_master.json")
    
    # Preferred source: (0213) 서비스 검증.txt or (0212)데이터 유저셋 (json) (1).txt
    # They both have the same list.
    user_file = raw_dir / "(0212)데이터 유저셋 (json) (1).txt"
    if not user_file.exists():
        user_file = raw_dir / "0213 (서비스 검증).txt"
    
    if not user_file.exists():
        print("User raw file not found.")
        return

    text = user_file.read_text(encoding="utf-8")
    # Extract the JSON array part
    import re
    match = re.search(r"\[\s*\{.*\}\s*\]", text, re.DOTALL)
    if match:
        try:
            users_data = json.loads(match.group())
            # Add 'content' field for RAG consistency
            for u in users_data:
                u["content"] = f"User {u['user_id']} ({u['role_level']}): {u['experience_months']}mo exp, Weak in {', '.join(u['weak_topics'])}, Prefers {u['preferred_style']} style."
            
            output_path.write_text(json.dumps(users_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"Successfully created: {output_path}")
        except Exception as e:
            print(f"Error parsing user JSON: {e}")

if __name__ == "__main__":
    build_users_master()
