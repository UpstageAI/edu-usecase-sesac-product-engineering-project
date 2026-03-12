from backend.agents.quiz_agent import nodes as quiz_nodes


def test_extract_requested_question_count_defaults_to_five():
    count = quiz_nodes._extract_requested_question_count("무역실무 퀴즈내줘", {})
    assert count == 5


def test_extract_requested_question_count_from_numeric_request():
    count = quiz_nodes._extract_requested_question_count("퀴즈 7문제 내줘", {})
    assert count == 7


def test_extract_requested_question_count_for_additional_request():
    count = quiz_nodes._extract_requested_question_count("퀴즈 더 내줘", {})
    assert count == 5


def test_prepare_llm_messages_node_sets_default_question_count_to_five():
    state = {
        "user_input": "무역실무 퀴즈내줘",
        "context": {},
        "rag_context_str": "",
        "retrieved_documents": [],
    }

    updated = quiz_nodes.prepare_llm_messages_node(state)

    assert updated["quiz_question_count"] == 5
    assert "5문제를 출제하세요" in updated["llm_messages"][0]["content"]
    assert "4지선다 5문제" in updated["llm_messages"][1]["content"]


def test_rebalance_answer_positions_avoids_single_index_bias():
    questions = [
        {
            "question": f"문제 {i}",
            "choices": ["A", "B", "C", "D"],
            "answer": 0,
            "explanation": "설명",
        }
        for i in range(8)
    ]

    balanced = quiz_nodes._rebalance_answer_positions(questions)
    answers = [int(item["answer"]) for item in balanced]

    assert len(set(answers)) > 1
    assert all(0 <= idx <= 3 for idx in answers)
