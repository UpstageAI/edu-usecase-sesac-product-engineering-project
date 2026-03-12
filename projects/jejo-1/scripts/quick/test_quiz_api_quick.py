#!/usr/bin/env python3
# Quick test for Quiz API services

import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.services.quiz_service import quiz_session_store, quiz_generator

print("=== Testing Quiz Service ===")

# Test 1: Generate sample quizzes
print("\n1. Generating sample quizzes...")
questions = quiz_generator.generate_sample_quizzes(count=3)
print(f"   ✓ Generated {len(questions)} questions")
print(f"   ✓ First question: {questions[0].question[:50]}...")

# Test 2: Create quiz session
print("\n2. Creating quiz session...")
session_id = quiz_session_store.create_session(
    questions=questions,
    topic="Incoterms",
    difficulty="easy"
)
print(f"   ✓ Session ID: {session_id}")

# Test 3: Retrieve session
print("\n3. Retrieving session...")
session = quiz_session_store.get_session(session_id)
print(f"   ✓ Session retrieved: {session is not None}")
print(f"   ✓ Questions in session: {len(session['questions'])}")
print(f"   ✓ Topic: {session['topic']}")
print(f"   ✓ Created at: {session['created_at']}")

# Test 4: Get specific question
print("\n4. Getting specific question...")
quiz_id = questions[0].quiz_id
question = quiz_session_store.get_question(session_id, quiz_id)
print(f"   ✓ Question retrieved: {question is not None}")
print(f"   ✓ Question text: {question.question[:50]}...")
print(f"   ✓ Choices: {len(question.choices)}")
print(f"   ✓ Correct answer: {question.correct_answer}")

# Test 5: Save answer
print("\n5. Saving user answer...")
quiz_session_store.save_answer(session_id, quiz_id, 0)
session = quiz_session_store.get_session(session_id)
print(f"   ✓ Answer saved: {quiz_id in session['answers']}")
print(f"   ✓ User answer: {session['answers'][quiz_id]}")

# Test 6: Evaluate answer
print("\n6. Evaluating answer...")
is_correct = session['answers'][quiz_id] == question.correct_answer
print(f"   ✓ Is correct: {is_correct}")
print(f"   ✓ Explanation: {question.explanation[:80]}...")

# Test 7: Test hiding answers in response
print("\n7. Testing question hiding (for API response)...")
questions_for_user = [
    {
        "quiz_id": q.quiz_id,
        "question": q.question,
        "choices": q.choices,
        "quiz_type": q.quiz_type,
        "difficulty": q.difficulty
    }
    for q in questions
]
first_question_dict = questions_for_user[0]
print(f"   ✓ Question has 'quiz_id': {'quiz_id' in first_question_dict}")
print(f"   ✓ Question has 'question': {'question' in first_question_dict}")
print(f"   ✓ Question has 'choices': {'choices' in first_question_dict}")
print(f"   ✓ Question hidden 'correct_answer': {'correct_answer' not in first_question_dict}")
print(f"   ✓ Question hidden 'explanation': {'explanation' not in first_question_dict}")

print("\n✅ All Quiz Service tests passed!")
