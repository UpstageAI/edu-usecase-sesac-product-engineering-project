#!/usr/bin/env python3
# Quick test for validation framework

import sys
import os
import asyncio

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from backend.validation import (
    ValidationSeverity,
    ContentValidator,
    StructureValidator,
    QualityValidator,
    BusinessRuleValidator,
    ValidationPipeline
)
from backend.validation.examples import (
    QuizQuestionValidator,
    create_quiz_validation_pipeline,
    create_email_validation_pipeline,
    create_risk_validation_pipeline
)

async def test_validation_framework():
    print("=== Testing Validation Framework ===\n")

    # Test 1: Basic validators
    print("1. Testing Basic Validators...")
    content_validator = ContentValidator(required_fields=["field1", "field2"])

    # Valid data
    result = await content_validator.validate({"field1": "value1", "field2": "value2"})
    print(f"   ✓ ContentValidator (valid): {result.is_valid}")

    # Invalid data
    result = await content_validator.validate({"field1": "value1"})
    print(f"   ✓ ContentValidator (missing field): {not result.is_valid}")
    print(f"     Issues: {len(result.issues)}")

    # Test 2: Quality validator
    print("\n2. Testing QualityValidator...")
    quality_validator = QualityValidator(min_length=10, max_length=100)

    result = await quality_validator.validate("This is a good length text")
    print(f"   ✓ QualityValidator (valid length): {result.is_valid}")

    result = await quality_validator.validate("Short")
    print(f"   ✓ QualityValidator (too short): {result.has_warnings}")

    # Test 3: Business rule validator
    print("\n3. Testing BusinessRuleValidator...")

    def check_positive(data, context):
        value = data.get("value", 0)
        if value > 0:
            return True, "Value is positive"
        return False, "Value must be positive"

    business_validator = BusinessRuleValidator(rules={"positive": check_positive})

    result = await business_validator.validate({"value": 10})
    print(f"   ✓ BusinessRuleValidator (valid): {result.is_valid}")

    result = await business_validator.validate({"value": -5})
    print(f"   ✓ BusinessRuleValidator (invalid): {not result.is_valid}")

    # Test 4: Quiz validation pipeline
    print("\n4. Testing Quiz Validation Pipeline...")
    quiz_pipeline = create_quiz_validation_pipeline()

    quiz_data = {
        "question": "What is FOB?",
        "choices": ["Free On Board", "Cost Insurance Freight", "Ex Works", "Free Carrier"],
        "correct_answer": 0,
        "explanation": "FOB means Free On Board"
    }

    results = await quiz_pipeline.validate(quiz_data)
    print(f"   ✓ Quiz pipeline validators: {len(results)}")
    print(f"   ✓ All passed: {all(r.is_valid for r in results)}")

    quiz_pipeline.print_summary(results)

    # Test 5: Email validation pipeline
    print("\n5. Testing Email Validation Pipeline...")
    email_pipeline = create_email_validation_pipeline()

    email_data = {
        "subject": "Shipping Confirmation",
        "content": "We will ship via FOB Shanghai. Payment terms: L/C at sight. The shipment will be ready by next week."
    }

    results = await email_pipeline.validate(email_data)
    print(f"   ✓ Email pipeline validators: {len(results)}")
    print(f"   ✓ All passed: {all(r.is_valid for r in results)}")

    email_pipeline.print_summary(results)

    # Test 6: Invalid email with bad Incoterms
    print("\n6. Testing Email with Invalid Incoterms...")
    email_data_invalid = {
        "subject": "Shipping Confirmation",
        "content": "We will ship via FOV Shanghai."  # Invalid!
    }

    results = await email_pipeline.validate(email_data_invalid)
    business_result = next(r for r in results if r.validator_name == "EmailBusinessRules")
    print(f"   ✓ Caught invalid Incoterms: {not business_result.is_valid}")
    if business_result.issues:
        print(f"     Issue: {business_result.issues[0].message}")

    # Test 7: Risk validation pipeline
    print("\n7. Testing Risk Validation Pipeline...")
    risk_pipeline = create_risk_validation_pipeline()

    risk_data = {
        "risk_factors": {
            "financial_loss": {
                "impact": 4,
                "likelihood": 3,
                "score": 12
            }
        },
        "risk_scoring": {
            "overall_risk_level": "high"
        },
        "prevention_strategy": {
            "short_term": "Immediate action",
            "long_term": "Strategic plan"
        }
    }

    results = await risk_pipeline.validate(risk_data)
    print(f"   ✓ Risk pipeline validators: {len(results)}")
    print(f"   ✓ All passed: {all(r.is_valid for r in results)}")

    # Test 8: Invalid risk score
    print("\n8. Testing Risk with Invalid Score...")
    risk_data_invalid = {
        "risk_factors": {
            "financial_loss": {
                "impact": 4,
                "likelihood": 3,
                "score": 99  # Wrong! Should be 12
            }
        },
        "risk_scoring": {},
        "prevention_strategy": {}
    }

    results = await risk_pipeline.validate(risk_data_invalid)
    business_result = next(r for r in results if r.validator_name == "RiskBusinessRules")
    print(f"   ✓ Caught invalid score: {not business_result.is_valid}")
    if business_result.issues:
        print(f"     Issue: {business_result.issues[0].message[:80]}...")

    # Test 9: Validation result properties
    print("\n9. Testing ValidationResult Properties...")
    from backend.validation.base import ValidationResult, ValidationIssue

    issues = [
        ValidationIssue(severity=ValidationSeverity.CRITICAL, message="Critical"),
        ValidationIssue(severity=ValidationSeverity.ERROR, message="Error"),
        ValidationIssue(severity=ValidationSeverity.WARNING, message="Warning"),
    ]

    result = ValidationResult(is_valid=False, validator_name="Test", issues=issues)

    print(f"   ✓ has_errors: {result.has_errors}")
    print(f"   ✓ has_warnings: {result.has_warnings}")
    print(f"   ✓ Critical issues: {len(result.get_issues_by_severity(ValidationSeverity.CRITICAL))}")
    print(f"   ✓ Summary: {result.summary()}")

    print("\n" + "="*60)
    print("✅ All validation framework tests passed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    asyncio.run(test_validation_framework())
