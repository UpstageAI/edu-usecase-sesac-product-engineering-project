# backend/validation/examples.py

"""
Examples of using the validation framework with different agents.
"""

from typing import Dict, Any, List, Optional
from .base import ValidationSeverity
from .validators import (
    ContentValidator,
    StructureValidator,
    QualityValidator,
    BusinessRuleValidator
)
from .pipeline import ValidationPipeline


# ============================================================================
# Quiz Validation Example
# ============================================================================

class QuizQuestionValidator(ContentValidator):
    """Validator for quiz questions"""

    def __init__(self):
        super().__init__(
            required_fields=["question", "choices", "correct_answer", "explanation"],
            name="QuizQuestionValidator"
        )

    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> "ValidationResult":
        """Validate quiz question structure and content"""
        # First, run base content validation
        result = await super().validate(data, context)

        # Additional quiz-specific checks
        if isinstance(data, dict):
            issues = list(result.issues)

            # Validate choices count
            if "choices" in data:
                choices = data["choices"]
                if not isinstance(choices, list):
                    issues.append(
                        self._create_issue(
                            ValidationSeverity.ERROR,
                            "Choices must be a list",
                            field="choices"
                        )
                    )
                elif len(choices) != 4:
                    issues.append(
                        self._create_issue(
                            ValidationSeverity.WARNING,
                            f"Expected 4 choices, found {len(choices)}",
                            field="choices",
                            expected="4",
                            actual=str(len(choices))
                        )
                    )

            # Validate correct_answer index
            if "correct_answer" in data and "choices" in data:
                correct_idx = data["correct_answer"]
                choices_len = len(data.get("choices", []))

                if not isinstance(correct_idx, int):
                    issues.append(
                        self._create_issue(
                            ValidationSeverity.ERROR,
                            "correct_answer must be an integer",
                            field="correct_answer"
                        )
                    )
                elif correct_idx < 0 or correct_idx >= choices_len:
                    issues.append(
                        self._create_issue(
                            ValidationSeverity.CRITICAL,
                            f"correct_answer index {correct_idx} out of range (0-{choices_len-1})",
                            field="correct_answer",
                            actual=str(correct_idx),
                            expected=f"0-{choices_len-1}"
                        )
                    )

            # Update result with new issues
            result.issues = issues
            result.is_valid = not result.has_errors

        return result


def create_quiz_validation_pipeline() -> ValidationPipeline:
    """
    Create validation pipeline for quiz questions.

    Returns:
        ValidationPipeline configured for quiz validation
    """
    return ValidationPipeline(
        validators=[
            QuizQuestionValidator(),
            QualityValidator(min_length=10, name="QuestionQuality"),
            StructureValidator(
                expected_schema={
                    "type": "dict",
                    "fields": {
                        "question": {"type": "str", "required": True},
                        "choices": {"type": "list", "required": True},
                        "correct_answer": {"type": "int", "required": True},
                        "explanation": {"type": "str", "required": True}
                    }
                },
                name="QuizStructure"
            )
        ],
        stop_on_critical=True,
        name="QuizValidationPipeline"
    )


# ============================================================================
# Email Validation Example
# ============================================================================

def validate_incoterms(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    """
    Business rule: Validate Incoterms usage.

    Args:
        data: Email content dict
        context: Optional context

    Returns:
        (is_valid, message) tuple
    """
    valid_incoterms = ["EXW", "FCA", "CPT", "CIP", "DAP", "DPU", "DDP", "FAS", "FOB", "CFR", "CIF"]

    content = data.get("content", "")
    found_terms = []

    for term in valid_incoterms:
        if term in content:
            found_terms.append(term)

    # Check for invalid terms (common typos)
    invalid_terms = ["FOV", "CIP2", "FOBS"]  # Common mistakes
    found_invalid = []

    for term in invalid_terms:
        if term in content:
            found_invalid.append(term)

    if found_invalid:
        return False, f"Invalid Incoterms found: {', '.join(found_invalid)}"

    return True, "Incoterms validation passed"


def create_email_validation_pipeline() -> ValidationPipeline:
    """
    Create validation pipeline for email content.

    Returns:
        ValidationPipeline configured for email validation
    """
    return ValidationPipeline(
        validators=[
            ContentValidator(
                required_fields=["content", "subject"],
                name="EmailContent"
            ),
            QualityValidator(min_length=50, max_length=5000, name="EmailQuality"),
            BusinessRuleValidator(
                rules={"incoterms": validate_incoterms},
                name="EmailBusinessRules"
            )
        ],
        name="EmailValidationPipeline"
    )


# ============================================================================
# Risk Analysis Validation Example
# ============================================================================

def validate_risk_score(data: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> tuple[bool, str]:
    """
    Business rule: Validate risk score calculation.

    Args:
        data: Risk analysis data
        context: Optional context

    Returns:
        (is_valid, message) tuple
    """
    if "risk_factors" not in data:
        return False, "Missing risk_factors"

    for factor_name, factor_data in data["risk_factors"].items():
        impact = factor_data.get("impact", 0)
        likelihood = factor_data.get("likelihood", 0)
        score = factor_data.get("score", 0)

        # Validate score calculation: score = impact * likelihood
        expected_score = impact * likelihood
        if abs(score - expected_score) > 0.01:  # Allow small floating point errors
            return False, f"Invalid score for '{factor_name}': {score} (expected {expected_score})"

        # Validate ranges
        if not (1 <= impact <= 5):
            return False, f"Impact for '{factor_name}' out of range: {impact} (expected 1-5)"

        if not (1 <= likelihood <= 5):
            return False, f"Likelihood for '{factor_name}' out of range: {likelihood} (expected 1-5)"

    return True, "Risk score validation passed"


def create_risk_validation_pipeline() -> ValidationPipeline:
    """
    Create validation pipeline for risk analysis.

    Returns:
        ValidationPipeline configured for risk validation
    """
    return ValidationPipeline(
        validators=[
            ContentValidator(
                required_fields=["risk_factors", "risk_scoring", "prevention_strategy"],
                name="RiskContent"
            ),
            BusinessRuleValidator(
                rules={"risk_score": validate_risk_score},
                name="RiskBusinessRules"
            ),
            StructureValidator(
                expected_schema={
                    "type": "dict",
                    "fields": {
                        "risk_factors": {"type": "dict", "required": True},
                        "risk_scoring": {"type": "dict", "required": True},
                        "prevention_strategy": {"type": "dict", "required": True}
                    }
                },
                name="RiskStructure"
            )
        ],
        stop_on_critical=True,
        name="RiskValidationPipeline"
    )
