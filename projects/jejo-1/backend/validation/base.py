# backend/validation/base.py

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class ValidationSeverity(str, Enum):
    """Validation issue severity levels"""
    INFO = "info"           # 정보성 (개선 권장)
    WARNING = "warning"     # 경고 (주의 필요)
    ERROR = "error"         # 오류 (수정 필요)
    CRITICAL = "critical"   # 치명적 (즉시 수정 필수)


class ValidationIssue(BaseModel):
    """Single validation issue"""
    severity: ValidationSeverity
    message: str
    field: Optional[str] = None          # 문제가 발견된 필드
    expected: Optional[str] = None       # 기대값
    actual: Optional[str] = None         # 실제값
    suggestion: Optional[str] = None     # 수정 제안
    context: Optional[Dict[str, Any]] = None  # 추가 컨텍스트


class ValidationResult(BaseModel):
    """Validation result with standardized structure"""
    is_valid: bool = Field(..., description="Overall validation passed")
    validator_name: str = Field(..., description="Name of the validator")
    issues: List[ValidationIssue] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

    @property
    def has_errors(self) -> bool:
        """Check if there are any errors or critical issues"""
        return any(
            issue.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]
            for issue in self.issues
        )

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return any(issue.severity == ValidationSeverity.WARNING for issue in self.issues)

    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[ValidationIssue]:
        """Get issues filtered by severity"""
        return [issue for issue in self.issues if issue.severity == severity]

    def summary(self) -> str:
        """Get a human-readable summary"""
        if self.is_valid:
            return f"✅ {self.validator_name}: 검증 통과"

        critical = len(self.get_issues_by_severity(ValidationSeverity.CRITICAL))
        errors = len(self.get_issues_by_severity(ValidationSeverity.ERROR))
        warnings = len(self.get_issues_by_severity(ValidationSeverity.WARNING))

        parts = []
        if critical > 0:
            parts.append(f"🚨 치명적: {critical}")
        if errors > 0:
            parts.append(f"❌ 오류: {errors}")
        if warnings > 0:
            parts.append(f"⚠️ 경고: {warnings}")

        return f"{self.validator_name}: {', '.join(parts)}"


class BaseValidator(ABC):
    """
    Abstract base class for all validators.

    All validators must implement the validate() method which returns
    a standardized ValidationResult.
    """

    def __init__(self, name: Optional[str] = None):
        """
        Initialize validator with optional custom name.

        Args:
            name: Custom validator name (defaults to class name)
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    async def validate(self, data: Any, context: Optional[Dict[str, Any]] = None) -> ValidationResult:
        """
        Validate the given data.

        Args:
            data: Data to validate (type depends on validator)
            context: Optional context for validation

        Returns:
            ValidationResult with standardized structure
        """
        pass

    def _create_issue(
        self,
        severity: ValidationSeverity,
        message: str,
        field: Optional[str] = None,
        expected: Optional[str] = None,
        actual: Optional[str] = None,
        suggestion: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationIssue:
        """Helper method to create validation issues"""
        return ValidationIssue(
            severity=severity,
            message=message,
            field=field,
            expected=expected,
            actual=actual,
            suggestion=suggestion,
            context=context
        )

    def _create_result(
        self,
        is_valid: bool,
        issues: List[ValidationIssue],
        metadata: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """Helper method to create validation results"""
        return ValidationResult(
            is_valid=is_valid,
            validator_name=self.name,
            issues=issues,
            metadata=metadata or {}
        )
