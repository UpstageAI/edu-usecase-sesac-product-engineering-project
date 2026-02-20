# backend/validation/__init__.py

"""
통합 검증 프레임워크

시스템 전체의 검증 로직을 표준화하고 재사용 가능한 검증 컴포넌트를 제공합니다.
"""

from .base import BaseValidator, ValidationResult, ValidationSeverity
from .pipeline import ValidationPipeline
from .validators import (
    ContentValidator,
    StructureValidator,
    QualityValidator,
    BusinessRuleValidator
)

__all__ = [
    "BaseValidator",
    "ValidationResult",
    "ValidationSeverity",
    "ValidationPipeline",
    "ContentValidator",
    "StructureValidator",
    "QualityValidator",
    "BusinessRuleValidator",
]
