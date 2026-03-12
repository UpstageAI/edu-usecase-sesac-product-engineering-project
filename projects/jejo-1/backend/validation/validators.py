# backend/validation/validators.py

import json
import re
from typing import Dict, Any, List, Optional

from .base import BaseValidator, ValidationResult, ValidationSeverity, ValidationIssue


class ContentValidator(BaseValidator):
    """
    Validates content accuracy against reference data.

    Used for:
    - Quiz question accuracy (against RAG documents)
    - Email trade term accuracy
    - Risk analysis data accuracy
    """

    def __init__(
        self,
        required_fields: Optional[List[str]] = None,
        name: str = "ContentValidator"
    ):
        """
        Initialize content validator.

        Args:
            required_fields: List of required field names
            name: Validator name
        """
        super().__init__(name)
        self.required_fields = required_fields or []

    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate content structure and required fields.

        Args:
            data: Data dictionary to validate
            context: Optional context with reference_data

        Returns:
            ValidationResult
        """
        issues = []

        # Type check
        if not isinstance(data, dict):
            issues.append(
                self._create_issue(
                    ValidationSeverity.CRITICAL,
                    "Data must be a dictionary",
                    actual=str(type(data)),
                    expected="dict"
                )
            )
            return self._create_result(False, issues)

        # Check required fields
        for field in self.required_fields:
            if field not in data:
                issues.append(
                    self._create_issue(
                        ValidationSeverity.ERROR,
                        f"Missing required field: {field}",
                        field=field
                    )
                )
            elif not data[field]:
                issues.append(
                    self._create_issue(
                        ValidationSeverity.WARNING,
                        f"Required field is empty: {field}",
                        field=field
                    )
                )

        # Content-specific validation (if reference data provided)
        if context and "reference_data" in context:
            ref_data = context["reference_data"]
            # Subclasses can override for specific content validation
            content_issues = await self._validate_against_reference(data, ref_data)
            issues.extend(content_issues)

        is_valid = len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0

        return self._create_result(
            is_valid,
            issues,
            metadata={"checked_fields": self.required_fields}
        )

    async def _validate_against_reference(
        self,
        data: Dict[str, Any],
        reference: Any
    ) -> List[ValidationIssue]:
        """
        Validate data against reference (to be overridden by subclasses).

        Args:
            data: Data to validate
            reference: Reference data

        Returns:
            List of validation issues
        """
        return []


class StructureValidator(BaseValidator):
    """
    Validates data structure and format.

    Used for:
    - JSON structure validation
    - API response format validation
    - Quiz question format validation
    """

    def __init__(
        self,
        expected_schema: Optional[Dict[str, Any]] = None,
        name: str = "StructureValidator"
    ):
        """
        Initialize structure validator.

        Args:
            expected_schema: Expected data schema (simplified JSON schema)
            name: Validator name
        """
        super().__init__(name)
        self.expected_schema = expected_schema or {}

    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate data structure.

        Args:
            data: Data to validate
            context: Optional context

        Returns:
            ValidationResult
        """
        issues = []

        # If expected schema is provided, validate against it
        if self.expected_schema:
            schema_issues = self._validate_schema(data, self.expected_schema)
            issues.extend(schema_issues)

        # JSON serialization check
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            issues.append(
                self._create_issue(
                    ValidationSeverity.ERROR,
                    f"Data is not JSON serializable: {str(e)}"
                )
            )

        is_valid = len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0

        return self._create_result(is_valid, issues)

    def _validate_schema(
        self,
        data: Any,
        schema: Dict[str, Any],
        path: str = ""
    ) -> List[ValidationIssue]:
        """
        Validate data against schema recursively.

        Args:
            data: Data to validate
            schema: Schema definition
            path: Current path in data (for error messages)

        Returns:
            List of validation issues
        """
        issues = []

        if "type" in schema:
            expected_type = schema["type"]
            if expected_type == "dict" and not isinstance(data, dict):
                issues.append(
                    self._create_issue(
                        ValidationSeverity.ERROR,
                        f"Expected dict at {path or 'root'}",
                        field=path,
                        expected="dict",
                        actual=str(type(data))
                    )
                )
            elif expected_type == "list" and not isinstance(data, list):
                issues.append(
                    self._create_issue(
                        ValidationSeverity.ERROR,
                        f"Expected list at {path or 'root'}",
                        field=path,
                        expected="list",
                        actual=str(type(data))
                    )
                )

        if "fields" in schema and isinstance(data, dict):
            for field_name, field_schema in schema["fields"].items():
                field_path = f"{path}.{field_name}" if path else field_name

                if field_schema.get("required", False) and field_name not in data:
                    issues.append(
                        self._create_issue(
                            ValidationSeverity.ERROR,
                            f"Missing required field: {field_name}",
                            field=field_path
                        )
                    )

                if field_name in data:
                    # Recursive validation
                    field_issues = self._validate_schema(data[field_name], field_schema, field_path)
                    issues.extend(field_issues)

        return issues


class QualityValidator(BaseValidator):
    """
    Validates content quality (grammar, tone, clarity).

    Used for:
    - Email tone validation
    - Quiz question clarity
    - Explanation quality
    """

    def __init__(
        self,
        min_length: int = 0,
        max_length: Optional[int] = None,
        name: str = "QualityValidator"
    ):
        """
        Initialize quality validator.

        Args:
            min_length: Minimum content length
            max_length: Maximum content length
            name: Validator name
        """
        super().__init__(name)
        self.min_length = min_length
        self.max_length = max_length

    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate content quality.

        Args:
            data: String or dict with text content
            context: Optional context

        Returns:
            ValidationResult
        """
        issues = []

        # Extract text content
        if isinstance(data, str):
            text = data
        elif isinstance(data, dict) and "content" in data:
            text = data["content"]
        else:
            issues.append(
                self._create_issue(
                    ValidationSeverity.ERROR,
                    "Cannot extract text content for quality validation",
                    actual=str(type(data))
                )
            )
            return self._create_result(False, issues)

        # Length validation
        text_length = len(text)

        if text_length < self.min_length:
            issues.append(
                self._create_issue(
                    ValidationSeverity.WARNING,
                    f"Content too short (minimum: {self.min_length} chars)",
                    actual=str(text_length),
                    expected=f"≥{self.min_length}",
                    suggestion=f"Add more detail to reach {self.min_length} characters"
                )
            )

        if self.max_length and text_length > self.max_length:
            issues.append(
                self._create_issue(
                    ValidationSeverity.WARNING,
                    f"Content too long (maximum: {self.max_length} chars)",
                    actual=str(text_length),
                    expected=f"≤{self.max_length}",
                    suggestion=f"Reduce content to {self.max_length} characters or less"
                )
            )

        # Basic quality checks
        if not text.strip():
            issues.append(
                self._create_issue(
                    ValidationSeverity.ERROR,
                    "Content is empty or whitespace only"
                )
            )

        # Check for placeholder text
        placeholders = ["TODO", "FIXME", "XXX", "TBD", "placeholder"]
        for placeholder in placeholders:
            if placeholder.lower() in text.lower():
                issues.append(
                    self._create_issue(
                        ValidationSeverity.WARNING,
                        f"Found placeholder text: {placeholder}",
                        suggestion="Replace placeholder with actual content"
                    )
                )

        is_valid = len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0

        return self._create_result(
            is_valid,
            issues,
            metadata={
                "text_length": text_length,
                "min_length": self.min_length,
                "max_length": self.max_length
            }
        )


class BusinessRuleValidator(BaseValidator):
    """
    Validates business rules and domain-specific logic.

    Used for:
    - Trade term correctness (Incoterms)
    - Payment term validation
    - Risk threshold validation
    """

    def __init__(
        self,
        rules: Optional[Dict[str, callable]] = None,
        name: str = "BusinessRuleValidator"
    ):
        """
        Initialize business rule validator.

        Args:
            rules: Dict mapping rule name to validation function
                  Each function receives data and returns (bool, str)
            name: Validator name
        """
        super().__init__(name)
        self.rules = rules or {}

    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> ValidationResult:
        """
        Validate data against business rules.

        Args:
            data: Data to validate
            context: Optional context

        Returns:
            ValidationResult
        """
        issues = []

        for rule_name, rule_func in self.rules.items():
            try:
                is_valid, message = rule_func(data, context)

                if not is_valid:
                    issues.append(
                        self._create_issue(
                            ValidationSeverity.ERROR,
                            f"Business rule '{rule_name}' failed: {message}",
                            context={"rule": rule_name}
                        )
                    )
            except Exception as e:
                issues.append(
                    self._create_issue(
                        ValidationSeverity.CRITICAL,
                        f"Error executing rule '{rule_name}': {str(e)}",
                        context={"rule": rule_name, "error": str(e)}
                    )
                )

        is_valid = len([i for i in issues if i.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]]) == 0

        return self._create_result(
            is_valid,
            issues,
            metadata={"rules_checked": list(self.rules.keys())}
        )
