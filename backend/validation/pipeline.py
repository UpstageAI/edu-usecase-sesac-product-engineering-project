# backend/validation/pipeline.py

from typing import List, Dict, Any, Optional
from .base import BaseValidator, ValidationResult, ValidationSeverity


class ValidationPipeline:
    """
    Pipeline for running multiple validators in sequence.

    Features:
    - Sequential validation execution
    - Early termination on critical errors (optional)
    - Aggregate results from all validators
    - Conditional validation (skip validators based on previous results)
    """

    def __init__(
        self,
        validators: List[BaseValidator],
        stop_on_critical: bool = False,
        name: str = "ValidationPipeline"
    ):
        """
        Initialize validation pipeline.

        Args:
            validators: List of validators to run
            stop_on_critical: Stop execution if critical error found
            name: Pipeline name for logging
        """
        self.validators = validators
        self.stop_on_critical = stop_on_critical
        self.name = name

    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        Run all validators in the pipeline.

        Args:
            data: Data to validate
            context: Optional context shared across validators

        Returns:
            List of ValidationResult from each validator
        """
        results = []

        for validator in self.validators:
            # Run validator
            result = await validator.validate(data, context)
            results.append(result)

            # Early termination if critical error found
            if self.stop_on_critical and result.has_errors:
                critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
                if critical_issues:
                    print(f"⚠️ {self.name}: Critical error found, stopping pipeline")
                    break

        return results

    def get_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """
        Get aggregated summary of all validation results.

        Args:
            results: List of validation results

        Returns:
            Summary dictionary with statistics
        """
        total_validators = len(results)
        passed = sum(1 for r in results if r.is_valid)
        failed = total_validators - passed

        total_issues = sum(len(r.issues) for r in results)
        critical = sum(len(r.get_issues_by_severity(ValidationSeverity.CRITICAL)) for r in results)
        errors = sum(len(r.get_issues_by_severity(ValidationSeverity.ERROR)) for r in results)
        warnings = sum(len(r.get_issues_by_severity(ValidationSeverity.WARNING)) for r in results)

        return {
            "pipeline_name": self.name,
            "overall_valid": failed == 0,
            "validators": {
                "total": total_validators,
                "passed": passed,
                "failed": failed
            },
            "issues": {
                "total": total_issues,
                "critical": critical,
                "errors": errors,
                "warnings": warnings
            },
            "results": [
                {
                    "validator": r.validator_name,
                    "valid": r.is_valid,
                    "summary": r.summary()
                }
                for r in results
            ]
        }

    def print_summary(self, results: List[ValidationResult]):
        """
        Print human-readable summary of validation results.

        Args:
            results: List of validation results
        """
        summary = self.get_summary(results)

        print(f"\n{'='*60}")
        print(f"🔍 {summary['pipeline_name']} - 검증 결과")
        print(f"{'='*60}")

        print(f"\n📊 통계:")
        print(f"  전체 검증기: {summary['validators']['total']}")
        print(f"  ✅ 통과: {summary['validators']['passed']}")
        print(f"  ❌ 실패: {summary['validators']['failed']}")

        if summary['issues']['total'] > 0:
            print(f"\n⚠️ 발견된 문제:")
            print(f"  🚨 치명적: {summary['issues']['critical']}")
            print(f"  ❌ 오류: {summary['issues']['errors']}")
            print(f"  ⚠️  경고: {summary['issues']['warnings']}")

        print(f"\n📋 상세 결과:")
        for result_info in summary['results']:
            status = "✅" if result_info['valid'] else "❌"
            print(f"  {status} {result_info['summary']}")

        if summary['overall_valid']:
            print(f"\n{'='*60}")
            print("✅ 전체 검증 통과")
            print(f"{'='*60}\n")
        else:
            print(f"\n{'='*60}")
            print("❌ 검증 실패 - 문제를 수정해주세요")
            print(f"{'='*60}\n")


class ConditionalPipeline(ValidationPipeline):
    """
    Validation pipeline with conditional execution.

    Validators can be skipped based on previous results.
    """

    def __init__(
        self,
        validators: List[BaseValidator],
        conditions: Optional[Dict[str, callable]] = None,
        stop_on_critical: bool = False,
        name: str = "ConditionalPipeline"
    ):
        """
        Initialize conditional pipeline.

        Args:
            validators: List of validators
            conditions: Dict mapping validator name to condition function
                       Condition function receives previous results and returns bool
            stop_on_critical: Stop on critical errors
            name: Pipeline name
        """
        super().__init__(validators, stop_on_critical, name)
        self.conditions = conditions or {}

    async def validate(
        self,
        data: Any,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ValidationResult]:
        """
        Run validators with conditional execution.

        Args:
            data: Data to validate
            context: Optional context

        Returns:
            List of ValidationResult (may be shorter if validators were skipped)
        """
        results = []

        for validator in self.validators:
            # Check if this validator should run
            validator_name = validator.name
            if validator_name in self.conditions:
                should_run = self.conditions[validator_name](results, context)
                if not should_run:
                    print(f"⏭️  Skipping {validator_name} (condition not met)")
                    continue

            # Run validator
            result = await validator.validate(data, context)
            results.append(result)

            # Early termination
            if self.stop_on_critical and result.has_errors:
                critical_issues = result.get_issues_by_severity(ValidationSeverity.CRITICAL)
                if critical_issues:
                    break

        return results
