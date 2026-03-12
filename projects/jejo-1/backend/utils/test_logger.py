"""
로깅 시스템 테스트 스크립트

Usage:
    uv run python backend/utils/test_logger.py
"""
import sys
import os

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.utils.logger import setup_logging, get_logger
from backend.config import get_settings


def test_logging():
    """로깅 시스템 테스트"""
    # 로깅 설정
    settings = get_settings()
    setup_logging(environment=settings.environment, app_name="test_trade_onboarding")

    # 로거 생성
    logger = get_logger(__name__)

    print("\n" + "=" * 80)
    print("🧪 로깅 시스템 테스트 시작")
    print("=" * 80 + "\n")

    # 다양한 로그 레벨 테스트
    logger.debug("🔍 [DEBUG] 디버그 메시지 - 변수 값 확인용")
    logger.info("ℹ️  [INFO] 정보 메시지 - 서버 시작, 요청 처리 등")
    logger.warning("⚠️  [WARNING] 경고 메시지 - 비정상 상황이지만 처리 가능")
    logger.error("❌ [ERROR] 에러 메시지 - 처리 실패, 예외 발생")

    # 예외와 함께 로깅
    try:
        result = 10 / 0
    except ZeroDivisionError as e:
        logger.error("❌ [ERROR] 계산 오류 발생", exc_info=True)

    # 컨텍스트 정보와 함께 로깅
    logger.info("📊 [INFO] 사용자 요청 처리 완료",
                extra={"user_id": "test_user", "request_id": "req_12345"})

    print("\n" + "=" * 80)
    print("✅ 로깅 테스트 완료!")
    print("=" * 80)
    print("\n📁 로그 파일 확인:")
    print(f"   - logs/test_trade_onboarding.log (전체 로그)")
    print(f"   - logs/test_trade_onboarding_error.log (에러만)")
    if settings.environment == "development":
        print(f"   - logs/test_trade_onboarding_debug.log (디버그 포함)")
    print()


if __name__ == "__main__":
    test_logging()
