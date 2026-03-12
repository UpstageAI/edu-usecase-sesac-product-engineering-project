"""
Logging Configuration for Trade Onboarding Agent

Features:
- 날짜별 자동 로테이션 (매일 자정에 새 파일 생성)
- 파일 크기 제한 (10MB, 최대 5개 백업 파일)
- 레벨별 파일 분리:
  - logs/app.log: 모든 로그 (INFO 이상)
  - logs/error.log: 에러만 (ERROR 이상)
  - logs/debug.log: 디버그 포함 (DEBUG 이상, 개발 환경만)
- 콘솔 + 파일 동시 출력
- 상세한 포맷 (타임스탬프, 레벨, 파일명, 함수명, 라인번호, 메시지)

Usage:
    from backend.utils.logger import setup_logging

    # 앱 시작 시 한 번만 호출
    setup_logging(environment="development")

    # 이후 일반적인 logging 사용
    import logging
    logger = logging.getLogger(__name__)
    logger.info("Hello World")
"""
import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    environment: str = "development",
    log_dir: str = "logs",
    app_name: str = "trade_onboarding"
) -> None:
    """
    로깅 시스템 초기화

    Args:
        environment: 환경 ("development" | "production" | "test")
        log_dir: 로그 파일 저장 디렉토리
        app_name: 애플리케이션 이름 (로그 파일명에 사용)

    Example:
        setup_logging(environment="development")
        logger = logging.getLogger(__name__)
        logger.info("Application started")
    """
    # 로그 디렉토리 생성
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # 환경별 로그 레벨 설정
    log_levels = {
        "development": logging.DEBUG,
        "test": logging.WARNING,
        "production": logging.INFO
    }
    log_level = log_levels.get(environment, logging.INFO)

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()

    # 포맷터 설정
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 콘솔 핸들러 (모든 로그 출력)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    root_logger.addHandler(console_handler)

    # 2. 전체 로그 파일 핸들러 (TimedRotatingFileHandler - 매일 자정 로테이션)
    app_log_file = log_path / f"{app_name}.log"
    app_file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=app_log_file,
        when="midnight",       # 매일 자정에 로테이션
        interval=1,            # 1일마다
        backupCount=30,        # 최대 30일치 보관
        encoding="utf-8"
    )
    app_file_handler.setLevel(logging.INFO)
    app_file_handler.setFormatter(detailed_formatter)
    app_file_handler.suffix = "%Y-%m-%d"  # 백업 파일명: app.log.2026-02-12
    root_logger.addHandler(app_file_handler)

    # 3. 에러 전용 파일 핸들러 (크기 기반 로테이션)
    error_log_file = log_path / f"{app_name}_error.log"
    error_file_handler = logging.handlers.RotatingFileHandler(
        filename=error_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,               # 최대 5개 백업 파일
        encoding="utf-8"
    )
    error_file_handler.setLevel(logging.ERROR)
    error_file_handler.setFormatter(detailed_formatter)
    root_logger.addHandler(error_file_handler)

    # 4. 디버그 로그 파일 (개발 환경만)
    if environment == "development":
        debug_log_file = log_path / f"{app_name}_debug.log"
        debug_file_handler = logging.handlers.RotatingFileHandler(
            filename=debug_log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=3,               # 최대 3개 백업 파일
            encoding="utf-8"
        )
        debug_file_handler.setLevel(logging.DEBUG)
        debug_file_handler.setFormatter(detailed_formatter)
        root_logger.addHandler(debug_file_handler)

    # 5. 외부 라이브러리 로그 레벨 조정 (노이즈 감소)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("openai._base_client").setLevel(logging.WARNING)
    logging.getLogger("langsmith").setLevel(logging.WARNING)
    logging.getLogger("posthog").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # 초기화 완료 로그
    logger = logging.getLogger(__name__)
    logger.info(f"🔧 Logging initialized: environment={environment}, level={logging.getLevelName(log_level)}")
    logger.info(f"📁 Log directory: {log_path.absolute()}")
    logger.info(f"📝 Log files: {app_name}.log, {app_name}_error.log" +
                (f", {app_name}_debug.log" if environment == "development" else ""))


def get_logger(name: str) -> logging.Logger:
    """
    로거 인스턴스 가져오기

    Args:
        name: 로거 이름 (보통 __name__ 사용)

    Returns:
        logging.Logger: 로거 인스턴스

    Example:
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    return logging.getLogger(name)
