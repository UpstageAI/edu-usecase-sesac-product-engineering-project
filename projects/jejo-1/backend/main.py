import sys
import os

# Add the parent directory (project root) to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
FastAPI main application
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.config import get_settings
from backend.api import routes
from backend.rag.chroma_client import get_or_create_collection
from backend.rag.ingest import (
    ingest_data,
    compute_dataset_fingerprint,
    load_ingest_manifest,
)
from backend.utils.logger import setup_logging, get_logger

# 로깅 설정
settings = get_settings()
setup_logging(environment=settings.environment, app_name="trade_onboarding")
logger = get_logger(__name__)

# LangSmith 트레이싱 설정
# @traceable 데코레이터는 LANGSMITH_API_KEY를 참조하므로 두 변수 모두 설정
if settings.langsmith_tracing and settings.langsmith_api_key:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project

async def run_startup_tasks() -> None:
    """
    서버 시작 시 벡터 DB 초기화 및 데이터 임베딩
    - ChromaDB 컬렉션 확인
    - 컬렉션이 비어있으면 자동으로 데이터셋 임베딩 업로드
    - 이미 데이터가 있으면 스킵
    - config.auto_ingest_on_startup 설정으로 자동 임베딩 비활성화 가능
    """
    if settings.environment.lower() in {"test", "testing"}:
        logger.info("🧪 테스트 환경: startup 벡터 초기화를 건너뜁니다.")
        return

    logger.info("🚀 무역 온보딩 AI 코치 API 시작 중...")
    logger.info("📊 벡터 데이터베이스 확인 중...")

    try:
        # ChromaDB 컬렉션 가져오기 (없으면 생성)
        collection = get_or_create_collection()
        current_count = collection.count()
        dataset_fingerprint = compute_dataset_fingerprint()
        manifest = load_ingest_manifest()
        previous_fingerprint = manifest.get("dataset_fingerprint")
        dataset_changed = (
            (previous_fingerprint != dataset_fingerprint)
            if previous_fingerprint
            else (current_count > 0)
        )
        force_reingest = bool(settings.force_reingest_on_startup)
        should_reingest = (
            settings.auto_ingest_on_startup
            and (
                force_reingest
                or (settings.reingest_on_dataset_change and dataset_changed)
            )
        )

        logger.info(f"✅ 벡터 데이터베이스 연결 완료. 현재 문서 수: {current_count}")
        if previous_fingerprint:
            logger.info("📌 마지막 인덱스 fingerprint 감지")
        elif current_count > 0:
            logger.warning("⚠️ 인덱스 manifest가 없어 안전 모드 재인덱싱 대기 상태입니다.")
        if dataset_changed:
            logger.warning("⚠️ 데이터셋 변경이 감지되었습니다.")
        if force_reingest:
            logger.warning("⚠️ force_reingest_on_startup=true: 강제 재인덱싱 모드")

        # 자동 임베딩이 활성화되어 있고, 컬렉션이 비어있으면 자동으로 데이터 임베딩
        if settings.auto_ingest_on_startup and current_count == 0:
            logger.info("📥 벡터 데이터베이스가 비어있습니다. 자동 데이터 임베딩 시작...")
            logger.info("⏳ 첫 실행 시 수 분이 소요될 수 있습니다...")

            # 데이터 임베딩 및 업로드
            ingest_data(reset=False)

            # 업로드 후 카운트 확인
            final_count = collection.count()
            logger.info(f"✅ 데이터 임베딩 완료! 총 문서 수: {final_count}")
        elif should_reingest:
            logger.info("🔁 데이터셋 변경/강제 옵션으로 재인덱싱을 수행합니다.")
            logger.info("⏳ 기존 컬렉션을 재구축하므로 시간이 걸릴 수 있습니다...")
            ingest_data(reset=True)

            # 업로드 후 카운트 확인
            final_count = collection.count()
            logger.info(f"✅ 데이터 임베딩 완료! 총 문서 수: {final_count}")
        elif current_count == 0:
            logger.warning("⚠️  벡터 데이터베이스가 비어있지만, 자동 임베딩이 비활성화되어 있습니다.")
            logger.warning("💡 수동 데이터 임베딩: uv run python backend/rag/ingest.py")
        else:
            logger.info("✅ 벡터 데이터베이스에 이미 데이터가 있습니다. 임베딩 생략.")
            logger.info(f"📚 총 {current_count}개 문서 로드 완료.")

    except Exception as e:
        logger.error(f"❌ 벡터 데이터베이스 초기화 중 오류 발생: {e}")
        logger.error("⚠️  서버는 시작되지만 RAG 기능이 정상 작동하지 않을 수 있습니다.")
        logger.error("💡 재시도: uv run python backend/rag/ingest.py --reset")

    logger.info("🎉 서버 시작 완료!")


@asynccontextmanager
async def lifespan(_: FastAPI):
    await run_startup_tasks()
    yield


app = FastAPI(
    title="Trade Onboarding AI Coach",
    description="물류·무역 온보딩 AI 코치 API",
    version="1.0.0",
    debug=settings.debug,
    lifespan=lifespan,
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(routes.router, prefix="/api")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "ok",
        "message": "Trade Onboarding AI Coach API",
        "environment": settings.environment
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    )
