"""
Configuration management using .env file
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from .env file"""
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Upstage API
    upstage_api_key: str = ""
    embedding_provider: str = "local"  # local | upstage | auto
    local_embedding_dim: int = 4096

    # LangSmith
    langsmith_api_key: str = ""
    langsmith_project: str = "trade-onboarding-agent"
    langsmith_tracing: bool = True

    # Database (optional)
    database_url: str = ""

    # Redis Session Store
    redis_url: str = ""  # Example: "redis://localhost:6379/0"
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0
    redis_ssl: bool = False
    session_ttl: int = 3600  # Session TTL in seconds (1 hour)
    use_redis_session: bool = False  # True for production, False for development

    # Application
    environment: str = "development"
    debug: bool = True

    # CORS
    cors_origins: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ]

    # Vector Database
    vector_db_dir: str = "backend/vectorstore"
    collection_name: str = "trade_coaching_knowledge"
    auto_ingest_on_startup: bool = True  # 서버 시작 시 자동 임베딩 여부
    reingest_on_dataset_change: bool = True  # 데이터셋 변경 시 재인덱싱
    force_reingest_on_startup: bool = False  # 서버 시작 시 강제 재인덱싱

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()
