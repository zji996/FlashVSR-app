"""应用配置管理."""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path


class Settings(BaseSettings):
    """应用设置."""

    # 基础配置
    APP_NAME: str = "FlashVSR API"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = False

    # API配置
    API_V1_PREFIX: str = "/api"

    # 数据库配置
    DATABASE_URL: str = "postgresql://flashvsr:flashvsr@postgres:5432/flashvsr"

    # Redis配置
    REDIS_URL: str = "redis://redis:6379/0"
    CELERY_BROKER_URL: str = "redis://redis:6379/1"
    CELERY_RESULT_BACKEND: str = "redis://redis:6379/2"

    # 项目与路径配置
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    STORAGE_ROOT: Path = PROJECT_ROOT / "storage"
    UPLOAD_DIR: Path = STORAGE_ROOT / "uploads"
    RESULT_DIR: Path = STORAGE_ROOT / "results"
    MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024 * 1024  # 2GB

    # 模型配置
    MODEL_ROOT: Path = PROJECT_ROOT / "models"
    FLASHVSR_MODEL_PATH: Path = MODEL_ROOT / "FlashVSR"
    THIRD_PARTY_ROOT: Path = PROJECT_ROOT / "third_party"
    THIRD_PARTY_FLASHVSR_PATH: Path = THIRD_PARTY_ROOT / "FlashVSR"

    # 任务配置
    MAX_CONCURRENT_TASKS: int = 1  # GPU限制
    TASK_RETENTION_DAYS: int = 30
    DEFAULT_SCALE: float = 4.0
    DEFAULT_SPARSE_RATIO: float = 2.0
    DEFAULT_LOCAL_RANGE: int = 11
    DEFAULT_SEED: int = 0

    # CORS配置
    BACKEND_CORS_ORIGINS: list[str] = ["http://localhost:5173", "http://localhost:3000"]

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",
    )

    @property
    def UPLOAD_PATH(self) -> Path:
        """兼容旧名称."""
        return self.UPLOAD_DIR

    @property
    def RESULT_PATH(self) -> Path:
        """兼容旧名称."""
        return self.RESULT_DIR


settings = Settings()

# 确保目录存在
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULT_DIR.mkdir(parents=True, exist_ok=True)
