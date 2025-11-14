"""应用配置管理."""

from pathlib import Path
import re
from typing import Any, ClassVar

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic_core import PydanticUndefined


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
    # PROJECT_ROOT 指向项目根目录 (FlashVSR-app/)
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
    # BACKEND_ROOT 指向 backend/ 目录
    BACKEND_ROOT: Path = Path(__file__).resolve().parent.parent
    
    # Storage 和 Models 现在在 backend/ 目录下
    STORAGE_ROOT: Path = BACKEND_ROOT / "storage"
    UPLOAD_DIR: Path = STORAGE_ROOT / "uploads"
    RESULT_DIR: Path = STORAGE_ROOT / "results"
    MAX_UPLOAD_SIZE: int = 4 * 1024 * 1024 * 1024  # 4GB

    # third_party 保留在项目根目录
    THIRD_PARTY_ROOT: Path = PROJECT_ROOT / "third_party"
    THIRD_PARTY_FLASHVSR_PATH: Path = THIRD_PARTY_ROOT / "FlashVSR"
    THIRD_PARTY_BLOCK_SPARSE_PATH: Path = THIRD_PARTY_ROOT / "Block-Sparse-Attention"

    # 模型配置（models/ 移动到 backend/ 下）
    MODEL_ROOT: Path = BACKEND_ROOT / "models"
    FLASHVSR_VERSION: str = "v1.1"
    FLASHVSR_MODEL_PATH: Path = MODEL_ROOT / "FlashVSR-v1.1"
    FLASHVSR_PROMPT_TENSOR_PATH: Path = (
        THIRD_PARTY_FLASHVSR_PATH / "examples" / "WanVSR" / "prompt_tensor" / "posi_prompt.pth"
    )
    FLASHVSR_CACHE_OFFLOAD: str = "auto"  # auto | cpu | none
    FLASHVSR_CACHE_OFFLOAD_AUTO_THRESHOLD_GB: float = 24.0
    # 可选：指定设备，例如 "cuda", "cuda:0", "cuda:1"，为空则自动选择
    FLASHVSR_DEVICE: str = ""
    # 流水线并行配置：逗号分隔设备列表，例如 "cuda:0,cuda:1" 或 "0,1"；为空表示关闭
    FLASHVSR_PP_DEVICES: str = ""
    # 流水线并行的分割层（以 block 索引计，包含左侧）。为空或 "auto" 则在中间切分
    FLASHVSR_PP_SPLIT_BLOCK: str = "auto"
    # 是否在单视频上启用两段流水线的窗口级重叠（Stage0(t+1) 与 Stage1(t) 并行）
    FLASHVSR_PP_OVERLAP: bool = False
    FLASHVSR_STREAMING_LQ_MAX_BYTES: int = 0
    FLASHVSR_STREAMING_PREFETCH_FRAMES: int = 25
    FLASHVSR_STREAMING_DECODE_THREADS: int = 2
    FLASHVSR_CHUNKED_SAVE_MIN_FRAMES: int = 600
    FLASHVSR_CHUNKED_SAVE_CHUNK_SIZE: int = 120
    FLASHVSR_CHUNKED_SAVE_TMP_DIR: Path = STORAGE_ROOT / "tmp"
    FFMPEG_BINARY: str = "ffmpeg"
    FFPROBE_BINARY: str = "ffprobe"
    PREPROCESS_TMP_DIR: Path = STORAGE_ROOT / "tmp"
    PREPROCESS_FFMPEG_PRESET: str = "veryfast"
    PREPROCESS_FFMPEG_CRF: int = 23
    PREPROCESS_FFMPEG_VIDEO_CODEC: str = "libx264"  # libx264|h264_nvenc|libx265|hevc_nvenc
    PREPROCESS_FFMPEG_HWACCEL: str = ""            # "cuda" to enable hwaccel
    PREPROCESS_NVENC_PRESET: str = "p5"            # p1..p7 (NVENC-specific)
    PREPROCESS_NVENC_RC: str = "vbr_hq"            # constqp|vbr|vbr_hq|cbr
    PREPROCESS_NVENC_CQ: int = 21                  # NVENC quality level

    # 任务配置
    MAX_CONCURRENT_TASKS: int = 1  # GPU限制
    TASK_RETENTION_DAYS: int = 30
    DEFAULT_SCALE: float = 2.0
    DEFAULT_SPARSE_RATIO: float = 2.0
    DEFAULT_LOCAL_RANGE: int = 11
    DEFAULT_SEED: int = 0
    DEFAULT_MODEL_VARIANT: str = "tiny_long"
    MODEL_VARIANTS_TO_PRELOAD: list[str] = []

    # CORS配置
    BACKEND_CORS_ORIGINS: list[str] = ["*"]

    model_config = SettingsConfigDict(
        # 从 backend/.env 加载环境变量
        env_file=str(Path(__file__).resolve().parent.parent / ".env"),
        case_sensitive=True,
        extra="ignore",
    )

    _SIZE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*([0-9]+(?:\.[0-9]+)?)\s*([kmgt]?i?b|[kmgt]?b|[kmgt]?|bytes)?\s*$",
        re.IGNORECASE,
    )

    UNIT_MAP: ClassVar[dict[str, int]] = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "k": 1024,
        "kb": 1024,
        "kib": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "mib": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "gib": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
        "tib": 1024**4,
    }

    @field_validator("FLASHVSR_STREAMING_LQ_MAX_BYTES", mode="before")
    @classmethod
    def _parse_stream_limit(cls, value: Any) -> int:
        return cls._parse_size_to_bytes(value, default=0)

    @field_validator("FLASHVSR_STREAMING_PREFETCH_FRAMES")
    @classmethod
    def _validate_stream_prefetch(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("FLASHVSR_STREAMING_PREFETCH_FRAMES must be > 0")
        return value

    @field_validator("FLASHVSR_STREAMING_DECODE_THREADS")
    @classmethod
    def _validate_stream_threads(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("FLASHVSR_STREAMING_DECODE_THREADS must be > 0")
        return value

    @classmethod
    def _parse_size_to_bytes(cls, value: Any, *, default: int | None = None) -> int:
        default_value = 0 if default is None else default
        if value is PydanticUndefined or value is None:
            return default_value
        if isinstance(value, (int, float)):
            return int(value)
        if not isinstance(value, str):
            raise ValueError("Size must be int, float, or string like '3GB'")

        normalized = value.replace("_", "").strip()
        if not normalized:
            return default_value
        if normalized == "0":
            return 0

        match = cls._SIZE_PATTERN.match(normalized)
        if not match:
            raise ValueError(
                f"无法解析大小 '{value}'，请使用 0、字节数或 '3GB'/'512MB' 形式。"
            )

        number = float(match.group(1))
        unit = (match.group(2) or "b").lower()
        unit = unit.rstrip("b") if unit in {"k", "m", "g", "t"} else unit
        multiplier = cls.UNIT_MAP.get(unit, 1)
        return int(number * multiplier)

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
