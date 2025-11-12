"""FastAPI 应用主入口."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.core.database import engine, Base
from app.api import tasks, files, system
from app.services.flashvsr_service import FlashVSRService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理."""
    # 启动时创建数据库表
    Base.metadata.create_all(bind=engine)

    # 预加载 FlashVSR 模型，避免首个任务时再初始化
    flashvsr_service = FlashVSRService()
    preload_variants = settings.MODEL_VARIANTS_TO_PRELOAD or [settings.DEFAULT_MODEL_VARIANT]
    for variant in dict.fromkeys(preload_variants):  # preserve order, drop duplicates
        flashvsr_service.preload_variant(variant)

    yield
    # 关闭时的清理工作


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS配置（支持关闭或全量放开）
cors_origins = settings.BACKEND_CORS_ORIGINS
if cors_origins:
    allow_all = any(origin == "*" for origin in cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if allow_all else cors_origins,
        allow_credentials=not allow_all,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# 注册路由
app.include_router(tasks.router, prefix=f"{settings.API_V1_PREFIX}/tasks", tags=["tasks"])
app.include_router(files.router, prefix=f"{settings.API_V1_PREFIX}/files", tags=["files"])
app.include_router(system.router, prefix=f"{settings.API_V1_PREFIX}/system", tags=["system"])


@app.get("/")
async def root():
    """根路径."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
    }


@app.get("/health")
async def health():
    """健康检查."""
    return {"status": "healthy"}
