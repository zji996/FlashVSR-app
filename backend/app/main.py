"""FastAPI 应用主入口."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.core.database import engine, Base
from app.api import tasks, files, system


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理."""
    # 启动时创建数据库表
    Base.metadata.create_all(bind=engine)
    yield
    # 关闭时的清理工作


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
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
