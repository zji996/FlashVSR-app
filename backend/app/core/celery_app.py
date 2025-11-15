"""Celery应用配置."""

from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "flashvsr",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["app.tasks.flashvsr_task", "app.tasks.maintenance"],
)

# Celery配置
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    # 对于超长视频推理，取消统一的任务时间上限，由调用方自行控制。
    # 如需重新启用，可在此处配置 task_time_limit / task_soft_time_limit。
    worker_prefetch_multiplier=1,  # 一次只取一个任务
    worker_max_tasks_per_child=1,  # 每个worker处理1个任务后重启（释放GPU内存）
)

# 定时任务调度（需要 Celery Beat）
celery_app.conf.beat_schedule = {
    "cleanup-storage-daily": {
        "task": "app.tasks.cleanup_storage",
        "schedule": crontab(hour=3, minute=0),
    },
}
