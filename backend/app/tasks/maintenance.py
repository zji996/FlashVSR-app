"""后台维护任务."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from celery import shared_task

from app.config import settings
from app.core.database import SessionLocal
from app.models.task import Task, TaskStatus


@shared_task(name="app.tasks.cleanup_storage")
def cleanup_storage():
    """清理过期的任务记录和文件."""
    cutoff = datetime.utcnow() - timedelta(days=settings.TASK_RETENTION_DAYS)
    removed_inputs = 0
    removed_outputs = 0
    removed_tasks = 0

    db = SessionLocal()
    try:
        tasks = (
            db.query(Task)
            .filter(Task.updated_at < cutoff)
            .filter(Task.status.in_([TaskStatus.COMPLETED, TaskStatus.FAILED]))
            .all()
        )

        for task in tasks:
            # 删除输入文件
            if task.input_file_path and os.path.exists(task.input_file_path):
                try:
                    Path(task.input_file_path).unlink()
                    removed_inputs += 1
                except OSError:
                    pass

            # 删除输出文件
            if task.output_file_path and os.path.exists(task.output_file_path):
                try:
                    Path(task.output_file_path).unlink()
                    removed_outputs += 1
                except OSError:
                    pass

            db.delete(task)
            removed_tasks += 1

        db.commit()
    finally:
        db.close()

    return {
        "removed_tasks": removed_tasks,
        "removed_inputs": removed_inputs,
        "removed_outputs": removed_outputs,
    }
