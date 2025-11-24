"""后台维护任务."""

import os
from datetime import datetime, timedelta
from pathlib import Path

from celery import shared_task

from app.config import settings
from app.core.database import SessionLocal
from app.models.task import Task, TaskStatus, Upload


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

        # 先删除任务及其输出文件，并记录涉及的 upload_id，避免在有其他任务引用时误删源文件。
        upload_ids: set = set()
        for task in tasks:
            if task.upload_id:
                upload_ids.add(task.upload_id)

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

        # 再根据 upload_id 检查是否还有任务引用该上传记录，若没有则删除源文件和 uploads 记录。
        for upload_id in upload_ids:
            upload = db.query(Upload).filter(Upload.id == upload_id).first()
            if not upload:
                continue

            remaining_tasks = (
                db.query(Task).filter(Task.upload_id == upload.id).count()
            )
            if remaining_tasks > 0:
                continue

            if upload.file_path and os.path.exists(upload.file_path):
                try:
                    Path(upload.file_path).unlink()
                    removed_inputs += 1
                except OSError:
                    pass

            db.delete(upload)

        db.commit()
    finally:
        db.close()

    return {
        "removed_tasks": removed_tasks,
        "removed_inputs": removed_inputs,
        "removed_outputs": removed_outputs,
    }
