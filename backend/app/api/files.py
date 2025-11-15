"""文件管理API路由."""

import os
from io import BytesIO
from uuid import UUID

import imageio
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.task import Task, TaskStatus

router = APIRouter()


@router.get("/{task_id}/result")
def download_result(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """下载处理结果视频."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if not task.output_file_path or not os.path.exists(task.output_file_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")

    # 允许两种情况下载：
    # 1. 任务正常完成（COMPLETED）——完整结果；
    # 2. 任务失败但已生成部分结果（FAILED + error_message 中包含 “已导出部分结果”）。
    if task.status == TaskStatus.COMPLETED:
        pass
    elif task.status == TaskStatus.FAILED and task.error_message and "已导出部分结果" in task.error_message:
        pass
    else:
        raise HTTPException(status_code=400, detail="任务尚未完成或没有可用的部分结果")
    
    return FileResponse(
        path=task.output_file_path,
        filename=task.output_file_name,
        media_type="video/mp4",
    )


@router.get("/{task_id}/preview")
def get_result_preview(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """获取结果视频的封面图."""
    task = db.query(Task).filter(Task.id == task_id).first()

    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if not task.output_file_path or not os.path.exists(task.output_file_path):
        raise HTTPException(status_code=404, detail="结果文件不存在")

    try:
        reader = imageio.get_reader(task.output_file_path)
        frame = reader.get_data(0)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"生成预览失败: {exc}") from exc
    finally:
        try:
            reader.close()
        except Exception:
            pass

    image = Image.fromarray(frame).convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="image/jpeg")


@router.get("/{task_id}/input")
def download_input(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """下载原始输入视频."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    if not os.path.exists(task.input_file_path):
        raise HTTPException(status_code=404, detail="输入文件不存在")
    
    return FileResponse(
        path=task.input_file_path,
        filename=task.input_file_name,
        media_type="video/mp4",
    )
