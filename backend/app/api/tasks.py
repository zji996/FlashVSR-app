"""任务管理API路由."""

from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc

from pydantic import ValidationError

from app.core.database import get_db
from app.core.celery_app import celery_app
from app.models.task import Task, TaskStatus
from app.services.flashvsr_service import FlashVSRService
from app.schemas.task import (
    TaskResponse,
    TaskListResponse,
    TaskProgressResponse,
    TaskParameters,
)
from app.tasks.flashvsr_task import process_video_task
from app.config import settings

router = APIRouter()


@router.post("/", response_model=TaskResponse, status_code=201)
async def create_task(
    file: UploadFile = File(...),
    scale: float = Form(default=settings.DEFAULT_SCALE),
    sparse_ratio: float = Form(default=2.0),
    local_range: int = Form(default=11),
    seed: int = Form(default=0),
    preprocess_width: int = Form(default=640),
    db: Session = Depends(get_db),
):
    """
    创建新任务并上传视频文件.
    
    - **file**: 视频文件
    - **scale**: 超分倍数 (1.0-8.0)
    - **sparse_ratio**: 稀疏比率 (1.0-4.0)
    - **local_range**: 局部范围 (7-15)
    - **seed**: 随机种子（当前仅 Tiny Long 变体，前端无需选择模型）
    - **preprocess_width**: 预处理目标宽度（像素，建议 640-1280 常见档位）

    注：接口会在文件写入并排队 Celery 后立即返回，详细的元信息由后台任务填充。
    """
    # 验证文件类型
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")
    
    file_ext = Path(file.filename).suffix.lower()
    allowed_extensions = [
        '.mp4',
        '.mov',
        '.avi',
        '.mkv',
        '.ts',
        '.m2ts',
        '.mts',
        '.m4s',
        '.mpg',
        '.mpeg',
        '.webm',
    ]
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型。支持的格式: {', '.join(allowed_extensions)}"
        )
    
    # 验证文件大小
    file.file.seek(0, 2)  # 移到文件末尾
    file_size = file.file.tell()
    file.file.seek(0)  # 回到开头
    
    if file_size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"文件太大。最大支持 {settings.MAX_UPLOAD_SIZE / 1024 / 1024 / 1024:.1f}GB"
        )
    
    # 生成唯一文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    original_name = Path(file.filename).name
    safe_filename = f"{timestamp}_{original_name}"
    upload_path = settings.UPLOAD_DIR / safe_filename
    
    # 保存上传的文件
    try:
        file.file.seek(0)
        with open(upload_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}") from e
    finally:
        await file.close()

    # 创建任务记录
    try:
        parameters = TaskParameters(
            scale=scale,
            sparse_ratio=sparse_ratio,
            local_range=local_range,
            seed=seed,
            model_variant=settings.DEFAULT_MODEL_VARIANT,
            preprocess_width=preprocess_width,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc
    
    task = Task(
        input_file_path=str(upload_path),
        input_file_name=original_name,
        parameters=parameters.model_dump(),
        status=TaskStatus.PENDING,
        video_info={
            "preprocess_width": preprocess_width,
        },
        total_frames=None,
    )
    
    db.add(task)
    db.commit()
    db.refresh(task)
    
    # 提交Celery任务
    celery_task = process_video_task.delay(str(task.id))
    
    # 更新Celery任务ID
    task.celery_task_id = celery_task.id
    db.commit()
    db.refresh(task)
    
    return task


@router.get("/", response_model=TaskListResponse)
def list_tasks(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    status: Optional[TaskStatus] = Query(default=None),
    db: Session = Depends(get_db),
):
    """
    获取任务列表（分页）.
    
    - **page**: 页码（从1开始）
    - **page_size**: 每页数量
    - **status**: 筛选状态（可选）
    """
    query = db.query(Task)
    
    # 状态筛选
    if status:
        query = query.filter(Task.status == status)
    
    # 总数
    total = query.count()
    
    # 分页
    query = query.order_by(desc(Task.created_at))
    query = query.offset((page - 1) * page_size).limit(page_size)
    
    tasks = query.all()
    
    return TaskListResponse(
        tasks=tasks,
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{task_id}", response_model=TaskResponse)
def get_task(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """获取任务详情."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return task


@router.get("/{task_id}/progress", response_model=TaskProgressResponse)
def get_task_progress(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """获取任务实时进度."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    return TaskProgressResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        started_at=task.started_at,
        finished_at=task.finished_at,
        processed_frames=task.processed_frames,
        total_frames=task.total_frames,
        estimated_time_remaining=task.estimated_time_remaining,
        error_message=task.error_message,
    )


@router.delete("/{task_id}", status_code=204)
def delete_task(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """删除任务及相关文件."""
    task = db.query(Task).filter(Task.id == task_id).first()
    
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    
    # 删除文件
    try:
        if os.path.exists(task.input_file_path):
            os.remove(task.input_file_path)
        
        if task.output_file_path and os.path.exists(task.output_file_path):
            os.remove(task.output_file_path)
    except Exception as e:
        print(f"文件删除失败: {str(e)}")
    
    # 删除数据库记录
    db.delete(task)
    db.commit()
    
    return None


@router.post("/{task_id}/export_from_chunks", response_model=TaskProgressResponse)
def export_from_chunks(
    task_id: UUID,
    db: Session = Depends(get_db),
):
    """
    基于磁盘上的 chunks_* 分片目录，尽可能导出当前可恢复的部分结果。

    仅依赖已写入磁盘的分片文件，不再尝试优雅取消正在运行的任务。
    适用于任务已结束（超时 / 崩溃等）但仍保留中间分片的情况。
    """
    task = db.query(Task).filter(Task.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")

    if not task.celery_task_id:
        raise HTTPException(status_code=400, detail="任务缺少 Celery 任务ID，无法导出当前进度")

    async_result = celery_app.AsyncResult(task.celery_task_id)

    # 仅在 Celery 任务已结束时恢复分片；仍在运行中的任务需要等待完成或手动停止。
    if not async_result.ready():
        raise HTTPException(status_code=400, detail="任务仍在处理中，请在任务结束后再尝试导出当前进度")

    # 尝试基于已有分片恢复部分结果。
    # 计算期望输出路径（与后台任务一致的命名规则）。
    expected_output_name = task.output_file_name or f"{Path(task.input_file_name).stem}_flashvsr.mp4"
    expected_output_path = str(settings.RESULT_DIR / expected_output_name)

    service = FlashVSRService()
    partial = service.export_partial_from_chunks(expected_output_path)

    if not partial:
        task.status = TaskStatus.FAILED
        if not task.error_message:
            task.error_message = "未找到可用于恢复的分片文件，无法导出当前进度。"
        db.commit()
        return TaskProgressResponse(
            task_id=task.id,
            status=task.status,
            progress=task.progress,
            started_at=task.started_at,
            finished_at=task.finished_at,
            processed_frames=task.processed_frames,
            total_frames=task.total_frames,
            estimated_time_remaining=task.estimated_time_remaining,
            error_message=task.error_message,
        )

    # 记录部分结果路径
    task.output_file_path = str(partial)
    task.output_file_name = partial.name
    task.status = TaskStatus.FAILED
    msg = task.error_message or ""
    extra = f"已导出部分结果: {partial}"
    if extra not in msg:
        msg = f"{msg}；{extra}" if msg else extra
    task.error_message = msg
    db.commit()

    return TaskProgressResponse(
        task_id=task.id,
        status=task.status,
        progress=task.progress,
        started_at=task.started_at,
        finished_at=task.finished_at,
        processed_frames=task.processed_frames,
        total_frames=task.total_frames,
        estimated_time_remaining=task.estimated_time_remaining,
        error_message=task.error_message,
    )
