"""任务管理API路由."""

import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import desc
from sqlalchemy.exc import IntegrityError

from pydantic import ValidationError

from app.core.database import get_db
from app.core.celery_app import celery_app
from app.models.task import Task, TaskStatus, Upload
from app.services.flashvsr_service import FlashVSRService
from app.schemas.task import (
    TaskResponse,
    TaskListResponse,
    TaskProgressResponse,
    TaskParameters,
    TaskParameterField,
    TaskParameterSchemaResponse,
    ParameterOption,
    TaskPresetProfile,
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
    preserve_aspect_ratio: bool = Form(default=False),
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
    hasher = hashlib.sha256()
    try:
        file.file.seek(0)
        with open(upload_path, "wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                buffer.write(chunk)
                hasher.update(chunk)
    except Exception as e:
        if upload_path.exists():
            upload_path.unlink()
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}") from e
    finally:
        await file.close()

    sha256 = hasher.hexdigest()

    # 创建任务记录
    try:
        parameters = TaskParameters(
            scale=scale,
            sparse_ratio=sparse_ratio,
            local_range=local_range,
            seed=seed,
            model_variant=settings.DEFAULT_MODEL_VARIANT,
            preprocess_width=preprocess_width,
            preserve_aspect_ratio=preserve_aspect_ratio,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=400, detail=exc.errors()) from exc

    # 复用或创建 Upload 记录，避免同一文件多次上传时重复存储元数据。
    upload = (
        db.query(Upload)
        .filter(Upload.sha256.isnot(None), Upload.sha256 == sha256)
        .first()
    )
    if upload:
        existing_path = upload.file_path
        existing_ok = bool(existing_path) and os.path.exists(existing_path)

        if existing_ok:
            # 同一内容已存在且物理文件仍在，删除刚写入的重复文件以节省空间。
            try:
                if upload_path.exists() and existing_path != str(upload_path):
                    upload_path.unlink()
            except Exception:
                pass
        else:
            # 数据库里记录的路径已失效（文件被手工删除或迁移），
            # 使用本次上传的文件作为新的 canonical 路径，修复旧记录。
            upload.file_path = str(upload_path)
            upload.size = file_size

        # 更新文件名为最近一次上传的名称，便于前端展示。
        if upload.file_name != original_name:
            upload.file_name = original_name
    else:
        upload = Upload(
            file_name=original_name,
            file_path=str(upload_path),
            size=file_size,
            sha256=sha256,
        )
        db.add(upload)
        db.flush()  # 立即获得 upload.id 以便后续关联任务

    task = Task(
        status=TaskStatus.PENDING,
        total_frames=None,
    )
    task.upload = upload
    task.parameters = parameters.model_dump()
    task.video_info = {
        "preprocess_width": preprocess_width,
    }

    db.add(task)
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        message = str(exc.orig) if hasattr(exc, "orig") else str(exc)
        if "uq_task_parameters_upload_config" in message:
            raise HTTPException(
                status_code=409,
                detail="已存在使用相同输入文件与参数配置的任务，请勿重复提交。",
            ) from exc
        raise HTTPException(status_code=500, detail="创建任务失败") from exc

    db.refresh(task)

    # 提交Celery任务
    celery_task = process_video_task.delay(str(task.id))

    # 更新Celery任务ID
    task.celery_task_id = celery_task.id
    db.commit()
    db.refresh(task)
    
    return task


@router.get("/parameter_schema", response_model=TaskParameterSchemaResponse)
def get_task_parameter_schema() -> TaskParameterSchemaResponse:
    """
    返回任务参数的元数据，供前端自动生成表单使用。

    - fields: 描述每个字段的标签、范围、推荐值等
    - presets: 预设组合（例如“预处理 960px + 2× 超分接近 1080p”）
    """
    # 预处理宽度字段（左侧独立区块）
    preprocess_field = TaskParameterField(
        name="preprocess_width",
        label="预处理宽度 (Preprocess Width)",
        description=(
            "预处理缩放宽度（像素，建议常见档位如 640/768/896/960/1024/1152/1280）。"
        ),
        field_type="number",
        min=128,
        max=None,
        step=1,
        required=True,
        default=640,
        recommended=[
            ParameterOption(label="640 px", value=640),
            ParameterOption(label="768 px", value=768),
            ParameterOption(label="896 px", value=896),
            ParameterOption(label="960 px", value=960),
            ParameterOption(label="1024 px", value=1024),
            ParameterOption(label="1152 px", value=1152),
            ParameterOption(label="1280 px", value=1280),
        ],
        ui_group="preprocess",
    )

    # 高级参数字段（右侧折叠区）
    advanced_fields: list[TaskParameterField] = [
        TaskParameterField(
            name="scale",
            label="超分倍数 (Scale)",
            description="放大倍数，过高会显著增加显存和时间开销。",
            field_type="number",
            min=1.0,
            max=8.0,
            step=0.1,
            required=True,
            default=settings.DEFAULT_SCALE,
            recommended=[
                ParameterOption(label="默认", value=2.0, description="推荐值: 2.0"),
            ],
            ui_group="advanced",
        ),
        TaskParameterField(
            name="sparse_ratio",
            label="稀疏比率 (Sparse Ratio)",
            description="控制 WanVSR 稀疏注意力的稀疏度，数值越大越稳定但越慢。",
            field_type="number",
            min=1.0,
            max=4.0,
            step=0.1,
            required=True,
            default=2.0,
            recommended=[
                ParameterOption(label="1.5（更快）", value=1.5, description="推荐值: 1.5 (快)"),
                ParameterOption(label="2.0（更稳定）", value=2.0, description="推荐值: 2.0 (稳定)"),
            ],
            ui_group="advanced",
        ),
        TaskParameterField(
            name="local_range",
            label="局部范围 (Local Range)",
            description="控制局部注意力窗口大小，影响锐度与稳定性。",
            field_type="number",
            min=7,
            max=15,
            step=2,
            required=True,
            default=11,
            recommended=[
                ParameterOption(label="9（更锐利）", value=9, description="推荐值: 9 (更锐利)"),
                ParameterOption(label="11（更稳定）", value=11, description="推荐值: 11 (更稳定)"),
            ],
            ui_group="advanced",
        ),
        TaskParameterField(
            name="seed",
            label="随机种子 (Seed)",
            description="控制随机性；设置为 0 表示每次随机。",
            field_type="number",
            min=0,
            max=None,
            step=1,
            required=True,
            default=0,
            recommended=[
                ParameterOption(label="0（随机）", value=0, description="0 为随机"),
            ],
            ui_group="advanced",
        ),
        TaskParameterField(
            name="preserve_aspect_ratio",
            label="按原始长宽比裁剪黑边",
            description=(
                "导出时按输入视频的长宽比裁剪黑边恢复画面（会多一次轻量级重编码）。"
            ),
            field_type="boolean",
            required=True,
            default=False,
            recommended=[],
            ui_group="advanced",
        ),
    ]

    presets: list[TaskPresetProfile] = [
        TaskPresetProfile(
            key="1080p",
            label="接近 1080p",
            description="预处理 960px + 2× 超分，适合高清流媒体素材",
            preprocess_width=960,
            scale=2.0,
        ),
        TaskPresetProfile(
            key="2k",
            label="锐利 2K",
            description="预处理 1152px + 2×，在 16:9 视频上接近 2304px",
            preprocess_width=1152,
            scale=2.0,
        ),
        TaskPresetProfile(
            key="fast",
            label="快速出图",
            description="预处理 768px + 2×，更省显存的批量模式",
            preprocess_width=768,
            scale=2.0,
        ),
    ]

    return TaskParameterSchemaResponse(
        fields=[preprocess_field, *advanced_fields],
        presets=presets,
    )


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

    upload = task.upload

    # 删除文件
    try:
        # 仅在没有其他任务引用同一上传记录时才删除源文件
        if upload and upload.file_path:
            other_tasks_count = (
                db.query(Task)
                .filter(Task.upload_id == upload.id, Task.id != task.id)
                .count()
            )
            if other_tasks_count == 0 and os.path.exists(upload.file_path):
                os.remove(upload.file_path)

        if task.output_file_path and os.path.exists(task.output_file_path):
            os.remove(task.output_file_path)
    except Exception as e:
        print(f"文件删除失败: {str(e)}")

    # 删除数据库记录（同时触发 task_parameters / task_video_info 的级联删除）
    db.delete(task)

    # 若不再有任务引用该上传记录，则一并清理 uploads 记录
    if upload:
        remaining = (
            db.query(Task).filter(Task.upload_id == upload.id).count()
        )
        if remaining == 0:
            db.delete(upload)

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
