"""任务相关的Pydantic schemas."""

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import settings
from app.models.task import TaskStatus


class TaskParameters(BaseModel):
    """任务处理参数."""
    scale: float = Field(
        default=settings.DEFAULT_SCALE,
        ge=1.0,
        le=8.0,
        description="超分倍数",
    )
    sparse_ratio: float = Field(default=2.0, ge=1.0, le=4.0, description="稀疏比率")
    local_range: int = Field(default=11, ge=7, le=15, description="局部范围")
    seed: int = Field(default=0, ge=0, description="随机种子")
    model_variant: Literal["tiny_long"] = Field(
        default=settings.DEFAULT_MODEL_VARIANT,
        description="FlashVSR 模型变体（仅 tiny_long）",
    )
    preprocess_width: int = Field(
        default=640,
        ge=128,
        description="预处理缩放宽度（像素，建议常见档位如 640/768/896/960/1024/1152/1280）",
    )


class VideoInfo(BaseModel):
    """视频信息."""
    width: Optional[int] = None
    height: Optional[int] = None
    total_frames: Optional[int] = None
    fps: Optional[int] = None
    duration: Optional[float] = None
    processed_frames: Optional[int] = None
    processing_time: Optional[float] = None
    inference_time: Optional[float] = None
    bit_rate: Optional[int] = None
    avg_frame_rate: Optional[float] = None
    preprocess_applied: Optional[bool] = None
    preprocess_width: Optional[int] = None
    preprocess_result_width: Optional[int] = None
    preprocess_result_height: Optional[int] = None
    predicted_output_width: Optional[int] = None
    predicted_output_height: Optional[int] = None


class TaskCreate(BaseModel):
    """创建任务请求."""
    parameters: TaskParameters = Field(default_factory=TaskParameters)


class TaskResponse(BaseModel):
    """任务响应."""
    model_config = ConfigDict(from_attributes=True)
    
    id: UUID
    created_at: datetime
    updated_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    status: TaskStatus
    
    input_file_name: str
    output_file_name: Optional[str] = None
    
    video_info: Optional[VideoInfo] = None
    parameters: TaskParameters
    
    progress: float
    total_frames: Optional[int] = None
    processed_frames: int
    estimated_time_remaining: Optional[int] = None
    
    error_message: Optional[str] = None


class TaskListResponse(BaseModel):
    """任务列表响应."""
    tasks: list[TaskResponse]
    total: int
    page: int
    page_size: int


class TaskProgressResponse(BaseModel):
    """任务进度响应."""
    task_id: UUID
    status: TaskStatus
    progress: float
    started_at: datetime | None = None
    finished_at: datetime | None = None
    processed_frames: int
    total_frames: Optional[int]
    estimated_time_remaining: Optional[int]
    error_message: Optional[str] = None
