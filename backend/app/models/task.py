"""任务模型."""

import uuid
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, String, Float, Integer, DateTime, Text, Enum as SQLEnum, JSON
from sqlalchemy.dialects.postgresql import UUID
import enum

from app.core.database import Base


class TaskStatus(str, enum.Enum):
    """任务状态枚举."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Task(Base):
    """任务表模型."""
    
    __tablename__ = "tasks"
    
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    
    # 任务状态
    status = Column(SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True)
    
    # 文件路径
    input_file_path = Column(String(500), nullable=False)
    input_file_name = Column(String(255), nullable=False)
    output_file_path = Column(String(500), nullable=True)
    output_file_name = Column(String(255), nullable=True)
    
    # 视频信息
    video_info = Column(JSON, nullable=True)  # 存储原始视频信息：宽、高、帧数、fps等
    
    # 处理参数
    parameters = Column(JSON, nullable=False)  # 存储scale, sparse_ratio, local_range, seed等
    
    # 进度追踪
    progress = Column(Float, default=0.0, nullable=False)  # 0-100
    total_frames = Column(Integer, nullable=True)
    processed_frames = Column(Integer, default=0, nullable=False)
    estimated_time_remaining = Column(Integer, nullable=True)  # 秒
    
    # Celery任务ID
    celery_task_id = Column(String(255), nullable=True, index=True)
    
    # 错误信息
    error_message = Column(Text, nullable=True)
    
    def __repr__(self):
        return f"<Task {self.id} - {self.status.value}>"

