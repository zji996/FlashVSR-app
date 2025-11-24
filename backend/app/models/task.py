"""任务与上传相关模型."""

import uuid
from datetime import datetime
from typing import Any, Dict, Mapping, Optional
import enum

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum as SQLEnum,
    Float,
    Integer,
    String,
    Text,
    Boolean,
    ForeignKey,
    BigInteger,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from app.core.database import Base


class TaskStatus(str, enum.Enum):
    """任务状态枚举."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Upload(Base):
    """输入文件表（去重存储上传视频的元信息）。"""

    __tablename__ = "uploads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    file_name = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False, unique=True)

    size = Column(BigInteger, nullable=True)
    sha256 = Column(String(64), nullable=True, unique=True, index=True)

    # 基础视频信息（按源文件维度聚合，便于做按文件级别的统计和查询）。
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    fps = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    total_frames = Column(Integer, nullable=True)
    bit_rate = Column(BigInteger, nullable=True)
    avg_frame_rate = Column(Float, nullable=True)

    tasks = relationship("Task", back_populates="upload")

    def __repr__(self) -> str:
        return f"<Upload {self.id} {self.file_name}>"


class Task(Base):
    """任务表模型（聚焦调度与状态），通过 upload_id 关联输入文件."""

    __tablename__ = "tasks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False
    )
    # 任务开始与完成时间（用于统计耗时）
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)

    # 任务状态
    status = Column(
        SQLEnum(TaskStatus), default=TaskStatus.PENDING, nullable=False, index=True
    )

    # 输入文件外键
    upload_id = Column(
        UUID(as_uuid=True),
        ForeignKey("uploads.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )

    # 输出信息
    output_file_path = Column(String(500), nullable=True)
    output_file_name = Column(String(255), nullable=True)

    # 进度追踪
    progress = Column(Float, default=0.0, nullable=False)  # 0-100
    total_frames = Column(Integer, nullable=True)
    processed_frames = Column(Integer, default=0, nullable=False)
    estimated_time_remaining = Column(Integer, nullable=True)  # 秒

    # Celery任务ID
    celery_task_id = Column(String(255), nullable=True, index=True)

    # 错误信息
    error_message = Column(Text, nullable=True)

    # 关系映射
    upload = relationship("Upload", back_populates="tasks")
    parameters_obj = relationship(
        "TaskParametersDB",
        uselist=False,
        back_populates="task",
        cascade="all, delete-orphan",
    )
    video_info_obj = relationship(
        "TaskVideoInfoDB",
        uselist=False,
        back_populates="task",
        cascade="all, delete-orphan",
    )

    # 兼容旧字段：用于 API 响应和删除逻辑，但实际数据存储在 Upload 表中。
    @property
    def input_file_path(self) -> Optional[str]:
        return self.upload.file_path if self.upload else None

    @property
    def input_file_name(self) -> Optional[str]:
        return self.upload.file_name if self.upload else None

    # video_info / parameters 提供 dict 视图，底层拆分到独立表，便于统计与索引。
    @property
    def video_info(self) -> Optional[Dict[str, Any]]:
        if self.video_info_obj is None:
            return None
        return self.video_info_obj.to_dict()

    @video_info.setter
    def video_info(self, value: Optional[Mapping[str, Any]]) -> None:
        if value is None:
            self.video_info_obj = None
            return
        if self.video_info_obj is None:
            self.video_info_obj = TaskVideoInfoDB.from_dict(value)
        else:
            self.video_info_obj.update_from_dict(value)

    @property
    def parameters(self) -> Optional[Dict[str, Any]]:
        if self.parameters_obj is None:
            return None
        return self.parameters_obj.to_dict()

    @parameters.setter
    def parameters(self, value: Optional[Mapping[str, Any]]) -> None:
        if value is None:
            self.parameters_obj = None
            return
        # 优先从已关联的 Upload 对象推导 upload_id，避免在 flush 之前为 NULL。
        upload_id = getattr(self, "upload_id", None)
        if upload_id is None and getattr(self, "upload", None) is not None:
            upload_id = self.upload.id
        if self.parameters_obj is None:
            self.parameters_obj = TaskParametersDB.from_dict(value, upload_id=upload_id)
        else:
            self.parameters_obj.update_from_dict(value)
            # 保持 upload_id 一致，便于唯一约束发挥作用。
            if upload_id is not None and self.parameters_obj.upload_id != upload_id:
                self.parameters_obj.upload_id = upload_id

    def __repr__(self) -> str:
        return f"<Task {self.id} - {self.status.value}>"


class TaskParametersDB(Base):
    """任务参数表（结构化 + 可扩展的高级参数存储）。"""

    __tablename__ = "task_parameters"

    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    upload_id = Column(
        UUID(as_uuid=True),
        ForeignKey("uploads.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    scale = Column(Float, nullable=False)
    sparse_ratio = Column(Float, nullable=False)
    local_range = Column(Integer, nullable=False)
    seed = Column(Integer, nullable=False)
    model_variant = Column(String(50), nullable=False)
    preprocess_width = Column(Integer, nullable=False)
    preserve_aspect_ratio = Column(Boolean, nullable=False, default=False)

    # 预留高级参数 JSON，便于将来增加新开关/数值而不再改 schema。
    advanced_options = Column(JSON, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "upload_id",
            "scale",
            "sparse_ratio",
            "local_range",
            "seed",
            "model_variant",
            "preprocess_width",
            "preserve_aspect_ratio",
            name="uq_task_parameters_upload_config",
        ),
    )

    task = relationship("Task", back_populates="parameters_obj")
    upload = relationship("Upload")

    def to_dict(self) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "scale": self.scale,
            "sparse_ratio": self.sparse_ratio,
            "local_range": self.local_range,
            "seed": self.seed,
            "model_variant": self.model_variant,
            "preprocess_width": self.preprocess_width,
            "preserve_aspect_ratio": bool(self.preserve_aspect_ratio),
        }
        if self.advanced_options:
            # 扩展字段统一摊平成顶层键，便于前后端读取。
            data.update(self.advanced_options)
        return data

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, Any],
        upload_id: Optional[uuid.UUID],
    ) -> "TaskParametersDB":
        known_keys = {
            "scale",
            "sparse_ratio",
            "local_range",
            "seed",
            "model_variant",
            "preprocess_width",
            "preserve_aspect_ratio",
        }
        data = dict(value) if value is not None else {}
        advanced: Dict[str, Any] = {
            k: v for k, v in data.items() if k not in known_keys
        }
        return cls(
            upload_id=upload_id,
            scale=float(data.get("scale")) if data.get("scale") is not None else 0.0,
            sparse_ratio=float(data.get("sparse_ratio"))
            if data.get("sparse_ratio") is not None
            else 0.0,
            local_range=int(data.get("local_range"))
            if data.get("local_range") is not None
            else 0,
            seed=int(data.get("seed")) if data.get("seed") is not None else 0,
            model_variant=str(data.get("model_variant") or ""),
            preprocess_width=int(data.get("preprocess_width"))
            if data.get("preprocess_width") is not None
            else 0,
            preserve_aspect_ratio=bool(data.get("preserve_aspect_ratio") or False),
            advanced_options=advanced or None,
        )

    def update_from_dict(self, value: Mapping[str, Any]) -> None:
        data = dict(value) if value is not None else {}
        for key in ("scale", "sparse_ratio", "local_range", "seed", "model_variant", "preprocess_width"):
            if key in data and data[key] is not None:
                setattr(self, key, data[key])
        if "preserve_aspect_ratio" in data and data["preserve_aspect_ratio"] is not None:
            self.preserve_aspect_ratio = bool(data["preserve_aspect_ratio"])

        # 同步扩展字段
        known_keys = {
            "scale",
            "sparse_ratio",
            "local_range",
            "seed",
            "model_variant",
            "preprocess_width",
            "preserve_aspect_ratio",
        }
        extras = {k: v for k, v in data.items() if k not in known_keys}
        if extras:
            base = dict(self.advanced_options or {})
            base.update(extras)
            self.advanced_options = base


class TaskVideoInfoDB(Base):
    """任务视频信息表（用于统计和报表的结构化字段）。"""

    __tablename__ = "task_video_info"

    task_id = Column(
        UUID(as_uuid=True),
        ForeignKey("tasks.id", ondelete="CASCADE"),
        primary_key=True,
    )

    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    total_frames = Column(Integer, nullable=True)
    fps = Column(Integer, nullable=True)
    duration = Column(Float, nullable=True)
    processed_frames = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)
    inference_time = Column(Float, nullable=True)
    bit_rate = Column(BigInteger, nullable=True)
    avg_frame_rate = Column(Float, nullable=True)
    preprocess_applied = Column(Boolean, nullable=True)
    preprocess_width = Column(Integer, nullable=True)
    preprocess_result_width = Column(Integer, nullable=True)
    preprocess_result_height = Column(Integer, nullable=True)
    predicted_output_width = Column(Integer, nullable=True)
    predicted_output_height = Column(Integer, nullable=True)

    task = relationship("Task", back_populates="video_info_obj")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "duration": self.duration,
            "processed_frames": self.processed_frames,
            "processing_time": self.processing_time,
            "inference_time": self.inference_time,
            "bit_rate": self.bit_rate,
            "avg_frame_rate": self.avg_frame_rate,
            "preprocess_applied": self.preprocess_applied,
            "preprocess_width": self.preprocess_width,
            "preprocess_result_width": self.preprocess_result_width,
            "preprocess_result_height": self.preprocess_result_height,
            "predicted_output_width": self.predicted_output_width,
            "predicted_output_height": self.predicted_output_height,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskVideoInfoDB":
        data = dict(value) if value is not None else {}
        return cls(
            width=data.get("width"),
            height=data.get("height"),
            total_frames=data.get("total_frames"),
            fps=data.get("fps"),
            duration=data.get("duration"),
            processed_frames=data.get("processed_frames"),
            processing_time=data.get("processing_time"),
            inference_time=data.get("inference_time"),
            bit_rate=data.get("bit_rate"),
            avg_frame_rate=data.get("avg_frame_rate"),
            preprocess_applied=data.get("preprocess_applied"),
            preprocess_width=data.get("preprocess_width"),
            preprocess_result_width=data.get("preprocess_result_width"),
            preprocess_result_height=data.get("preprocess_result_height"),
            predicted_output_width=data.get("predicted_output_width"),
            predicted_output_height=data.get("predicted_output_height"),
        )

    def update_from_dict(self, value: Mapping[str, Any]) -> None:
        data = dict(value) if value is not None else {}
        for key, val in data.items():
            if hasattr(self, key) and val is not None:
                setattr(self, key, val)
