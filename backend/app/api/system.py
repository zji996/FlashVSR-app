"""系统信息API路由."""

import torch
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.database import get_db
from app.models.task import Task, TaskStatus

router = APIRouter()


@router.get("/status")
def get_system_status(db: Session = Depends(get_db)):
    """获取系统状态信息."""
    # 任务统计
    total_tasks = db.query(Task).count()
    pending_tasks = db.query(Task).filter(Task.status == TaskStatus.PENDING).count()
    processing_tasks = db.query(Task).filter(Task.status == TaskStatus.PROCESSING).count()
    completed_tasks = db.query(Task).filter(Task.status == TaskStatus.COMPLETED).count()
    failed_tasks = db.query(Task).filter(Task.status == TaskStatus.FAILED).count()
    
    # GPU信息
    gpu_available = torch.cuda.is_available()
    gpu_info = {}
    
    if gpu_available:
        gpu_info = {
            "name": torch.cuda.get_device_name(0),
            "count": torch.cuda.device_count(),
            "memory_allocated": torch.cuda.memory_allocated(0) / 1024**3,  # GB
            "memory_reserved": torch.cuda.memory_reserved(0) / 1024**3,  # GB
            "memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,  # GB
        }
    
    return {
        "gpu_available": gpu_available,
        "gpu_info": gpu_info if gpu_available else None,
        "tasks": {
            "total": total_tasks,
            "pending": pending_tasks,
            "processing": processing_tasks,
            "completed": completed_tasks,
            "failed": failed_tasks,
        }
    }

