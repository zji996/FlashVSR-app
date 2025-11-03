"""FlashVSR 处理任务."""

import os
import time
from pathlib import Path
from uuid import UUID

from celery import Task as CeleryTask
from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.task import Task, TaskStatus
from app.services.flashvsr_service import FlashVSRService
from app.config import settings


class CallbackTask(CeleryTask):
    """带数据库会话的Celery任务基类."""
    
    def __call__(self, *args, **kwargs):
        """执行任务."""
        return self.run(*args, **kwargs)


@celery_app.task(bind=True, base=CallbackTask, name="app.tasks.process_video")
def process_video_task(self, task_id: str):
    """
    处理视频超分辨率任务.
    
    Args:
        task_id: 任务ID
    """
    db = SessionLocal()
    
    try:
        # 获取任务
        task = db.query(Task).filter(Task.id == UUID(task_id)).first()
        if not task:
            raise ValueError(f"任务不存在: {task_id}")
        
        # 更新任务状态
        task.status = TaskStatus.PROCESSING
        task.celery_task_id = self.request.id
        db.commit()
        
        print(f"📝 开始处理任务: {task_id}")
        
        # 准备文件路径
        input_path = task.input_file_path
        output_filename = f"{Path(task.input_file_name).stem}_flashvsr.mp4"
        output_path = str(settings.RESULT_DIR / output_filename)
        
        # 获取处理参数
        params = task.parameters
        scale = params.get("scale", settings.DEFAULT_SCALE)
        sparse_ratio = params.get("sparse_ratio", settings.DEFAULT_SPARSE_RATIO)
        local_range = params.get("local_range", settings.DEFAULT_LOCAL_RANGE)
        seed = params.get("seed", settings.DEFAULT_SEED)
        
        # 进度回调函数
        def progress_callback(processed_frames: int, total_frames: int, avg_frame_time: float):
            """更新进度到数据库."""
            progress = (processed_frames / total_frames * 100) if total_frames > 0 else 0
            remaining_frames = total_frames - processed_frames
            estimated_time = int(remaining_frames * avg_frame_time) if avg_frame_time > 0 else None
            
            task.progress = progress
            task.processed_frames = processed_frames
            task.total_frames = total_frames
            task.estimated_time_remaining = estimated_time
            db.commit()
            
            # 更新Celery任务状态
            self.update_state(
                state='PROGRESS',
                meta={
                    'progress': progress,
                    'processed_frames': processed_frames,
                    'total_frames': total_frames,
                    'estimated_time_remaining': estimated_time,
                }
            )
        
        # 获取FlashVSR服务实例
        flashvsr_service = FlashVSRService()
        
        # 处理视频
        result = flashvsr_service.process_video(
            input_path=input_path,
            output_path=output_path,
            scale=scale,
            sparse_ratio=sparse_ratio,
            local_range=local_range,
            seed=seed,
            progress_callback=progress_callback,
        )
        
        # 更新任务状态为完成
        task.status = TaskStatus.COMPLETED
        task.progress = 100.0
        task.output_file_path = output_path
        task.output_file_name = output_filename
        task.processed_frames = result.get("total_frames", task.processed_frames or 0)
        task.total_frames = result.get("total_frames", task.total_frames)
        task.estimated_time_remaining = None

        # 合并视频信息
        video_info = task.video_info or {}
        video_info.update(result)
        task.video_info = video_info

        task.error_message = None
        db.commit()
        
        print(f"✅ 任务完成: {task_id}")
        
        return {
            "task_id": task_id,
            "status": "completed",
            "output_path": output_path,
        }
    
    except Exception as e:
        # 更新任务状态为失败
        print(f"❌ 任务失败: {task_id}, 错误: {str(e)}")
        
        task = db.query(Task).filter(Task.id == UUID(task_id)).first()
        if task:
            task.status = TaskStatus.FAILED
            task.error_message = str(e)
            db.commit()
        
        raise
    
    finally:
        db.close()
