"""FlashVSR 处理任务."""

import os
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID

from celery import Task as CeleryTask
from app.core.celery_app import celery_app
from app.core.database import SessionLocal
from app.models.task import Task, TaskStatus
from app.schemas.task import TaskParameters
from app.services.flashvsr_service import FlashVSRService
from app.services.video_preprocessor import VideoPreprocessor
from app.services.video_metadata import VideoMetadataService
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
        
        # 更新任务状态与开始时间
        task.status = TaskStatus.PROCESSING
        if task.started_at is None:
            task.started_at = datetime.utcnow()
        task.celery_task_id = self.request.id
        db.commit()
        
        print(f"📝 开始处理任务: {task_id}")
        
        # 准备文件路径
        input_path = task.input_file_path
        output_filename = f"{Path(task.input_file_name).stem}_flashvsr.mp4"
        output_path = str(settings.RESULT_DIR / output_filename)
        
        # 获取处理参数
        raw_params = task.parameters or {}
        validated_params = TaskParameters.model_validate(raw_params)
        scale = validated_params.scale or settings.DEFAULT_SCALE
        sparse_ratio = validated_params.sparse_ratio or settings.DEFAULT_SPARSE_RATIO
        local_range = validated_params.local_range or settings.DEFAULT_LOCAL_RANGE
        seed = validated_params.seed or settings.DEFAULT_SEED
        model_variant = validated_params.model_variant or settings.DEFAULT_MODEL_VARIANT

        metadata = VideoMetadataService.extract_metadata(input_path)
        preprocessor = VideoPreprocessor()
        preprocess_result = preprocessor.maybe_preprocess(
            Path(input_path),
            metadata,
            validated_params,
        )
        effective_metadata = preprocess_result.metadata
        processing_input_path = str(preprocess_result.input_path)
        preprocessed_audio_path = preprocess_result.audio_path
        audio_path = str(preprocessed_audio_path) if preprocessed_audio_path else None

        predicted_width = None
        predicted_height = None
        if effective_metadata.width and effective_metadata.height:
            _, _, predicted_width, predicted_height = FlashVSRService._compute_scaled_dims(
                effective_metadata.width,
                effective_metadata.height,
                scale,
            )

        # 更新视频信息
        video_info = task.video_info or {}
        video_info.update({
            "width": effective_metadata.width or video_info.get("width"),
            "height": effective_metadata.height or video_info.get("height"),
            "fps": effective_metadata.fps or video_info.get("fps"),
            "total_frames": effective_metadata.total_frames or video_info.get("total_frames"),
            "bit_rate": effective_metadata.bit_rate or metadata.bit_rate,
            "avg_frame_rate": effective_metadata.avg_frame_rate or metadata.avg_frame_rate,
            "preprocess_applied": preprocess_result.applied,
            "preprocess_width": validated_params.preprocess_width,
            "preprocess_result_width": effective_metadata.width,
            "preprocess_result_height": effective_metadata.height,
            "predicted_output_width": predicted_width,
            "predicted_output_height": predicted_height,
        })
        task.video_info = video_info
        if effective_metadata.total_frames:
            task.total_frames = effective_metadata.total_frames
        db.commit()
        
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
        try:
            result = flashvsr_service.process_video(
                input_path=processing_input_path,
                output_path=output_path,
                scale=scale,
                sparse_ratio=sparse_ratio,
                local_range=local_range,
                seed=seed,
                model_variant=model_variant,
                progress_callback=progress_callback,
                audio_path=audio_path,
            )
        finally:
            if preprocess_result.applied:
                preprocessor.cleanup(preprocess_result.input_path)
            preprocessor.cleanup(preprocessed_audio_path)
        
        # 更新任务状态为完成
        task.status = TaskStatus.COMPLETED
        task.finished_at = datetime.utcnow()
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
        # 更新任务状态为失败（不再在此处自动导出部分结果）
        print(f"❌ 任务失败: {task_id}, 错误: {str(e)}")

        task = db.query(Task).filter(Task.id == UUID(task_id)).first()
        if task:
            task.status = TaskStatus.FAILED
            if task.started_at is None:
                task.started_at = datetime.utcnow()
            if task.finished_at is None:
                task.finished_at = datetime.utcnow()
            task.error_message = str(e)
            db.commit()

        raise
    
    finally:
        db.close()
