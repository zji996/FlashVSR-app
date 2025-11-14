/**
 * 任务卡片组件
 */

import { Link } from 'react-router-dom';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { tasksApi } from '../api/tasks';
import { MODEL_VARIANT_LABELS, TaskStatus } from '../types';
import type { Task } from '../types';
import ProgressBar from './ProgressBar';

interface TaskCardProps {
  task: Task;
}

export default function TaskCard({ task }: TaskCardProps) {
  const queryClient = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: () => tasksApi.deleteTask(task.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['tasks'] });
    },
  });

  const handleDelete = () => {
    if (confirm('确定要删除这个任务吗？')) {
      deleteMutation.mutate();
    }
  };

  const getStatusBadge = (status: TaskStatus) => {
    const badges = {
      [TaskStatus.PENDING]: 'bg-yellow-100 text-yellow-800',
      [TaskStatus.PROCESSING]: 'bg-blue-100 text-blue-800',
      [TaskStatus.COMPLETED]: 'bg-green-100 text-green-800',
      [TaskStatus.FAILED]: 'bg-red-100 text-red-800',
    };
    const labels = {
      [TaskStatus.PENDING]: '等待中',
      [TaskStatus.PROCESSING]: '处理中',
      [TaskStatus.COMPLETED]: '已完成',
      [TaskStatus.FAILED]: '失败',
    };
    return (
      <span className={`px-3 py-1 rounded-full text-sm font-medium ${badges[status]}`}>
        {labels[status]}
      </span>
    );
  };

  const formatTime = (seconds: number | undefined) => {
    if (seconds === undefined || seconds === null) return '--';
    if (seconds < 60) return `${seconds}秒`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)}分${seconds % 60}秒`;
    return `${Math.floor(seconds / 3600)}小时${Math.floor((seconds % 3600) / 60)}分`;
  };

  const variantValue = (task.parameters.model_variant ?? 'tiny_long') as keyof typeof MODEL_VARIANT_LABELS;
  const variantLabel = MODEL_VARIANT_LABELS[variantValue] ?? variantValue;

  return (
    <div className="card h-full flex flex-col hover:shadow-lg transition-shadow">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between mb-4">
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-gray-900 mb-2 break-words">
            {task.input_file_name}
          </h3>
          <div className="text-xs text-gray-500">
            创建时间: {new Date(task.created_at).toLocaleString('zh-CN')}
          </div>
        </div>
        <div className="flex-shrink-0">{getStatusBadge(task.status)}</div>
      </div>

      {task.status === TaskStatus.COMPLETED && (
        <div className="mb-4">
          <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden border border-gray-200">
            <img
              src={tasksApi.getPreviewUrl(task.id)}
              alt={`预览 - ${task.input_file_name}`}
              className="w-full h-full object-cover"
              loading="lazy"
              onError={(event) => {
                (event.target as HTMLImageElement).style.display = 'none';
              }}
            />
          </div>
        </div>
      )}

      {/* 进度条 */}
      {task.status === TaskStatus.PROCESSING && (
        <div className="mb-4">
          <ProgressBar
            progress={task.progress}
            processedFrames={task.processed_frames}
            totalFrames={task.total_frames}
          />
          <div className="text-sm text-gray-600 mt-2">
            预计剩余时间: {formatTime(task.estimated_time_remaining)}
          </div>
        </div>
      )}

      {/* 错误信息 */}
      {task.status === TaskStatus.FAILED && task.error_message && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg">
          <div className="text-sm text-red-800">{task.error_message}</div>
        </div>
      )}

      {/* 参数信息 */}
      <div className="mb-4 text-sm text-gray-600 bg-gray-50 rounded-lg p-3 grid grid-cols-2 gap-2">
        <div>
          <span className="text-xs text-gray-500">模型</span>
          <div className="font-medium text-gray-900">{variantLabel}</div>
        </div>
        <div>
          <span className="text-xs text-gray-500">超分倍数</span>
          <div className="font-medium text-gray-900">{task.parameters.scale}x</div>
        </div>
        <div>
          <span className="text-xs text-gray-500">稀疏比率</span>
          <div className="font-medium text-gray-900">{task.parameters.sparse_ratio}</div>
        </div>
        <div>
          <span className="text-xs text-gray-500">局部范围</span>
          <div className="font-medium text-gray-900">{task.parameters.local_range}</div>
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="mt-auto flex flex-col gap-2.5 sm:flex-row">
        <Link to={`/tasks/${task.id}`} className="btn btn-primary flex-1 text-center">
          查看详情
        </Link>
        {task.status === TaskStatus.COMPLETED && (
          <a
            href={tasksApi.getResultUrl(task.id)}
            download
            className="btn btn-secondary flex-1 text-center"
          >
            下载结果
          </a>
        )}
        <button
          onClick={handleDelete}
          disabled={deleteMutation.isPending}
          className="btn btn-danger sm:w-auto"
        >
          删除
        </button>
      </div>
    </div>
  );
}
