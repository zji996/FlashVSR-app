/**
 * 任务详情页面
 */

import { useParams, Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { tasksApi } from '../api/tasks';
import { MODEL_VARIANT_LABELS, TaskStatus } from '../types';
import ProgressBar from '../components/ProgressBar';

export default function TaskDetailPage() {
  const { taskId } = useParams<{ taskId: string }>();

  const { data: task, isLoading } = useQuery({
    queryKey: ['task', taskId],
    queryFn: () => tasksApi.getTask(taskId!),
    enabled: !!taskId,
    refetchInterval: (query) => {
      const task = query.state.data;
      // 如果任务正在处理，每2秒刷新一次
      return task?.status === TaskStatus.PROCESSING ? 2000 : false;
    },
  });

  if (isLoading) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="card">
          <div className="text-center py-12 text-gray-600">加载中...</div>
        </div>
      </div>
    );
  }

  if (!task) {
    return (
      <div className="max-w-4xl mx-auto">
        <div className="card">
          <div className="text-center py-12 text-red-600">任务不存在</div>
        </div>
      </div>
    );
  }

  const variantValue = (task.parameters.model_variant ?? 'tiny') as keyof typeof MODEL_VARIANT_LABELS;
  const variantLabel = MODEL_VARIANT_LABELS[variantValue] ?? variantValue;

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
      <span className={`px-4 py-2 rounded-full text-base font-medium ${badges[status]}`}>
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

  return (
    <div className="max-w-4xl mx-auto">
      <Link to="/" className="inline-block mb-6 text-primary-600 hover:text-primary-700">
        ← 返回任务列表
      </Link>

      <div className="card">
        {/* 标题和状态 */}
        <div className="flex justify-between items-start mb-6">
          <div>
            <h1 className="text-3xl font-bold mb-2">{task.input_file_name}</h1>
            <div className="text-gray-500">
              创建时间: {new Date(task.created_at).toLocaleString('zh-CN')}
            </div>
          </div>
          <div>{getStatusBadge(task.status)}</div>
        </div>

        {/* 进度信息 */}
        {task.status === TaskStatus.PROCESSING && (
          <div className="mb-8 p-6 bg-blue-50 rounded-lg">
            <h3 className="text-lg font-semibold mb-4">处理进度</h3>
            <ProgressBar
              progress={task.progress}
              processedFrames={task.processed_frames}
              totalFrames={task.total_frames}
            />
            <div className="mt-4 text-gray-700">
              预计剩余时间: <span className="font-semibold">{formatTime(task.estimated_time_remaining)}</span>
            </div>
          </div>
        )}

        {/* 错误信息 */}
        {task.status === TaskStatus.FAILED && task.error_message && (
          <div className="mb-8 p-6 bg-red-50 border border-red-200 rounded-lg">
            <h3 className="text-lg font-semibold text-red-800 mb-2">错误信息</h3>
            <div className="text-red-700">{task.error_message}</div>
          </div>
        )}

        {/* 处理参数 */}
        <div className="mb-8">
          <h3 className="text-lg font-semibold mb-4">处理参数</h3>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600">模型变体</div>
              <div className="text-xl font-semibold">{variantLabel}</div>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600">超分倍数</div>
              <div className="text-xl font-semibold">{task.parameters.scale}x</div>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600">稀疏比率</div>
              <div className="text-xl font-semibold">{task.parameters.sparse_ratio}</div>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600">局部范围</div>
              <div className="text-xl font-semibold">{task.parameters.local_range}</div>
            </div>
            <div className="p-4 bg-gray-50 rounded-lg">
              <div className="text-sm text-gray-600">随机种子</div>
              <div className="text-xl font-semibold">{task.parameters.seed}</div>
            </div>
          </div>
        </div>

        {/* 视频信息 */}
        {task.video_info && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold mb-4">视频信息</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">分辨率</div>
                <div className="text-lg font-semibold">
                  {task.video_info.width && task.video_info.height
                    ? `${task.video_info.width} × ${task.video_info.height}`
                    : '--'}
                </div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">总帧数</div>
                <div className="text-lg font-semibold">
                  {task.video_info.total_frames ?? '--'}
                </div>
              </div>
              <div className="p-4 bg-gray-50 rounded-lg">
                <div className="text-sm text-gray-600">帧率</div>
                <div className="text-lg font-semibold">
                  {task.video_info.fps ? `${task.video_info.fps} fps` : '--'}
                </div>
              </div>
              {task.video_info.duration && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">时长</div>
                  <div className="text-lg font-semibold">
                    {task.video_info.duration.toFixed(1)}秒
                  </div>
                </div>
              )}
              {task.video_info.processing_time && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">总处理时间</div>
                  <div className="text-lg font-semibold">
                    {task.video_info.processing_time.toFixed(1)}秒
                  </div>
                </div>
              )}
              {task.video_info.inference_time && (
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm text-gray-600">推理时间</div>
                  <div className="text-lg font-semibold">
                    {task.video_info.inference_time.toFixed(1)}秒
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* 视频预览 */}
        {task.status === TaskStatus.COMPLETED && task.output_file_name && (
          <div className="mb-8">
            <h3 className="text-lg font-semibold mb-4">结果预览</h3>
            <div className="aspect-video bg-black rounded-lg overflow-hidden">
              <video
                src={tasksApi.getResultUrl(task.id)}
                controls
                className="w-full h-full"
              >
                您的浏览器不支持视频播放
              </video>
            </div>
          </div>
        )}

        {/* 操作按钮 */}
        <div className="flex gap-4">
          {task.status === TaskStatus.COMPLETED && (
            <>
              <a
                href={tasksApi.getResultUrl(task.id)}
                download
                className="btn btn-primary flex-1"
              >
                下载处理结果
              </a>
              <a
                href={tasksApi.getInputUrl(task.id)}
                download
                className="btn btn-secondary"
              >
                下载原始视频
              </a>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
