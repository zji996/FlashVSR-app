/**
 * 任务列表组件
 */

import { useQuery } from '@tanstack/react-query';
import { tasksApi } from '../api/tasks';
import { TaskStatus } from '../types';
import TaskCard from './TaskCard';
import { useTaskFilterStore } from '../stores/useTaskFilters';

export default function TaskList() {
  const { page, statusFilter, setPage, setStatusFilter } = useTaskFilterStore();

  const { data, isLoading, error } = useQuery({
    queryKey: ['tasks', page, statusFilter],
    queryFn: () => tasksApi.getTasks(page, 20, statusFilter || undefined),
    refetchInterval: 3000, // 每3秒刷新一次
  });

  if (isLoading) {
    return (
      <div className="text-center py-12">
        <div className="text-gray-600">加载中...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-12">
        <div className="text-red-600">加载失败: {(error as Error).message}</div>
      </div>
    );
  }

  const totalPages = Math.max(1, Math.ceil((data?.total || 0) / 20));

  return (
    <div className="max-w-6xl mx-auto">
      <div className="flex justify-between items-center mb-6">
        <h2 className="text-2xl font-bold">任务列表</h2>

        {/* 状态筛选 */}
        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as TaskStatus | '')}
          className="input w-auto"
        >
          <option value="">全部状态</option>
          <option value={TaskStatus.PENDING}>等待中</option>
          <option value={TaskStatus.PROCESSING}>处理中</option>
          <option value={TaskStatus.COMPLETED}>已完成</option>
          <option value={TaskStatus.FAILED}>失败</option>
        </select>
      </div>

      {/* 任务列表 */}
      {data?.tasks && data.tasks.length > 0 ? (
        <div className="space-y-4">
          {data.tasks.map((task) => (
            <TaskCard key={task.id} task={task} />
          ))}
        </div>
      ) : (
        <div className="card text-center py-12">
          <div className="text-gray-500">暂无任务</div>
        </div>
      )}

      {/* 分页 */}
      {totalPages > 1 && (
        <div className="flex justify-center items-center gap-2 mt-8">
          <button
            onClick={() => setPage(Math.max(1, page - 1))}
            disabled={page === 1}
            className="btn btn-secondary"
          >
            上一页
          </button>
          <span className="text-gray-700">
            第 {page} / {totalPages} 页
          </span>
          <button
            onClick={() => setPage(Math.min(totalPages, page + 1))}
            disabled={page === totalPages}
            className="btn btn-secondary"
          >
            下一页
          </button>
        </div>
      )}
    </div>
  );
}
