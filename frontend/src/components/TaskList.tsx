/**
 * 任务列表组件（支持分页 + 状态筛选 + 自适应布局）
 */

import { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { tasksApi } from '../api/tasks';
import { TaskStatus } from '../types';
import TaskCard from './TaskCard';
import { useTaskFilterStore } from '../stores/useTaskFilters';

const PAGE_SIZE_OPTIONS = [6, 8, 12, 20];
const STATUS_FILTERS: Array<{ label: string; value: TaskStatus | '' }> = [
  { label: '全部', value: '' },
  { label: '等待中', value: TaskStatus.PENDING },
  { label: '处理中', value: TaskStatus.PROCESSING },
  { label: '已完成', value: TaskStatus.COMPLETED },
  { label: '失败', value: TaskStatus.FAILED },
];

export default function TaskList() {
  const { page, pageSize, statusFilter, setPage, setPageSize, setStatusFilter } = useTaskFilterStore();
  const [jumpInput, setJumpInput] = useState('');

  const { data, isLoading, error } = useQuery({
    queryKey: ['tasks', page, pageSize, statusFilter],
    queryFn: () => tasksApi.getTasks(page, pageSize, statusFilter || undefined),
    refetchInterval: 3000,
  });

  useEffect(() => {
    setJumpInput('');
  }, [page]);

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

  const totalItems = data?.total ?? 0;
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize));
  const rangeStart = totalItems === 0 ? 0 : (page - 1) * pageSize + 1;
  const rangeEnd = Math.min(totalItems, page * pageSize);

  const handleJumpSubmit = (event: React.FormEvent) => {
    event.preventDefault();
    if (!jumpInput) {
      return;
    }
    const parsed = Number(jumpInput);
    if (Number.isNaN(parsed)) {
      return;
    }
    const clamped = Math.min(totalPages, Math.max(1, Math.trunc(parsed)));
    if (clamped !== page) {
      setPage(clamped);
    }
  };

  return (
    <div className="space-y-6">
      <div className="card">
        <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-2xl font-bold text-gray-900">任务列表</h2>
            <p className="text-sm text-gray-600 mt-1.5">
              自动每 3 秒刷新，可按状态筛选并切换每页数量。
            </p>
          </div>
          <label className="text-sm text-gray-700 flex items-center gap-2">
            <span className="font-medium">每页</span>
            <select
              value={pageSize}
              onChange={(e) => setPageSize(Number(e.target.value))}
              className="input w-20"
            >
              {PAGE_SIZE_OPTIONS.map((size) => (
                <option key={size} value={size}>
                  {size}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="mt-5 flex flex-wrap gap-2">
          {STATUS_FILTERS.map((option) => {
            const active = statusFilter === option.value;
            return (
              <button
                key={option.value || 'all'}
                type="button"
                onClick={() => setStatusFilter(option.value)}
                className={`rounded-full px-4 py-1.5 text-sm transition ${
                  active ? 'bg-primary-600 text-white shadow-sm' : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {option.label}
              </button>
            );
          })}
        </div>
      </div>

      {data?.tasks && data.tasks.length > 0 ? (
        <div className="grid grid-cols-1 gap-5 lg:grid-cols-2">
          {data.tasks.map((task) => (
            <TaskCard key={task.id} task={task} />
          ))}
        </div>
      ) : (
        <div className="card text-center py-16">
          <div className="text-gray-400 text-lg">暂无任务</div>
          <p className="text-sm text-gray-500 mt-2">创建新任务后将在此显示</p>
        </div>
      )}

      {totalPages > 1 && (
        <div className="card flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
          <div className="text-sm text-gray-600 font-medium">
            显示 {rangeStart}-{rangeEnd} / 共 {totalItems} 条
          </div>
          <div className="flex flex-wrap items-center gap-2.5">
            <button
              onClick={() => setPage(1)}
              disabled={page === 1}
              className="btn btn-secondary px-3 py-2"
              title="第一页"
            >
              «
            </button>
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="btn btn-secondary px-4 py-2"
            >
              上一页
            </button>
            <span className="text-gray-700 text-sm font-medium px-2">
              第 {page} / {totalPages} 页
            </span>
            <button
              onClick={() => setPage(Math.min(totalPages, page + 1))}
              disabled={page === totalPages}
              className="btn btn-secondary px-4 py-2"
            >
              下一页
            </button>
            <button
              onClick={() => setPage(totalPages)}
              disabled={page === totalPages}
              className="btn btn-secondary px-3 py-2"
              title="最后一页"
            >
              »
            </button>
            <form className="flex items-center gap-2 ml-2" onSubmit={handleJumpSubmit}>
              <label className="text-sm text-gray-600 hidden sm:inline font-medium">跳至</label>
              <input
                type="number"
                min="1"
                max={totalPages}
                value={jumpInput}
                onChange={(e) => setJumpInput(e.target.value)}
                placeholder="页码"
                className="input w-16 text-sm"
              />
              <button type="submit" className="btn btn-primary px-3 py-2">
                Go
              </button>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
