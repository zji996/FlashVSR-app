import { Routes, Route, NavLink } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { systemApi } from './api/system';
import HomePage from './pages/HomePage';
import TaskDetailPage from './pages/TaskDetailPage';
import TasksPage from './pages/TasksPage';
import type { SystemStatus } from './types';

const numericTaskKeys: Array<keyof SystemStatus['tasks']> = [
  'total',
  'pending',
  'processing',
  'completed',
  'failed',
];

const isSystemStatus = (value: unknown): value is SystemStatus => {
  if (!value || typeof value !== 'object') {
    return false;
  }
  const candidate = value as SystemStatus;
  if (typeof candidate.gpu_available !== 'boolean') {
    return false;
  }
  const tasks = candidate.tasks;
  if (!tasks || typeof tasks !== 'object') {
    return false;
  }
  return numericTaskKeys.every((key) => typeof tasks[key] === 'number');
};

function App() {
  const { data: systemStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: systemApi.getStatus,
    refetchInterval: 10000, // 每10秒刷新一次
  });

  const systemStatusForDisplay = isSystemStatus(systemStatus) ? systemStatus : undefined;
  const navLinks = [
    { to: '/', label: '新建任务' },
    { to: '/tasks', label: '任务列表' },
  ];

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 导航栏 */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              {/* 左侧：标题和导航 */}
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between lg:justify-start">
                <div className="flex-shrink-0">
                  <h1 className="text-2xl font-bold text-primary-600">
                    ⚡ FlashVSR
                  </h1>
                  <span className="text-xs text-gray-500">
                    视频超分辨率处理平台
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  {navLinks.map((item) => (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      className={({ isActive }) =>
                        `inline-flex items-center rounded-full px-4 py-2 text-sm font-medium transition ${
                          isActive
                            ? 'bg-primary-600 text-white shadow-sm'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                        }`
                      }
                    >
                      {item.label}
                    </NavLink>
                  ))}
                </div>
              </div>

              {/* 右侧：系统状态 */}
              {systemStatusForDisplay && (
                <div className="flex flex-col gap-2 text-xs text-gray-600 lg:items-end">
                  <div className="flex items-center gap-2">
                    <div
                      className={`w-2 h-2 rounded-full flex-shrink-0 ${
                        systemStatusForDisplay.gpu_available ? 'bg-green-500' : 'bg-red-500'
                      }`}
                    />
                    <span className="truncate">
                      {systemStatusForDisplay.gpu_available
                        ? `GPU: ${systemStatusForDisplay.gpu_info?.name || 'Available'}`
                        : 'GPU: 不可用'}
                    </span>
                  </div>
                  <div className="flex items-center gap-3">
                    <span>
                      任务: {systemStatusForDisplay.tasks.processing} 处理中 / {systemStatusForDisplay.tasks.pending} 等待中
                    </span>
                  </div>
                  {systemStatusForDisplay.flashvsr && (
                    <div>
                      FlashVSR {systemStatusForDisplay.flashvsr.version} · 模型
                      {systemStatusForDisplay.flashvsr.weights_ready ? '已就绪' : '未完全就绪'}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </nav>

      {/* 主内容 */}
      <main className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/tasks" element={<TasksPage />} />
          <Route path="/tasks/:taskId" element={<TaskDetailPage />} />
        </Routes>
      </main>

      {/* 页脚 */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="text-center text-gray-500 text-sm">
            <p>
              基于{' '}
              <a
                href="https://github.com/OpenImagingLab/FlashVSR"
                target="_blank"
                rel="noopener noreferrer"
                className="text-primary-600 hover:text-primary-700"
              >
                FlashVSR
              </a>{' '}
              开发
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
