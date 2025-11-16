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
      <nav className="sticky top-0 z-50 bg-white shadow-md border-b border-gray-200">
        <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-4">
            <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              {/* 左侧：标题和导航 */}
              <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between lg:justify-start lg:gap-6">
                <div className="flex-shrink-0">
                  <h1 className="text-2xl font-bold text-primary-600">
                    ⚡ FlashVSR
                  </h1>
                </div>
                <div className="flex items-center gap-3">
                  {navLinks.map((item) => (
                    <NavLink
                      key={item.to}
                      to={item.to}
                      className={({ isActive }) =>
                        `inline-flex items-center justify-center rounded-lg px-5 py-2.5 text-sm font-semibold transition-all duration-200 ${
                          isActive
                            ? 'bg-primary-600 text-white shadow-lg hover:bg-primary-700 hover:shadow-xl'
                            : 'bg-gray-100 text-gray-700 hover:bg-gray-200 hover:shadow-md'
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
                <div className="flex flex-wrap items-center gap-3 text-sm">
                  {/* GPU 状态 */}
                  <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium ${
                    systemStatusForDisplay.gpu_available 
                      ? 'bg-green-100 text-green-700 border border-green-300' 
                      : 'bg-red-100 text-red-700 border border-red-300'
                  }`}>
                    <div
                      className={`w-2.5 h-2.5 rounded-full flex-shrink-0 ${
                        systemStatusForDisplay.gpu_available ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                      }`}
                    />
                    <span>
                      {systemStatusForDisplay.gpu_available
                        ? `${systemStatusForDisplay.gpu_info?.name || 'GPU'}${
                            systemStatusForDisplay.gpu_info?.count && systemStatusForDisplay.gpu_info.count > 1
                              ? ` ×${systemStatusForDisplay.gpu_info.count}`
                              : ''
                          }`
                        : 'GPU 不可用'}
                    </span>
                  </div>

                  {/* 任务状态 */}
                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-100 text-blue-700 border border-blue-300 font-medium">
                    <span>📋</span>
                    <span>
                      {systemStatusForDisplay.tasks.processing} 处理中 / {systemStatusForDisplay.tasks.pending} 等待中
                    </span>
                  </div>

                  {/* 模型状态 */}
                  {systemStatusForDisplay.flashvsr && (
                    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg font-medium ${
                      systemStatusForDisplay.flashvsr.weights_ready
                        ? 'bg-indigo-100 text-indigo-700 border border-indigo-300'
                        : 'bg-yellow-100 text-yellow-700 border border-yellow-300'
                    }`}>
                      <span>{systemStatusForDisplay.flashvsr.weights_ready ? '✓' : '⚠'}</span>
                      <span>
                        FlashVSR {systemStatusForDisplay.flashvsr.version}
                      </span>
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
