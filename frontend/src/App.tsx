import { Routes, Route } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { systemApi } from './api/system';
import HomePage from './pages/HomePage';
import TaskDetailPage from './pages/TaskDetailPage';

function App() {
  const { data: systemStatus } = useQuery({
    queryKey: ['system-status'],
    queryFn: systemApi.getStatus,
    refetchInterval: 10000, // 每10秒刷新一次
  });

  return (
    <div className="min-h-screen bg-gray-50">
      {/* 导航栏 */}
      <nav className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <h1 className="text-2xl font-bold text-primary-600">
                ⚡ FlashVSR
              </h1>
              <span className="ml-3 text-sm text-gray-500">
                视频超分辨率处理平台
              </span>
            </div>

            {/* 系统状态 */}
            {systemStatus && (
              <div className="flex items-center gap-6 text-sm">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-2 h-2 rounded-full ${
                      systemStatus.gpu_available ? 'bg-green-500' : 'bg-red-500'
                    }`}
                  />
                  <span className="text-gray-700">
                    {systemStatus.gpu_available
                      ? `GPU: ${systemStatus.gpu_info?.name || 'Available'}`
                      : 'GPU: 不可用'}
                  </span>
                </div>
                <div className="text-gray-700">
                  任务: {systemStatus.tasks.processing} 处理中 /{' '}
                  {systemStatus.tasks.pending} 等待中
                </div>
              </div>
            )}
          </div>
        </div>
      </nav>

      {/* 主内容 */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/tasks/:taskId" element={<TaskDetailPage />} />
        </Routes>
      </main>

      {/* 页脚 */}
      <footer className="bg-white border-t border-gray-200 mt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
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
