/**
 * 主页
 */

import { Link } from 'react-router-dom';
import UploadForm from '../components/UploadForm';

export default function HomePage() {
  return (
    <div className="min-h-screen bg-slate-50 py-10 sm:py-12">
      <div className="space-y-12 max-w-7xl mx-auto">
        <UploadForm />
        <div className="card flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900">需要查看任务进度？</h2>
            <p className="text-sm text-gray-500 mt-1">
              任务列表现已独立为单独页面，支持分页、状态筛选与实时刷新。
            </p>
          </div>
          <Link
            to="/tasks"
            className="btn btn-primary text-center"
          >
            打开任务列表
          </Link>
        </div>
      </div>
    </div>
  );
}
