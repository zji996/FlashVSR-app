/**
 * 任务列表页面
 */

import { Link } from 'react-router-dom';
import TaskList from '../components/TaskList';

export default function TasksPage() {
  return (
    <div className="py-8 sm:py-10">
      <div className="mb-6">
        <Link
          to="/"
          className="inline-flex items-center gap-1 text-sm text-gray-600 hover:text-primary-600 transition-colors"
        >
          <svg
            className="w-4 h-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M15 19l-7-7 7-7"
            />
          </svg>
          返回新建任务
        </Link>
      </div>
      <TaskList />
    </div>
  );
}

