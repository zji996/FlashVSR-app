/**
 * 主页
 */

import UploadForm from '../components/UploadForm';
import TaskList from '../components/TaskList';

export default function HomePage() {
  return (
    <div className="space-y-12">
      <UploadForm />
      <TaskList />
    </div>
  );
}

