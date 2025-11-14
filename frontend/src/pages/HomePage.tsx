/**
 * 主页
 */

import { Link } from 'react-router-dom';
import UploadForm from '../components/UploadForm';

export default function HomePage() {
  const handleScrollToUpload = () => {
    const target = document.getElementById('upload-section');
    if (target) {
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  return (
    <div className="space-y-8">
      <section>
        <div className="rounded-2xl bg-gradient-to-r from-primary-600 to-indigo-500 px-6 py-8 sm:px-10 sm:py-12 text-white shadow-lg">
          <div className="flex flex-col gap-6 md:flex-row md:items-center md:justify-between">
            <div className="space-y-3 md:max-w-xl">
              <h2 className="text-2xl sm:text-3xl font-bold tracking-tight">
                一站式视频超分平台
              </h2>
              <p className="text-sm sm:text-base text-primary-50/90">
                上传原始视频，选择预处理宽度和超分倍数，即可在后台自动完成预处理、推理和合并导出。
              </p>
              <div className="flex flex-wrap gap-3">
                <button
                  type="button"
                  onClick={handleScrollToUpload}
                  className="btn btn-primary bg-white text-primary-700 hover:bg-primary-50 border-0"
                >
                  新建任务
                </button>
                <Link
                  to="/tasks"
                  className="btn btn-secondary bg-transparent border border-white/70 text-white hover:bg-white/10"
                >
                  查看任务列表
                </Link>
              </div>
            </div>
            <div className="grid grid-cols-1 gap-3 text-sm sm:grid-cols-3 lg:max-w-lg">
              <div className="rounded-xl bg-white/10 p-4 backdrop-blur">
                <div className="text-xs uppercase tracking-wide text-primary-100/70 mb-1.5">
                  步骤 1
                </div>
                <div className="font-medium">
                  上传或拖拽视频文件
                </div>
              </div>
              <div className="rounded-xl bg-white/10 p-4 backdrop-blur">
                <div className="text-xs uppercase tracking-wide text-primary-100/70 mb-1.5">
                  步骤 2
                </div>
                <div className="font-medium">
                  选择预处理宽度与倍数
                </div>
              </div>
              <div className="rounded-xl bg-white/10 p-4 backdrop-blur">
                <div className="text-xs uppercase tracking-wide text-primary-100/70 mb-1.5">
                  步骤 3
                </div>
                <div className="font-medium">
                  在任务页查看进度并下载
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="upload-section">
        <UploadForm />
      </section>

      <section>
        <div className="grid gap-5 md:grid-cols-3">
          <div className="card h-full">
            <h3 className="text-base font-semibold text-gray-900 mb-2">推荐配置</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              一般场景推荐预处理宽度 960、超分倍数 2.0，可在画质与速度之间取得较好平衡；长视频可优先使用"快速出图"预设。
            </p>
          </div>
          <div className="card h-full">
            <h3 className="text-base font-semibold text-gray-900 mb-2">长视频体验</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              后端采用分片导出和流式缓冲，超长视频中途中断时会尽量保留已完成片段，可在任务详情页查看部分结果。
            </p>
          </div>
          <div className="card h-full">
            <h3 className="text-base font-semibold text-gray-900 mb-2">任务管理</h3>
            <p className="text-sm text-gray-600 leading-relaxed">
              所有任务都集中在"任务列表"页面，可按状态筛选、分页浏览并随时删除，适合批量处理多个视频。
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
