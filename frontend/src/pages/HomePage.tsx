/**
 * 主页
 */

import UploadForm from '../components/UploadForm';

export default function HomePage() {
  return (
    <div className="space-y-8">
      <section>
        <div className="rounded-2xl bg-gradient-to-r from-primary-600 to-indigo-500 px-6 py-16 sm:px-10 sm:py-20 text-white shadow-2xl">
          <div className="max-w-3xl mx-auto text-center space-y-6">
            <h2 className="text-3xl sm:text-4xl lg:text-5xl font-bold tracking-tight">
              一站式视频超分平台
            </h2>
            <p className="text-base sm:text-lg text-primary-50/90 max-w-2xl mx-auto">
              上传原始视频，选择预处理宽度和超分倍数，即可在后台自动完成预处理、推理和合并导出。
            </p>
          </div>
        </div>
      </section>

      <section>
        <div className="text-center mb-8">
          <h3 className="text-xl sm:text-2xl font-bold text-gray-900 mb-2">使用流程</h3>
          <p className="text-sm text-gray-600">三步即可完成视频超分处理</p>
        </div>
        <div className="grid grid-cols-1 gap-6 sm:grid-cols-3">
          <div className="rounded-xl bg-white border border-gray-200 p-6 text-center shadow-sm hover:shadow-md transition-shadow">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary-100 text-primary-600 font-bold text-lg mb-4">
              1
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">上传或拖拽视频文件</h4>
            <p className="text-sm text-gray-600">支持常见视频格式，自动转码处理</p>
          </div>
          <div className="rounded-xl bg-white border border-gray-200 p-6 text-center shadow-sm hover:shadow-md transition-shadow">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary-100 text-primary-600 font-bold text-lg mb-4">
              2
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">选择预处理宽度与倍数</h4>
            <p className="text-sm text-gray-600">灵活配置分辨率与超分比例</p>
          </div>
          <div className="rounded-xl bg-white border border-gray-200 p-6 text-center shadow-sm hover:shadow-md transition-shadow">
            <div className="inline-flex items-center justify-center w-12 h-12 rounded-full bg-primary-100 text-primary-600 font-bold text-lg mb-4">
              3
            </div>
            <h4 className="font-semibold text-gray-900 mb-2">在任务页查看进度并下载</h4>
            <p className="text-sm text-gray-600">实时监控处理进度，完成后即可下载</p>
          </div>
        </div>
      </section>

      <section id="upload-section">
        <UploadForm />
      </section>
    </div>
  );
}
